"""Audio utilities: 16-bit mono PCM clips, WAV I/O and speech-like synthesis.

All audio inside voiceprobe is represented as :class:`AudioClip` — raw
little-endian 16-bit mono PCM plus a sample rate. Only the Python stdlib
(``wave``, ``audioop``-free math) is used.
"""

from __future__ import annotations

import math
import struct
import wave
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

DEFAULT_SAMPLE_RATE = 16_000


class AudioError(Exception):
    """Raised when an audio file cannot be read or has an unsupported format."""


@dataclass(frozen=True)
class AudioClip:
    """Raw 16-bit mono PCM audio.

    ``transcript_hint`` carries the scenario-provided ground-truth text for
    synthesized utterances; mock STT backends echo it so the rest of the
    pipeline sees realistic transcripts. Real backends ignore it.
    """

    pcm: bytes
    sample_rate: int = DEFAULT_SAMPLE_RATE
    transcript_hint: str | None = field(default=None, compare=False)

    @property
    def num_samples(self) -> int:
        return len(self.pcm) // 2

    @property
    def duration_ms(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return self.num_samples * 1000.0 / self.sample_rate

    def with_hint(self, hint: str | None) -> "AudioClip":
        return AudioClip(pcm=self.pcm, sample_rate=self.sample_rate, transcript_hint=hint)


def read_wav(path: str | Path) -> AudioClip:
    """Read a PCM WAV file into an :class:`AudioClip`.

    Accepts 16-bit PCM; multi-channel input is downmixed to mono by taking
    the first channel. Raises :class:`AudioError` with a human-readable
    message on unsupported formats.
    """
    p = Path(path)
    if not p.exists():
        raise AudioError(f"audio file not found: {p}")
    try:
        with wave.open(str(p), "rb") as wf:
            sample_width = wf.getsampwidth()
            channels = wf.getnchannels()
            rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
    except wave.Error as exc:
        raise AudioError(f"not a valid WAV file: {p} ({exc})") from exc
    if sample_width != 2:
        raise AudioError(
            f"unsupported WAV sample width in {p}: {sample_width * 8}-bit (expected 16-bit PCM)"
        )
    if channels < 1:
        raise AudioError(f"WAV file has no channels: {p}")
    if channels > 1:
        # Keep the first channel only.
        mono = bytearray()
        frame_size = 2 * channels
        for i in range(0, len(frames) - frame_size + 1, frame_size):
            mono.extend(frames[i : i + 2])
        frames = bytes(mono)
    return AudioClip(pcm=frames, sample_rate=rate)


def write_wav(path: "str | Path | BinaryIO", clip: AudioClip) -> None:
    """Write an :class:`AudioClip` as 16-bit mono PCM WAV to a path or file object."""
    target = path if hasattr(path, "write") else str(path)
    with wave.open(target, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(clip.sample_rate)
        wf.writeframes(clip.pcm)


def frame_rms(pcm: bytes) -> float:
    """Root-mean-square amplitude of a 16-bit PCM frame (0.0 .. 32767.0)."""
    n = len(pcm) // 2
    if n == 0:
        return 0.0
    samples = struct.unpack(f"<{n}h", pcm[: n * 2])
    return math.sqrt(sum(s * s for s in samples) / n)


def silence(duration_ms: float, sample_rate: int = DEFAULT_SAMPLE_RATE) -> AudioClip:
    """Generate a clip of digital silence."""
    n = max(0, int(round(duration_ms * sample_rate / 1000.0)))
    return AudioClip(pcm=b"\x00\x00" * n, sample_rate=sample_rate)


def synthesize_speech_like(
    duration_ms: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    seed: str = "voiceprobe",
    trailing_silence_ms: float = 300.0,
) -> AudioClip:
    """Generate deterministic speech-shaped audio (voiced bursts + pauses).

    The waveform is a sequence of amplitude-modulated sine bursts separated
    by short gaps, followed by ``trailing_silence_ms`` of silence — enough
    structure for energy-based VAD to find a genuine end-of-speech point.
    The output depends only on the arguments, so runs are reproducible.
    """
    speech_ms = max(0.0, duration_ms - trailing_silence_ms)
    rng_state = zlib.crc32(seed.encode("utf-8"))
    samples: list[int] = []
    t_ms = 0.0
    while t_ms < speech_ms:
        # Deterministic pseudo-random burst/gap lengths and pitch.
        rng_state = zlib.crc32(struct.pack("<I", rng_state))
        burst_ms = 120.0 + (rng_state % 160)  # 120..279 ms voiced burst
        rng_state = zlib.crc32(struct.pack("<I", rng_state))
        gap_ms = 30.0 + (rng_state % 70)  # 30..99 ms gap
        rng_state = zlib.crc32(struct.pack("<I", rng_state))
        freq = 110.0 + (rng_state % 180)  # 110..289 Hz fundamental
        burst_ms = min(burst_ms, speech_ms - t_ms)
        n_burst = int(burst_ms * sample_rate / 1000.0)
        for i in range(n_burst):
            # Raised-cosine envelope keeps burst edges click-free.
            env = 0.5 * (1.0 - math.cos(2.0 * math.pi * min(1.0, i / max(1, n_burst - 1))))
            value = 0.45 * env * (
                math.sin(2.0 * math.pi * freq * i / sample_rate)
                + 0.35 * math.sin(2.0 * math.pi * freq * 2.1 * i / sample_rate)
            )
            samples.append(int(max(-1.0, min(1.0, value)) * 32767))
        t_ms += burst_ms
        if t_ms >= speech_ms:
            break
        gap_ms = min(gap_ms, speech_ms - t_ms)
        samples.extend([0] * int(gap_ms * sample_rate / 1000.0))
        t_ms += gap_ms
    samples.extend([0] * int(trailing_silence_ms * sample_rate / 1000.0))
    pcm = struct.pack(f"<{len(samples)}h", *samples) if samples else b""
    return AudioClip(pcm=pcm, sample_rate=sample_rate)


def estimate_speech_ms(text: str) -> float:
    """Estimate spoken duration of ``text`` at a typical conversational pace.

    Uses ~150 words per minute (400 ms per word) with a 600 ms floor. This
    is only used when a scenario turn provides text without audio.
    """
    words = max(1, len(text.split()))
    return max(600.0, words * 400.0)
