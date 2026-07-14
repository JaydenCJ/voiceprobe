"""Deterministic mock backends with configurable latency profiles.

These backends never touch the network and never load models. They sleep
on the injected :class:`~voiceprobe.clock.Clock` for a configured latency
(base + deterministic seeded jitter), so:

- with ``MonotonicClock`` they behave like a realistic remote service and
  produce genuinely *measured* wall-clock numbers, and
- with ``SimulatedClock`` tests get exact, reproducible latency attribution.

Fixed input + fixed seed => fixed output, always.
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass
from typing import AsyncIterator

from voiceprobe.audio import AudioClip
from voiceprobe.backends.base import STTResult, VADResult
from voiceprobe.clock import Clock


def _stable_rng(seed: int, *parts: str) -> random.Random:
    """Random generator keyed by seed + string parts, stable across runs."""
    key = f"{seed}|" + "|".join(parts)
    return random.Random(zlib.crc32(key.encode("utf-8")))


@dataclass(frozen=True)
class LatencyProfile:
    """Per-stage latency parameters (milliseconds) for the mock backends.

    Defaults model a typical cloud voice stack; they are simulation inputs,
    not benchmark claims. ``jitter_ratio`` is the max relative deviation
    applied per request via seeded uniform jitter.
    """

    vad_ms: float = 60.0
    stt_ms: float = 320.0
    llm_ttft_ms: float = 480.0
    llm_token_ms: float = 24.0
    tts_ttfb_ms: float = 180.0
    tts_chunk_ms: float = 40.0
    jitter_ratio: float = 0.25

    @classmethod
    def named(cls, name: str) -> "LatencyProfile":
        profiles = {
            "fast": cls(
                vad_ms=25.0,
                stt_ms=140.0,
                llm_ttft_ms=220.0,
                llm_token_ms=12.0,
                tts_ttfb_ms=90.0,
                tts_chunk_ms=20.0,
            ),
            "typical": cls(),
            "slow": cls(
                vad_ms=120.0,
                stt_ms=650.0,
                llm_ttft_ms=1100.0,
                llm_token_ms=45.0,
                tts_ttfb_ms=420.0,
                tts_chunk_ms=80.0,
            ),
        }
        if name not in profiles:
            raise ValueError(
                f"unknown latency profile {name!r}; choose from {sorted(profiles)}"
            )
        return profiles[name]

    def jittered(self, base_ms: float, rng: random.Random) -> float:
        if base_ms <= 0:
            return 0.0
        spread = base_ms * self.jitter_ratio
        return max(1.0, base_ms + rng.uniform(-spread, spread))


class MockVAD:
    """Simulated VAD: sleeps its configured decision latency."""

    def __init__(self, clock: Clock, profile: LatencyProfile, seed: int = 0) -> None:
        self._clock = clock
        self._profile = profile
        self._seed = seed
        self.name = "mock-vad"

    async def detect(self, audio: AudioClip) -> VADResult:
        rng = _stable_rng(self._seed, "vad", str(len(audio.pcm)))
        latency_ms = self._profile.jittered(self._profile.vad_ms, rng)
        await self._clock.sleep(latency_ms / 1000.0)
        return VADResult(
            speech_detected=audio.num_samples > 0,
            speech_end_ms=audio.duration_ms,
            detail={"simulated_latency_ms": round(latency_ms, 1)},
        )


class MockSTT:
    """Simulated STT: echoes the scenario transcript deterministically."""

    def __init__(self, clock: Clock, profile: LatencyProfile, seed: int = 0) -> None:
        self._clock = clock
        self._profile = profile
        self._seed = seed
        self.name = "mock-stt"

    async def transcribe(self, audio: AudioClip) -> STTResult:
        rng = _stable_rng(self._seed, "stt", str(len(audio.pcm)), audio.transcript_hint or "")
        # Longer utterances cost more, like a real batch STT endpoint.
        base = self._profile.stt_ms + 0.04 * audio.duration_ms
        latency_ms = self._profile.jittered(base, rng)
        await self._clock.sleep(latency_ms / 1000.0)
        if audio.transcript_hint:
            text = audio.transcript_hint
        else:
            text = f"utterance of {audio.duration_ms / 1000.0:.1f} seconds"
        return STTResult(text=text)


class MockLLM:
    """Simulated LLM: streams a deterministic reply with TTFT + token delays."""

    def __init__(self, clock: Clock, profile: LatencyProfile, seed: int = 0) -> None:
        self._clock = clock
        self._profile = profile
        self._seed = seed
        self.name = "mock-llm"

    def _compose_reply(self, transcript: str, rng: random.Random) -> str:
        openers = [
            "Sure, let me help with that.",
            "Thanks for the details.",
            "I can take care of that.",
            "Understood.",
        ]
        opener = openers[rng.randrange(len(openers))]
        return f"{opener} Regarding \"{transcript.strip()}\": here is what I found for you."

    async def stream_reply(
        self, transcript: str, history: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        rng = _stable_rng(self._seed, "llm", transcript, str(len(history)))
        ttft_ms = self._profile.jittered(self._profile.llm_ttft_ms, rng)
        reply = self._compose_reply(transcript, rng)
        tokens = reply.split(" ")
        await self._clock.sleep(ttft_ms / 1000.0)
        for i, token in enumerate(tokens):
            if i > 0:
                token_ms = self._profile.jittered(self._profile.llm_token_ms, rng)
                await self._clock.sleep(token_ms / 1000.0)
            yield token + (" " if i < len(tokens) - 1 else "")


class MockTTS:
    """Simulated TTS: streams audio chunks with TTFB + per-chunk delays."""

    def __init__(self, clock: Clock, profile: LatencyProfile, seed: int = 0) -> None:
        self._clock = clock
        self._profile = profile
        self._seed = seed
        self.name = "mock-tts"

    async def stream_speech(self, text: str) -> AsyncIterator[bytes]:
        rng = _stable_rng(self._seed, "tts", text)
        ttfb_ms = self._profile.jittered(self._profile.tts_ttfb_ms, rng)
        await self._clock.sleep(ttfb_ms / 1000.0)
        # One 40 ms PCM chunk per ~4 characters of text, minimum 3 chunks.
        n_chunks = max(3, len(text) // 4)
        chunk = b"\x00\x00" * 640  # 40 ms of 16 kHz mono 16-bit silence
        yield chunk
        for _ in range(n_chunks - 1):
            chunk_ms = self._profile.jittered(self._profile.tts_chunk_ms, rng)
            await self._clock.sleep(chunk_ms / 1000.0)
            yield chunk
