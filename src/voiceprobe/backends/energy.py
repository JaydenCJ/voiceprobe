"""Energy-based voice activity detection (real implementation, no models).

Classic frame-energy VAD: split the clip into fixed-size frames, compute
RMS per frame, mark frames above an adaptive threshold as speech, and
declare end-of-speech after ``hangover_ms`` of consecutive silence. This
is the same principle production telephony stacks used before neural VAD,
and it is fully deterministic and dependency-free.
"""

from __future__ import annotations

from voiceprobe.audio import AudioClip, frame_rms
from voiceprobe.backends.base import VADResult


class EnergyVAD:
    """Frame-RMS voice activity detector.

    ``threshold_ratio`` sets the speech threshold relative to the clip's
    peak frame RMS (adaptive, so absolute recording level does not matter).
    ``min_floor`` guards against declaring speech in an all-silence clip.
    """

    def __init__(
        self,
        frame_ms: int = 20,
        hangover_ms: int = 200,
        threshold_ratio: float = 0.1,
        min_floor: float = 120.0,
    ) -> None:
        if frame_ms <= 0:
            raise ValueError("frame_ms must be positive")
        self.frame_ms = frame_ms
        self.hangover_ms = hangover_ms
        self.threshold_ratio = threshold_ratio
        self.min_floor = min_floor
        self.name = f"energy-vad(frame={frame_ms}ms,hangover={hangover_ms}ms)"

    async def detect(self, audio: AudioClip) -> VADResult:
        frame_bytes = max(2, int(audio.sample_rate * self.frame_ms / 1000.0) * 2)
        energies: list[float] = []
        for offset in range(0, len(audio.pcm), frame_bytes):
            energies.append(frame_rms(audio.pcm[offset : offset + frame_bytes]))
        if not energies:
            return VADResult(speech_detected=False, speech_end_ms=0.0)
        peak = max(energies)
        threshold = max(self.min_floor, peak * self.threshold_ratio)
        speech_frames = [e >= threshold for e in energies]
        if not any(speech_frames):
            return VADResult(
                speech_detected=False,
                speech_end_ms=0.0,
                detail={"peak_rms": round(peak, 1), "threshold": round(threshold, 1)},
            )
        last_speech = max(i for i, is_speech in enumerate(speech_frames) if is_speech)
        speech_end_ms = (last_speech + 1) * self.frame_ms
        # End-of-speech can only be *declared* hangover_ms after it happened:
        # that algorithmic delay is inherent to any VAD and is reported so the
        # pipeline can attribute it.
        decision_ms = min(audio.duration_ms, speech_end_ms + self.hangover_ms)
        return VADResult(
            speech_detected=True,
            speech_end_ms=speech_end_ms,
            detail={
                "decision_ms": round(decision_ms, 1),
                "hangover_ms": float(self.hangover_ms),
                "peak_rms": round(peak, 1),
                "threshold": round(threshold, 1),
                "frames": float(len(energies)),
            },
        )
