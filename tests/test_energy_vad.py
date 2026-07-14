"""EnergyVAD: real end-of-speech detection on synthetic PCM."""

from __future__ import annotations

import asyncio
import math
import struct

import pytest

from voiceprobe.audio import AudioClip, silence, synthesize_speech_like
from voiceprobe.backends.energy import EnergyVAD


def _tone_then_silence(tone_ms: int, silence_ms: int, sample_rate: int = 16000) -> AudioClip:
    n_tone = int(tone_ms * sample_rate / 1000)
    n_sil = int(silence_ms * sample_rate / 1000)
    samples = [int(12000 * math.sin(2 * math.pi * 220 * i / sample_rate)) for i in range(n_tone)]
    samples += [0] * n_sil
    return AudioClip(pcm=struct.pack(f"<{len(samples)}h", *samples), sample_rate=sample_rate)


def test_detects_speech_end_before_trailing_silence():
    clip = _tone_then_silence(tone_ms=500, silence_ms=400)
    result = asyncio.run(EnergyVAD(frame_ms=20).detect(clip))
    assert result.speech_detected
    # Speech ends at ~500 ms; allow one frame of quantization.
    assert result.speech_end_ms == pytest.approx(500, abs=25)


def test_decision_delay_includes_hangover():
    clip = _tone_then_silence(tone_ms=400, silence_ms=500)
    vad = EnergyVAD(frame_ms=20, hangover_ms=200)
    result = asyncio.run(vad.detect(clip))
    assert result.detail["decision_ms"] == pytest.approx(result.speech_end_ms + 200, abs=1)


def test_decision_is_capped_at_clip_end():
    clip = _tone_then_silence(tone_ms=400, silence_ms=50)
    vad = EnergyVAD(frame_ms=20, hangover_ms=500)
    result = asyncio.run(vad.detect(clip))
    assert result.detail["decision_ms"] <= clip.duration_ms


def test_pure_silence_is_not_speech():
    result = asyncio.run(EnergyVAD().detect(silence(600)))
    assert not result.speech_detected
    assert result.speech_end_ms == 0.0


def test_empty_clip_is_not_speech():
    result = asyncio.run(EnergyVAD().detect(AudioClip(pcm=b"")))
    assert not result.speech_detected


def test_works_on_speech_like_synthesis():
    clip = synthesize_speech_like(duration_ms=1500, seed="vad test", trailing_silence_ms=300)
    result = asyncio.run(EnergyVAD().detect(clip))
    assert result.speech_detected
    # End of speech must land before the 300 ms silent tail finishes.
    assert result.speech_end_ms < clip.duration_ms - 200


def test_invalid_frame_size_rejected():
    with pytest.raises(ValueError):
        EnergyVAD(frame_ms=0)
