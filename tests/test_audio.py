"""Audio utilities: synthesis, WAV round-trips and format validation."""

from __future__ import annotations

import io
import struct
import wave

import pytest

from voiceprobe.audio import (
    AudioClip,
    AudioError,
    estimate_speech_ms,
    frame_rms,
    read_wav,
    silence,
    synthesize_speech_like,
    write_wav,
)


def test_synthesized_clip_matches_requested_duration():
    clip = synthesize_speech_like(duration_ms=2000, seed="hello")
    assert clip.duration_ms == pytest.approx(2000, abs=120)
    assert clip.sample_rate == 16_000


def test_synthesis_is_deterministic_for_same_seed():
    a = synthesize_speech_like(duration_ms=1200, seed="same words")
    b = synthesize_speech_like(duration_ms=1200, seed="same words")
    c = synthesize_speech_like(duration_ms=1200, seed="other words")
    assert a.pcm == b.pcm
    assert a.pcm != c.pcm


def test_synthesized_clip_has_speech_energy_and_silent_tail():
    clip = synthesize_speech_like(duration_ms=1500, seed="x", trailing_silence_ms=300)
    n = len(clip.pcm)
    head_rms = frame_rms(clip.pcm[: n // 4])
    tail_rms = frame_rms(clip.pcm[-3200:])  # last 100 ms at 16 kHz
    assert head_rms > 500.0
    assert tail_rms == 0.0


def test_wav_round_trip_via_path(tmp_path):
    clip = synthesize_speech_like(duration_ms=800, seed="roundtrip")
    path = tmp_path / "utterance.wav"
    write_wav(path, clip)
    loaded = read_wav(path)
    assert loaded.pcm == clip.pcm
    assert loaded.sample_rate == clip.sample_rate


def test_wav_round_trip_via_file_object():
    clip = silence(100)
    buf = io.BytesIO()
    write_wav(buf, clip)
    buf.seek(0)
    with wave.open(buf, "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == clip.sample_rate


def test_read_wav_missing_file_raises_readable_error(tmp_path):
    with pytest.raises(AudioError, match="not found"):
        read_wav(tmp_path / "nope.wav")


def test_read_wav_rejects_non_wav_bytes(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"this is definitely not RIFF data")
    with pytest.raises(AudioError, match="not a valid WAV"):
        read_wav(bad)


def test_read_wav_rejects_8_bit_samples(tmp_path):
    path = tmp_path / "8bit.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(8000)
        wf.writeframes(b"\x80" * 800)
    with pytest.raises(AudioError, match="16-bit"):
        read_wav(path)


def test_read_wav_downmixes_stereo_to_first_channel(tmp_path):
    path = tmp_path / "stereo.wav"
    left = [1000, 2000, 3000]
    right = [-1, -2, -3]
    interleaved = b"".join(struct.pack("<hh", l, r) for l, r in zip(left, right))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(interleaved)
    clip = read_wav(path)
    assert struct.unpack("<3h", clip.pcm) == (1000, 2000, 3000)


def test_frame_rms_on_known_samples():
    pcm = struct.pack("<4h", 100, -100, 100, -100)
    assert frame_rms(pcm) == pytest.approx(100.0)
    assert frame_rms(b"") == 0.0


def test_estimate_speech_ms_scales_with_words_and_has_floor():
    assert estimate_speech_ms("hi") == 600.0
    ten_words = " ".join(["word"] * 10)
    assert estimate_speech_ms(ten_words) == pytest.approx(4000.0)


def test_clip_hint_is_carried_but_not_compared():
    clip = silence(10)
    hinted = clip.with_hint("hello")
    assert hinted.transcript_hint == "hello"
    assert hinted == clip  # hint excluded from equality
    assert AudioClip(pcm=clip.pcm).duration_ms == clip.duration_ms
