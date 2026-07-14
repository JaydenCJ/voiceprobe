"""Mock backends: determinism and simulated latency behavior."""

from __future__ import annotations

import pytest

from voiceprobe.audio import synthesize_speech_like
from voiceprobe.backends.mock import LatencyProfile, MockLLM, MockSTT, MockTTS, MockVAD
from voiceprobe.clock import SimulatedClock
from conftest import run_simulated

PROFILE = LatencyProfile.named("typical")


def test_named_profiles_exist_and_unknown_rejected():
    fast = LatencyProfile.named("fast")
    slow = LatencyProfile.named("slow")
    assert fast.stt_ms < PROFILE.stt_ms < slow.stt_ms
    with pytest.raises(ValueError, match="unknown latency profile"):
        LatencyProfile.named("warp")


def test_mock_stt_echoes_transcript_hint_and_sleeps():
    audio = synthesize_speech_like(1000, seed="s").with_hint("turn on the lights")

    async def scenario(clock: SimulatedClock):
        stt = MockSTT(clock, PROFILE, seed=7)
        result = await stt.transcribe(audio)
        return result.text, clock.now()

    _, (text, elapsed) = run_simulated(scenario)
    assert text == "turn on the lights"
    # Jitter is bounded: base 320 ms + 4% of 1000 ms audio, +/- 25%.
    base = (PROFILE.stt_ms + 0.04 * audio.duration_ms) / 1000.0
    assert base * 0.74 <= elapsed <= base * 1.26


def test_mock_stt_without_hint_describes_duration():
    audio = synthesize_speech_like(2000, seed="s")

    async def scenario(clock: SimulatedClock):
        return (await MockSTT(clock, PROFILE).transcribe(audio)).text

    _, text = run_simulated(scenario)
    assert "seconds" in text


def test_mock_stt_latency_is_deterministic_across_runs():
    audio = synthesize_speech_like(800, seed="d").with_hint("same input")

    def elapsed_for_run() -> float:
        async def scenario(clock: SimulatedClock):
            await MockSTT(clock, PROFILE, seed=3).transcribe(audio)
            return clock.now()

        _, value = run_simulated(scenario)
        return value

    assert elapsed_for_run() == elapsed_for_run()


def test_mock_llm_streams_deterministic_reply_mentioning_transcript():
    async def scenario(clock: SimulatedClock):
        llm = MockLLM(clock, PROFILE, seed=1)
        tokens = [t async for t in llm.stream_reply("cancel my order", [])]
        return "".join(tokens)

    _, reply_a = run_simulated(scenario)
    _, reply_b = run_simulated(scenario)
    assert reply_a == reply_b
    assert "cancel my order" in reply_a


def test_mock_llm_ttft_precedes_token_stream():
    async def scenario(clock: SimulatedClock):
        llm = MockLLM(clock, PROFILE, seed=1)
        arrival_times = []
        async for _ in llm.stream_reply("hello", []):
            arrival_times.append(clock.now())
        return arrival_times

    _, times = run_simulated(scenario)
    assert len(times) > 3
    # First token waits for TTFT (>= 75% of base with max jitter).
    assert times[0] >= PROFILE.llm_ttft_ms / 1000.0 * 0.74
    # Later tokens arrive strictly after earlier ones.
    assert times == sorted(times)
    assert times[-1] > times[0]


def test_mock_tts_streams_chunks_after_ttfb():
    async def scenario(clock: SimulatedClock):
        tts = MockTTS(clock, PROFILE, seed=1)
        chunk_times = []
        total = 0
        async for chunk in tts.stream_speech("This is a reasonably long reply sentence."):
            chunk_times.append(clock.now())
            total += len(chunk)
        return chunk_times, total

    _, (times, total_bytes) = run_simulated(scenario)
    assert len(times) >= 3
    assert times[0] >= PROFILE.tts_ttfb_ms / 1000.0 * 0.74
    assert total_bytes > 0


def test_mock_vad_reports_full_clip_as_speech():
    audio = synthesize_speech_like(900, seed="v")

    async def scenario(clock: SimulatedClock):
        return await MockVAD(clock, PROFILE, seed=1).detect(audio)

    _, result = run_simulated(scenario)
    assert result.speech_detected
    assert result.speech_end_ms == pytest.approx(audio.duration_ms)
    assert result.detail["simulated_latency_ms"] > 0


def test_jittered_latency_stays_within_bounds_and_positive():
    profile = LatencyProfile(jitter_ratio=0.5)
    import random

    rng = random.Random(42)
    for _ in range(200):
        value = profile.jittered(100.0, rng)
        assert 50.0 <= value <= 150.0
    assert profile.jittered(0.0, rng) == 0.0
