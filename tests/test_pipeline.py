"""VoicePipeline latency attribution, verified with exact virtual time.

These are the core tests of the project: fixed-latency backends run under
a SimulatedClock, so every span duration and derived e2e metric must come
out exact (no tolerances beyond float rounding).
"""

from __future__ import annotations

import pytest

from voiceprobe.clock import SimulatedClock
from voiceprobe.pipeline import VoicePipeline
from voiceprobe.scenario import Scenario, Turn
from conftest import FixedLLM, FixedSTT, FixedTTS, FixedVAD, fixed_backend_set, run_simulated

APPROX = dict(rel=1e-6, abs=1e-6)


def one_turn_scenario(pause_ms: float = 0.0) -> Scenario:
    return Scenario(
        name="unit",
        turns=(Turn(user_text="I want to check my order status", pause_ms=pause_ms),),
    )


def run_call(scenario: Scenario, **backend_overrides):
    def make(clock: SimulatedClock):
        backends = fixed_backend_set(clock, **backend_overrides)
        return VoicePipeline(backends, clock).run_call(scenario, call_id=0)

    return run_simulated(make)


def test_stage_spans_have_exact_durations():
    _, result = run_call(one_turn_scenario())
    turn = result.turns[0]
    durations = {span.stage: span.duration_ms for span in turn.spans}
    assert durations["vad"] == pytest.approx(50.0, **APPROX)
    assert durations["stt"] == pytest.approx(300.0, **APPROX)
    # LLM: 400 ms TTFT + 2 inter-token gaps of 20 ms.
    assert durations["llm"] == pytest.approx(440.0, **APPROX)
    # TTS: 150 ms TTFB + 2 inter-chunk gaps of 30 ms.
    assert durations["tts"] == pytest.approx(210.0, **APPROX)


def test_stages_run_back_to_back_without_gaps():
    _, result = run_call(one_turn_scenario())
    spans = result.turns[0].spans
    assert [s.stage for s in spans] == ["vad", "stt", "llm", "tts"]
    for prev, nxt in zip(spans, spans[1:]):
        assert nxt.start == pytest.approx(prev.end, **APPROX)


def test_first_token_latency_is_vad_plus_stt_plus_ttft():
    _, result = run_call(one_turn_scenario())
    turn = result.turns[0]
    assert turn.first_token_ms == pytest.approx(50 + 300 + 400, **APPROX)
    assert turn.span("llm").detail["ttft_ms"] == pytest.approx(400.0, abs=0.01)


def test_first_audio_latency_is_full_chain_to_first_tts_byte():
    _, result = run_call(one_turn_scenario())
    turn = result.turns[0]
    # vad 50 + stt 300 + llm total 440 + tts ttfb 150 (sequential stages).
    assert turn.first_audio_ms == pytest.approx(940.0, **APPROX)
    assert turn.span("tts").detail["ttfb_ms"] == pytest.approx(150.0, abs=0.01)


def test_turn_total_covers_all_stages():
    _, result = run_call(one_turn_scenario())
    assert result.turns[0].total_ms == pytest.approx(50 + 300 + 440 + 210, **APPROX)


def test_pause_delays_turn_but_is_not_a_span():
    clock, result = run_call(one_turn_scenario(pause_ms=250.0))
    turn = result.turns[0]
    assert turn.started == pytest.approx(0.25, **APPROX)
    assert turn.total_ms == pytest.approx(1000.0, **APPROX)
    assert clock.now() == pytest.approx(1.25, **APPROX)


def test_transcript_and_reply_are_recorded():
    _, result = run_call(one_turn_scenario())
    turn = result.turns[0]
    assert turn.transcript == "I want to check my order status"
    assert turn.reply == "hello there caller"


def test_history_grows_across_turns():
    scenario = Scenario(
        name="two-turns",
        turns=(
            Turn(user_text="first question", pause_ms=0.0),
            Turn(user_text="second question", pause_ms=0.0),
        ),
    )

    captured = {}

    def make(clock: SimulatedClock):
        llm = FixedLLM(clock)
        captured["llm"] = llm
        backends = fixed_backend_set(clock, llm=llm)
        return VoicePipeline(backends, clock).run_call(scenario, call_id=0)

    _, result = run_simulated(make)
    llm = captured["llm"]
    assert llm.histories[0] == []
    assert [m["role"] for m in llm.histories[1]] == ["user", "assistant"]
    assert llm.histories[1][0]["content"] == "first question"
    assert result.turns[1].error is None


def test_vad_rejecting_audio_fails_the_turn_and_stops_the_call():
    scenario = Scenario(
        name="fail",
        turns=(
            Turn(user_text="only silence here", pause_ms=0.0),
            Turn(user_text="never reached", pause_ms=0.0),
        ),
    )

    def make(clock: SimulatedClock):
        backends = fixed_backend_set(clock, vad=FixedVAD(clock, detect_speech=False))
        return VoicePipeline(backends, clock).run_call(scenario, call_id=0)

    _, result = run_simulated(make)
    assert not result.ok
    assert "no speech detected" in result.turns[0].error
    assert result.error is not None and "turn 1" in result.error
    # The second turn is never attempted.
    assert len(result.turns) == 1
    # Only the VAD span was recorded before the failure.
    assert [s.stage for s in result.turns[0].spans] == ["vad"]


def test_result_dict_round_trip_preserves_measurements():
    from voiceprobe.pipeline import CallResult

    _, result = run_call(one_turn_scenario())
    restored = CallResult.from_dict(result.to_dict())
    assert restored.call_id == result.call_id
    assert restored.turns[0].first_audio_ms == pytest.approx(
        result.turns[0].first_audio_ms, abs=0.01
    )
    assert [s.stage for s in restored.turns[0].spans] == ["vad", "stt", "llm", "tts"]
    assert restored.turns[0].span("llm").detail["ttft_ms"] == pytest.approx(400.0, abs=0.01)
