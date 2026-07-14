"""Concurrent runner: parallelism, ramp-up and timeout accounting."""

from __future__ import annotations

import asyncio

import pytest

from voiceprobe.backends.base import BackendSet
from voiceprobe.clock import MonotonicClock, SimulatedClock
from voiceprobe.runner import LoadTestConfig, run_load_test
from voiceprobe.scenario import Scenario, Turn
from conftest import FixedLLM, FixedSTT, FixedTTS, FixedVAD, fixed_backend_set, run_simulated

SCENARIO = Scenario(name="load", turns=(Turn(user_text="ping", pause_ms=0.0),))
# One call with the Fixed* defaults takes exactly 1.0 virtual seconds:
# vad 0.05 + stt 0.30 + llm 0.44 + tts 0.21.
CALL_S = 1.0


def run_load(calls: int, ramp_s: float = 0.0):
    def make(clock: SimulatedClock):
        backends = fixed_backend_set(clock)
        config = LoadTestConfig(calls=calls, ramp_s=ramp_s)
        return run_load_test(SCENARIO, backends, config, clock)

    return run_simulated(make)


def test_calls_run_concurrently_not_sequentially():
    clock, result = run_load(calls=8)
    assert len(result.calls) == 8
    assert result.ok_calls == 8
    # Eight concurrent 1 s calls finish in ~1 s of virtual time, not 8 s.
    assert clock.now() == pytest.approx(CALL_S, rel=1e-6)
    assert result.duration_s == pytest.approx(CALL_S, rel=1e-6)


def test_linear_ramp_spaces_call_starts():
    _, result = run_load(calls=3, ramp_s=10.0)
    starts = [c.started for c in result.calls]
    assert starts == [
        pytest.approx(0.0, abs=1e-6),
        pytest.approx(5.0, rel=1e-6),
        pytest.approx(10.0, rel=1e-6),
    ]
    assert result.duration_s == pytest.approx(10.0 + CALL_S, rel=1e-6)


def test_results_are_ordered_by_call_id():
    _, result = run_load(calls=5, ramp_s=4.0)
    assert [c.call_id for c in result.calls] == [0, 1, 2, 3, 4]


def test_result_document_shape():
    _, result = run_load(calls=2)
    doc = result.to_dict()
    assert doc["schema_version"] == 1
    assert doc["tool"]["name"] == "voiceprobe"
    assert doc["scenario"] == "load"
    assert doc["config"]["calls"] == 2
    assert doc["ok_calls"] == 2 and doc["failed_calls"] == 0
    assert set(doc["backends"]) == {"vad", "stt", "llm", "tts"}
    assert len(doc["calls"]) == 2


def test_config_validation():
    with pytest.raises(ValueError):
        LoadTestConfig(calls=0)
    with pytest.raises(ValueError):
        LoadTestConfig(calls=1, ramp_s=-1)
    with pytest.raises(ValueError):
        LoadTestConfig(calls=1, call_timeout_s=0)


def test_hung_backend_hits_wall_clock_timeout():
    clock = MonotonicClock()

    class HangingSTT:
        name = "hanging-stt"

        async def transcribe(self, audio):
            await asyncio.sleep(30)
            raise AssertionError("should have been cancelled")

    backends = BackendSet(
        vad=FixedVAD(clock, latency_s=0.0),
        stt=HangingSTT(),
        llm=FixedLLM(clock, ttft_s=0.0, token_s=0.0),
        tts=FixedTTS(clock, ttfb_s=0.0, chunk_s=0.0),
    )
    scenario = Scenario(name="hang", turns=(Turn(user_text="hello", pause_ms=0.0),))
    config = LoadTestConfig(calls=1, call_timeout_s=0.2)
    result = asyncio.run(run_load_test(scenario, backends, config, clock))
    assert result.failed_calls == 1
    assert "timed out" in result.calls[0].error
