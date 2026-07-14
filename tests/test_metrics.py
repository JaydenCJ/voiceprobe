"""Metrics: percentile math and per-stage aggregation."""

from __future__ import annotations

import pytest

from voiceprobe.metrics import compute_metrics, format_summary, percentile, stat_line
from voiceprobe.pipeline import CallResult, Span, TurnResult


def test_percentile_nearest_rank_definition():
    samples = [float(v) for v in range(1, 101)]  # 1..100
    assert percentile(samples, 50) == 50.0
    assert percentile(samples, 95) == 95.0
    assert percentile(samples, 99) == 99.0
    assert percentile(samples, 100) == 100.0


def test_percentile_small_sets_and_unsorted_input():
    assert percentile([30.0, 10.0, 20.0], 50) == 20.0
    assert percentile([42.0], 99) == 42.0


def test_percentile_input_validation():
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 0)
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_stat_line_summarizes_samples():
    line = stat_line("stt", [100.0, 200.0, 300.0])
    assert line.count == 3
    assert line.mean == pytest.approx(200.0)
    assert line.p50 == 200.0
    assert line.max == 300.0
    assert stat_line("empty", []) is None


def _call_with_stage_durations(call_id: int, durations_ms: dict[str, float]) -> CallResult:
    spans = []
    t = 0.0
    for stage in ("vad", "stt", "llm", "tts"):
        d = durations_ms[stage] / 1000.0
        spans.append(Span(stage, t, t + d))
        t += d
    turn = TurnResult(index=0, started=0.0, spans=spans, total_ms=t * 1000.0)
    turn.first_token_ms = durations_ms["vad"] + durations_ms["stt"]
    turn.first_audio_ms = turn.first_token_ms + durations_ms["llm"]
    return CallResult(call_id=call_id, started=0.0, ended=t, turns=[turn])


def test_compute_metrics_exact_stage_means_and_shares():
    calls = [
        _call_with_stage_durations(0, {"vad": 50, "stt": 300, "llm": 400, "tts": 250}),
        _call_with_stage_durations(1, {"vad": 70, "stt": 500, "llm": 600, "tts": 350}),
    ]
    metrics = compute_metrics(calls)
    by_name = {s.name: s for s in metrics.stage_stats}
    assert by_name["vad"].mean == pytest.approx(60.0)
    assert by_name["stt"].mean == pytest.approx(400.0)
    assert by_name["llm"].mean == pytest.approx(500.0)
    assert by_name["tts"].mean == pytest.approx(300.0)
    assert metrics.turns_measured == 2
    assert metrics.turns_failed == 0
    # Shares are stage-mean fractions and sum to 1.
    assert sum(metrics.stage_share.values()) == pytest.approx(1.0)
    assert metrics.stage_share["llm"] == pytest.approx(500.0 / 1260.0)


def test_failed_turns_are_excluded_from_stats_but_counted():
    good = _call_with_stage_durations(0, {"vad": 50, "stt": 300, "llm": 400, "tts": 250})
    bad = CallResult(
        call_id=1,
        started=0.0,
        ended=1.0,
        turns=[TurnResult(index=0, started=0.0, error="stt: boom", total_ms=100.0)],
        error="turn 1 failed: stt: boom",
    )
    metrics = compute_metrics([good, bad])
    assert metrics.turns_measured == 1
    assert metrics.turns_failed == 1
    by_name = {s.name: s for s in metrics.stage_stats}
    assert by_name["stt"].count == 1  # the failed turn contributed nothing


def test_metrics_dict_and_summary_render():
    calls = [_call_with_stage_durations(0, {"vad": 50, "stt": 300, "llm": 400, "tts": 250})]
    metrics = compute_metrics(calls)
    doc = metrics.to_dict()
    assert doc["stages"][0]["name"] == "vad"
    assert doc["first_audio"]["mean_ms"] == pytest.approx(750.0)
    text = format_summary(metrics, duration_s=1.0, calls=1, failed_calls=0)
    assert "stage" in text and "llm" in text
    assert "first audio (e2e)" in text
    assert "p95" in text
