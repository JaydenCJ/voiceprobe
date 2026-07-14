"""HTML report: self-containment and geometric correctness of the SVGs."""

from __future__ import annotations

import re

import pytest

from voiceprobe.pipeline import CallResult, Span, TurnResult
from voiceprobe.report import render_report
from voiceprobe.report.html import flamegraph_svg, histogram_svg, waterfall_svg
from voiceprobe.runner import LoadTestConfig, run_load_test
from voiceprobe.scenario import Scenario, Turn
from conftest import fixed_backend_set, run_simulated


def _sample_doc():
    scenario = Scenario(
        name="report-scenario",
        turns=(
            Turn(user_text="first question", pause_ms=0.0),
            Turn(user_text="second question", pause_ms=100.0),
        ),
    )

    def make(clock):
        backends = fixed_backend_set(clock)
        return run_load_test(scenario, backends, LoadTestConfig(calls=3, ramp_s=1.0), clock)

    _, result = run_simulated(make)
    return result.to_dict()


def test_report_is_self_contained_html():
    html = render_report(_sample_doc())
    assert html.startswith("<!DOCTYPE html>")
    assert "<svg" in html
    # No external assets and no scripts: openable offline, safe to archive.
    assert "<script" not in html
    assert 'src="http' not in html and "href=\"http" not in html
    assert "@import" not in html and "url(" not in html


def test_report_contains_stages_scenario_and_metrics():
    html = render_report(_sample_doc())
    for stage in ("vad", "stt", "llm", "tts"):
        assert stage in html
    assert "report-scenario" in html
    assert "first audio" in html
    assert "Flame chart" in html
    assert "Waterfalls" in html


def _bar_widths(svg: str) -> list[float]:
    # Chart bars are rounded with rx="3"; legend swatches use rx="2".
    return [
        float(m.group(1))
        for m in re.finditer(r'<rect [^>]*width="([0-9.]+)" [^>]*rx="3"', svg)
    ]


def test_waterfall_bar_widths_are_proportional_to_duration():
    turn = TurnResult(
        index=0,
        started=0.0,
        spans=[Span("stt", 0.0, 0.1), Span("llm", 0.1, 0.3)],  # 100 ms vs 200 ms
        total_ms=300.0,
    )
    call = CallResult(call_id=0, started=0.0, ended=0.3, turns=[turn])
    svg = waterfall_svg(call)
    widths = _bar_widths(svg)
    assert len(widths) == 2
    assert widths[1] / widths[0] == pytest.approx(2.0, rel=0.02)
    assert "turn 1 · stt" in svg


def test_waterfall_shows_ttft_and_ttfb_in_tooltips():
    turn = TurnResult(
        index=0,
        started=0.0,
        spans=[
            Span("llm", 0.0, 0.5, {"ttft_ms": 320.0}),
            Span("tts", 0.5, 0.8, {"ttfb_ms": 150.0}),
        ],
        total_ms=800.0,
    )
    call = CallResult(call_id=0, started=0.0, ended=0.8, turns=[turn])
    svg = waterfall_svg(call)
    assert "TTFT 320 ms" in svg
    assert "TTFB 150 ms" in svg


def test_flamegraph_places_ramped_calls_at_offset_positions():
    doc = _sample_doc()
    calls = [CallResult.from_dict(c) for c in doc["calls"]]
    clock_start = min(c.started for c in calls)
    clock_end = max(c.ended for c in calls)
    svg = flamegraph_svg(calls, clock_start, clock_end)
    assert svg.count("call 0") >= 1
    assert svg.count("call 2") >= 1
    # Every stage color appears in the deepest layer.
    for color in ("#5b8def", "#3fa97c", "#e2a93b", "#c76bd1"):
        assert color in svg
    # Ramped call rects start at increasing x positions.
    xs = [
        float(m.group(1))
        for m in re.finditer(r'<rect x="([0-9.]+)"[^>]*fill="#aebbd3"', svg)
    ]
    assert len(xs) == 3
    assert xs[0] < xs[1] < xs[2]


def test_histogram_bins_account_for_every_sample():
    samples = [100.0, 110.0, 500.0, 505.0, 900.0, 910.0, 915.0, 100.0]
    svg = histogram_svg(samples, "test histogram")
    counts = [int(m.group(1)) for m in re.finditer(r"(\d+) turn\(s\)", svg)]
    assert sum(counts) == len(samples)


def test_report_flags_failed_calls():
    doc = _sample_doc()
    doc["calls"][1]["error"] = "turn 1 failed: stt: boom"
    doc["failed_calls"] = 1
    html = render_report(doc)
    assert "1 call(s) failed" in html
    assert "stt: boom" in html


def test_report_caps_waterfalls_but_keeps_aggregates():
    doc = _sample_doc()
    # Duplicate calls beyond the waterfall cap.
    base = doc["calls"][0]
    doc["calls"] = [dict(base, call_id=i) for i in range(9)]
    html = render_report(doc)
    assert "showing the first 6 of 9 calls" in html


def test_html_escapes_untrusted_scenario_text():
    doc = _sample_doc()
    doc["scenario"] = '<img src=x onerror=alert(1)>'
    html = render_report(doc)
    assert "<img src=x" not in html
    assert "&lt;img" in html
