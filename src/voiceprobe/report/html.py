"""Self-contained HTML latency report: waterfall, flame chart, histogram.

The generated file embeds all CSS and SVG inline — no JavaScript, no
external fonts, no CDN — so it can be opened from disk, attached to a
ticket, or archived next to CI artifacts. All numbers come straight from
the measured spans in the results document.
"""

from __future__ import annotations

import datetime as _dt
import html as _html
from typing import Any

from voiceprobe.metrics import RunMetrics, compute_metrics
from voiceprobe.pipeline import CallResult

STAGE_COLORS = {
    "vad": "#5b8def",
    "stt": "#3fa97c",
    "llm": "#e2a93b",
    "tts": "#c76bd1",
}
_FALLBACK_COLOR = "#8a93a6"
_WIDTH = 1080
_MAX_WATERFALL_CALLS = 6


def _esc(value: Any) -> str:
    return _html.escape(str(value), quote=True)


def _color(stage: str) -> str:
    return STAGE_COLORS.get(stage, _FALLBACK_COLOR)


def _fmt_ms(value: float) -> str:
    return f"{value:,.0f} ms" if value >= 10 else f"{value:.1f} ms"


def _time_axis(total_ms: float, width: float, x0: float, y: float, height: float) -> str:
    """Vertical gridlines with millisecond labels for a time-scaled chart."""
    if total_ms <= 0:
        return ""
    # Pick a tick step that yields 4-8 gridlines.
    step = 1.0
    for candidate in (10, 25, 50, 100, 250, 500, 1000, 2000, 5000, 10000, 30000, 60000):
        if total_ms / candidate <= 8:
            step = float(candidate)
            break
    else:
        step = total_ms / 6.0
    parts: list[str] = []
    t = 0.0
    while t <= total_ms + 1e-9:
        x = x0 + (t / total_ms) * width
        parts.append(
            f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x:.1f}" y2="{y + height:.1f}" '
            f'stroke="#e3e6ec" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{y + height + 14:.1f}" font-size="10" '
            f'fill="#7a8194" text-anchor="middle">{t:,.0f}</text>'
        )
        t += step
    parts.append(
        f'<text x="{x0 + width:.1f}" y="{y + height + 28:.1f}" font-size="10" '
        f'fill="#7a8194" text-anchor="end">time (ms)</text>'
    )
    return "".join(parts)


def _legend(x: float, y: float) -> str:
    parts: list[str] = []
    offset = 0.0
    for stage, color in STAGE_COLORS.items():
        parts.append(
            f'<rect x="{x + offset:.1f}" y="{y:.1f}" width="10" height="10" rx="2" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{x + offset + 14:.1f}" y="{y + 9:.1f}" font-size="11" fill="#3c4356">'
            f"{stage}</text>"
        )
        offset += 60.0
    return "".join(parts)


def waterfall_svg(call: CallResult) -> str:
    """Waterfall for one call: each measured span is a time-positioned bar."""
    rows: list[tuple[str, float, float, str, str]] = []
    origin = call.started
    for turn in call.turns:
        for span in turn.spans:
            label = f"turn {turn.index + 1} · {span.stage}"
            tooltip = f"{label}: {span.duration_ms:.0f} ms"
            extras = []
            if "ttft_ms" in span.detail:
                extras.append(f"TTFT {span.detail['ttft_ms']:.0f} ms")
            if "ttfb_ms" in span.detail:
                extras.append(f"TTFB {span.detail['ttfb_ms']:.0f} ms")
            if extras:
                tooltip += " (" + ", ".join(extras) + ")"
            rows.append(
                (
                    span.stage,
                    (span.start - origin) * 1000.0,
                    (span.end - origin) * 1000.0,
                    label,
                    tooltip,
                )
            )
    if not rows:
        return "<p>No measured spans for this call.</p>"
    total_ms = max(end for _, _, end, _, _ in rows)
    label_w = 140.0
    chart_w = _WIDTH - label_w - 20.0
    row_h = 22.0
    top = 26.0
    height = top + len(rows) * row_h + 40.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_WIDTH} {height:.0f}" '
        f'width="100%" role="img" aria-label="Latency waterfall for call {call.call_id}">'
    ]
    parts.append(_time_axis(total_ms, chart_w, label_w, top, len(rows) * row_h))
    parts.append(_legend(label_w, 4.0))
    for i, (stage, start_ms, end_ms, label, tooltip) in enumerate(rows):
        y = top + i * row_h
        x = label_w + (start_ms / total_ms) * chart_w
        w = max(1.5, ((end_ms - start_ms) / total_ms) * chart_w)
        parts.append(
            f'<text x="{label_w - 8:.1f}" y="{y + 14:.1f}" font-size="11" fill="#3c4356" '
            f'text-anchor="end">{_esc(label)}</text>'
        )
        parts.append(
            f'<rect x="{x:.1f}" y="{y + 3:.1f}" width="{w:.1f}" height="{row_h - 7:.1f}" '
            f'rx="3" fill="{_color(stage)}"><title>{_esc(tooltip)}</title></rect>'
        )
        duration = end_ms - start_ms
        text_x = x + w + 6
        if text_x < label_w + chart_w - 60:
            parts.append(
                f'<text x="{text_x:.1f}" y="{y + 14:.1f}" font-size="10" fill="#7a8194">'
                f"{duration:,.0f} ms</text>"
            )
    parts.append("</svg>")
    return "".join(parts)


def flamegraph_svg(calls: list[CallResult], clock_start: float, clock_end: float) -> str:
    """Flame chart of the whole run: run > calls > turns > stages.

    X position and width are proportional to measured time, so concurrency
    (overlapping calls) and stage cost are visible in one picture.
    """
    total_ms = (clock_end - clock_start) * 1000.0
    if total_ms <= 0 or not calls:
        return "<p>No calls to draw.</p>"
    row_h = 24.0
    top = 26.0
    depth = 4  # run, call, turn, stage
    height = top + depth * row_h + 40.0
    chart_w = float(_WIDTH - 20)
    x0 = 10.0

    def rect(level: int, start_ms: float, end_ms: float, color: str, label: str, tooltip: str) -> str:
        x = x0 + (start_ms / total_ms) * chart_w
        w = max(1.0, ((end_ms - start_ms) / total_ms) * chart_w)
        y = top + level * row_h
        chunk = (
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{row_h - 3:.1f}" rx="2" '
            f'fill="{color}" stroke="#ffffff" stroke-width="0.5">'
            f"<title>{_esc(tooltip)}</title></rect>"
        )
        # Only label boxes wide enough to hold text.
        if w > 7.5 * len(label) * 0.85 + 8:
            chunk += (
                f'<text x="{x + 5:.1f}" y="{y + row_h / 2 + 3:.1f}" font-size="11" '
                f'fill="#1e2430">{_esc(label)}</text>'
            )
        return chunk

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_WIDTH} {height:.0f}" '
        f'width="100%" role="img" aria-label="Flame chart of the load-test run">'
    ]
    parts.append(_time_axis(total_ms, chart_w, x0, top, depth * row_h))
    parts.append(_legend(x0, 4.0))
    parts.append(
        rect(0, 0.0, total_ms, "#d8dde7", "run", f"run: {total_ms:,.0f} ms, {len(calls)} calls")
    )
    for call in calls:
        c_start = (call.started - clock_start) * 1000.0
        c_end = (call.ended - clock_start) * 1000.0
        call_label = f"call {call.call_id}"
        status = "ok" if call.ok else f"failed: {call.error}"
        parts.append(
            rect(
                1,
                c_start,
                c_end,
                "#aebbd3",
                call_label,
                f"{call_label}: {c_end - c_start:,.0f} ms ({status})",
            )
        )
        for turn in call.turns:
            if not turn.spans:
                continue
            t_start = (turn.spans[0].start - clock_start) * 1000.0
            t_end = (turn.spans[-1].end - clock_start) * 1000.0
            t_label = f"t{turn.index + 1}"
            parts.append(
                rect(
                    2,
                    t_start,
                    t_end,
                    "#c5cede",
                    t_label,
                    f"{call_label} turn {turn.index + 1}: {t_end - t_start:,.0f} ms",
                )
            )
            for span in turn.spans:
                s_start = (span.start - clock_start) * 1000.0
                s_end = (span.end - clock_start) * 1000.0
                parts.append(
                    rect(
                        3,
                        s_start,
                        s_end,
                        _color(span.stage),
                        span.stage,
                        f"{call_label} turn {turn.index + 1} {span.stage}: "
                        f"{span.duration_ms:,.0f} ms",
                    )
                )
    parts.append("</svg>")
    return "".join(parts)


def histogram_svg(samples: list[float], title: str, color: str = "#5b8def") -> str:
    """Simple binned histogram of millisecond samples."""
    if not samples:
        return "<p>No samples.</p>"
    lo, hi = min(samples), max(samples)
    span = max(1.0, hi - lo)
    bins = min(24, max(6, len(samples) // 2))
    counts = [0] * bins
    for v in samples:
        idx = min(bins - 1, int((v - lo) / span * bins))
        counts[idx] += 1
    peak = max(counts)
    chart_h, chart_w, x0, y0 = 120.0, float(_WIDTH - 40), 20.0, 16.0
    bar_w = chart_w / bins
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_WIDTH} {chart_h + 50:.0f}" '
        f'width="100%" role="img" aria-label="{_esc(title)}">'
    ]
    for i, count in enumerate(counts):
        if count == 0:
            continue
        h = (count / peak) * chart_h
        x = x0 + i * bar_w
        bin_lo = lo + i * span / bins
        bin_hi = lo + (i + 1) * span / bins
        parts.append(
            f'<rect x="{x + 1:.1f}" y="{y0 + chart_h - h:.1f}" width="{bar_w - 2:.1f}" '
            f'height="{h:.1f}" rx="2" fill="{color}">'
            f"<title>{_fmt_ms(bin_lo)} – {_fmt_ms(bin_hi)}: {count} turn(s)</title></rect>"
        )
    for frac, value in ((0.0, lo), (0.5, lo + span / 2), (1.0, hi)):
        x = x0 + frac * chart_w
        parts.append(
            f'<text x="{x:.1f}" y="{y0 + chart_h + 16:.1f}" font-size="10" fill="#7a8194" '
            f'text-anchor="middle">{value:,.0f} ms</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _stats_table(metrics: RunMetrics) -> str:
    rows = []
    for s in metrics.stage_stats:
        share = metrics.stage_share.get(s.name, 0.0)
        rows.append(
            "<tr>"
            f'<td><span class="dot" style="background:{_color(s.name)}"></span>{_esc(s.name)}</td>'
            f"<td>{s.count}</td><td>{s.mean:,.0f}</td><td>{s.p50:,.0f}</td>"
            f"<td>{s.p95:,.0f}</td><td>{s.p99:,.0f}</td><td>{s.max:,.0f}</td>"
            f"<td>{share * 100:.0f}%</td>"
            "</tr>"
        )
    for label, line in (
        ("first token (e2e)", metrics.first_token),
        ("first audio (e2e)", metrics.first_audio),
        ("turn total", metrics.turn_total),
    ):
        if line is None:
            continue
        rows.append(
            '<tr class="e2e">'
            f"<td>{_esc(label)}</td>"
            f"<td>{line.count}</td><td>{line.mean:,.0f}</td><td>{line.p50:,.0f}</td>"
            f"<td>{line.p95:,.0f}</td><td>{line.p99:,.0f}</td><td>{line.max:,.0f}</td><td></td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>stage</th><th>count</th><th>mean&nbsp;ms</th>"
        "<th>p50&nbsp;ms</th><th>p95&nbsp;ms</th><th>p99&nbsp;ms</th><th>max&nbsp;ms</th>"
        "<th>share</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


_CSS = """
:root { color-scheme: light; }
body { font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
       margin: 0; background: #f5f6f9; color: #1e2430; }
main { max-width: 1120px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 22px; margin: 8px 0 4px; }
h2 { font-size: 16px; margin: 32px 0 10px; color: #2c3345; }
.meta { color: #7a8194; font-size: 13px; margin-bottom: 16px; }
.card { background: #ffffff; border: 1px solid #e3e6ec; border-radius: 10px;
        padding: 16px; margin-bottom: 8px; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: right; padding: 6px 10px; border-bottom: 1px solid #eef0f4; }
th:first-child, td:first-child { text-align: left; }
th { color: #7a8194; font-weight: 600; }
tr.e2e td { font-weight: 600; background: #fafbfd; }
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 3px;
       margin-right: 7px; }
.warn { background: #fdf3e7; border: 1px solid #f0d9b8; border-radius: 8px;
        padding: 10px 14px; font-size: 13px; color: #7a5b22; margin-bottom: 12px; }
footer { color: #9aa1b2; font-size: 12px; margin-top: 32px; }
"""


def render_report(doc: dict[str, Any]) -> str:
    """Render a results document (``LoadTestResult.to_dict()``) to HTML."""
    calls = [CallResult.from_dict(c) for c in doc.get("calls", [])]
    metrics = compute_metrics(calls)
    clock_starts = [c.started for c in calls] or [0.0]
    clock_ends = [c.ended for c in calls] or [0.0]
    clock_start = min(clock_starts)
    clock_end = max(clock_ends)
    backends = doc.get("backends", {})
    backend_desc = ", ".join(f"{k}={_esc(v)}" for k, v in backends.items())
    generated = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    failed_calls = int(doc.get("failed_calls", sum(1 for c in calls if not c.ok)))

    sections: list[str] = []
    sections.append(
        f"<h1>voiceprobe latency report</h1>"
        f'<div class="meta">scenario <b>{_esc(doc.get("scenario", "?"))}</b>'
        f' · {len(calls)} call(s) · {_esc(doc.get("duration_s", "?"))}s wall time'
        f' · started {_esc(doc.get("started_wall", "?"))}'
        f" · backends: {backend_desc}</div>"
    )
    if failed_calls:
        failures = "; ".join(
            f"call {c.call_id}: {_esc(c.error or 'turn error')}" for c in calls if not c.ok
        )
        sections.append(f'<div class="warn">{failed_calls} call(s) failed — {failures}</div>')
    sections.append("<h2>Per-stage latency breakdown</h2>")
    sections.append(f'<div class="card">{_stats_table(metrics)}</div>')

    if metrics.first_audio is not None:
        first_audio_samples = [
            t.first_audio_ms
            for c in calls
            for t in c.turns
            if t.first_audio_ms is not None and t.error is None
        ]
        sections.append("<h2>Distribution: caller-perceived response latency (first audio)</h2>")
        sections.append(f'<div class="card">{histogram_svg(first_audio_samples, "first audio latency")}</div>')

    sections.append("<h2>Flame chart (whole run, all calls)</h2>")
    sections.append(f'<div class="card">{flamegraph_svg(calls, clock_start, clock_end)}</div>')

    shown = calls[:_MAX_WATERFALL_CALLS]
    sections.append("<h2>Waterfalls (per call)</h2>")
    if len(calls) > len(shown):
        sections.append(
            f'<div class="meta">showing the first {len(shown)} of {len(calls)} calls; '
            f"aggregate statistics above cover all calls.</div>"
        )
    for call in shown:
        status = "ok" if call.ok else f"failed — {_esc(call.error or 'turn error')}"
        sections.append(
            f'<div class="card"><div class="meta">call {call.call_id} · '
            f"{call.duration_ms:,.0f} ms · {status}</div>{waterfall_svg(call)}</div>"
        )

    tool = doc.get("tool", {})
    sections.append(
        f"<footer>generated {generated} by {_esc(tool.get('name', 'voiceprobe'))} "
        f"v{_esc(tool.get('version', '?'))} — self-contained report (no external assets)</footer>"
    )
    body = "".join(sections)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>voiceprobe report — {_esc(doc.get('scenario', ''))}</title>"
        f"<style>{_CSS}</style></head><body><main>{body}</main></body></html>\n"
    )
