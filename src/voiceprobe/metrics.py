"""Latency aggregation: per-stage and end-to-end statistics.

Percentiles use the nearest-rank method (ceil(p/100 * n)), the same
definition most load-testing tools report, so numbers are comparable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from voiceprobe.pipeline import STAGES, CallResult


@dataclass(frozen=True)
class StatLine:
    """Summary statistics over a list of millisecond samples."""

    name: str
    count: int
    mean: float
    p50: float
    p95: float
    p99: float
    max: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "mean_ms": round(self.mean, 1),
            "p50_ms": round(self.p50, 1),
            "p95_ms": round(self.p95, 1),
            "p99_ms": round(self.p99, 1),
            "max_ms": round(self.max, 1),
        }


def percentile(samples: list[float], pct: float) -> float:
    """Nearest-rank percentile; ``samples`` need not be sorted."""
    if not samples:
        raise ValueError("percentile of empty sample set")
    if not 0 < pct <= 100:
        raise ValueError("pct must be in (0, 100]")
    ordered = sorted(samples)
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return ordered[rank - 1]


def stat_line(name: str, samples: Iterable[float]) -> StatLine | None:
    values = [float(v) for v in samples]
    if not values:
        return None
    return StatLine(
        name=name,
        count=len(values),
        mean=sum(values) / len(values),
        p50=percentile(values, 50),
        p95=percentile(values, 95),
        p99=percentile(values, 99),
        max=max(values),
    )


@dataclass
class RunMetrics:
    """Aggregated view over all calls in a load-test result."""

    stage_stats: list[StatLine]
    first_token: StatLine | None
    first_audio: StatLine | None
    turn_total: StatLine | None
    turns_measured: int
    turns_failed: int
    stage_share: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": [s.to_dict() for s in self.stage_stats],
            "first_token": self.first_token.to_dict() if self.first_token else None,
            "first_audio": self.first_audio.to_dict() if self.first_audio else None,
            "turn_total": self.turn_total.to_dict() if self.turn_total else None,
            "turns_measured": self.turns_measured,
            "turns_failed": self.turns_failed,
            "stage_share": {k: round(v, 3) for k, v in self.stage_share.items()},
        }


def compute_metrics(calls: list[CallResult]) -> RunMetrics:
    """Aggregate spans from all calls into per-stage and e2e statistics."""
    stage_samples: dict[str, list[float]] = {stage: [] for stage in STAGES}
    first_token: list[float] = []
    first_audio: list[float] = []
    turn_total: list[float] = []
    turns_measured = 0
    turns_failed = 0
    for call in calls:
        for turn in call.turns:
            if turn.error is not None:
                turns_failed += 1
                continue
            turns_measured += 1
            for span in turn.spans:
                if span.stage in stage_samples:
                    stage_samples[span.stage].append(span.duration_ms)
            if turn.first_token_ms is not None:
                first_token.append(turn.first_token_ms)
            if turn.first_audio_ms is not None:
                first_audio.append(turn.first_audio_ms)
            turn_total.append(turn.total_ms)
    stage_stats = [
        line
        for stage in STAGES
        if (line := stat_line(stage, stage_samples[stage])) is not None
    ]
    stage_mean_sum = sum(s.mean for s in stage_stats)
    stage_share = {
        s.name: (s.mean / stage_mean_sum if stage_mean_sum > 0 else 0.0) for s in stage_stats
    }
    return RunMetrics(
        stage_stats=stage_stats,
        first_token=stat_line("first_token", first_token),
        first_audio=stat_line("first_audio", first_audio),
        turn_total=stat_line("turn_total", turn_total),
        turns_measured=turns_measured,
        turns_failed=turns_failed,
        stage_share=stage_share,
    )


def format_summary(metrics: RunMetrics, duration_s: float, calls: int, failed_calls: int) -> str:
    """Render the terminal summary table with a per-stage share bar."""
    lines: list[str] = []
    lines.append(
        f"calls: {calls}  ok: {calls - failed_calls}  failed: {failed_calls}"
        f"  turns: {metrics.turns_measured}  wall time: {duration_s:.2f}s"
    )
    lines.append("")
    header = (
        f"{'stage':<12} {'count':>5} {'mean ms':>8} {'p50 ms':>8} {'p95 ms':>8} "
        f"{'p99 ms':>8} {'max ms':>8}  share"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for s in metrics.stage_stats:
        share = metrics.stage_share.get(s.name, 0.0)
        bar = "#" * max(1, round(share * 24))
        lines.append(
            f"{s.name:<12} {s.count:>5} {s.mean:>8.0f} {s.p50:>8.0f} {s.p95:>8.0f} "
            f"{s.p99:>8.0f} {s.max:>8.0f}  {bar} {share * 100:.0f}%"
        )
    lines.append("-" * len(header))
    for label, line in (
        ("first token (e2e)", metrics.first_token),
        ("first audio (e2e)", metrics.first_audio),
        ("turn total", metrics.turn_total),
    ):
        if line is not None:
            lines.append(
                f"{label:<18} mean {line.mean:>6.0f} ms   p50 {line.p50:>6.0f} ms   "
                f"p95 {line.p95:>6.0f} ms   max {line.max:>6.0f} ms"
            )
    return "\n".join(lines)
