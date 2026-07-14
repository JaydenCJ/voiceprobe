"""Concurrent call simulation: N callers playing a scenario at once.

The runner launches ``calls`` pipeline playthroughs as asyncio tasks with
optional linear ramp-up, waits for all of them, and packages the spans
plus run metadata into a JSON-serializable result document
(``schema_version`` guards future format changes).
"""

from __future__ import annotations

import asyncio
import datetime as _dt
from dataclasses import dataclass
from typing import Any

from voiceprobe import __version__
from voiceprobe.backends.base import BackendSet
from voiceprobe.clock import Clock
from voiceprobe.pipeline import CallResult, VoicePipeline
from voiceprobe.scenario import Scenario

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LoadTestConfig:
    """Knobs for one load-test run."""

    calls: int = 1
    ramp_s: float = 0.0
    call_timeout_s: float = 300.0

    def __post_init__(self) -> None:
        if self.calls < 1:
            raise ValueError("calls must be >= 1")
        if self.ramp_s < 0:
            raise ValueError("ramp_s must be >= 0")
        if self.call_timeout_s <= 0:
            raise ValueError("call_timeout_s must be > 0")


@dataclass
class LoadTestResult:
    """All measured calls plus run metadata."""

    scenario_name: str
    config: LoadTestConfig
    backends: dict[str, str]
    started_wall: str
    clock_start: float
    clock_end: float
    calls: list[CallResult]

    @property
    def duration_s(self) -> float:
        return self.clock_end - self.clock_start

    @property
    def ok_calls(self) -> int:
        return sum(1 for c in self.calls if c.ok)

    @property
    def failed_calls(self) -> int:
        return len(self.calls) - self.ok_calls

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "tool": {"name": "voiceprobe", "version": __version__},
            "scenario": self.scenario_name,
            "config": {
                "calls": self.config.calls,
                "ramp_s": self.config.ramp_s,
                "call_timeout_s": self.config.call_timeout_s,
            },
            "backends": self.backends,
            "started_wall": self.started_wall,
            "duration_s": round(self.duration_s, 3),
            "ok_calls": self.ok_calls,
            "failed_calls": self.failed_calls,
            "calls": [c.to_dict() for c in self.calls],
        }


async def run_load_test(
    scenario: Scenario,
    backends: BackendSet,
    config: LoadTestConfig,
    clock: Clock,
) -> LoadTestResult:
    """Run ``config.calls`` concurrent scenario playthroughs and collect spans."""
    pipeline = VoicePipeline(backends, clock)
    started_wall = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    clock_start = clock.now()

    async def one_call(call_id: int) -> CallResult:
        if config.calls > 1 and config.ramp_s > 0:
            # Linear ramp: call k starts at k/(calls-1) * ramp_s.
            offset = config.ramp_s * call_id / (config.calls - 1)
            await clock.sleep(offset)
        try:
            # The safety timeout is wall-clock (event-loop) time by design:
            # it protects real runs against hung backends and never fires in
            # fast simulated-clock tests.
            return await asyncio.wait_for(
                pipeline.run_call(scenario, call_id), timeout=config.call_timeout_s
            )
        except asyncio.TimeoutError:
            now = clock.now()
            result = CallResult(call_id=call_id, started=now, ended=now)
            result.error = f"call timed out after {config.call_timeout_s:.0f}s"
            return result

    tasks = [asyncio.create_task(one_call(i)) for i in range(config.calls)]
    calls = list(await asyncio.gather(*tasks))
    calls.sort(key=lambda c: c.call_id)
    clock_end = clock.now()
    return LoadTestResult(
        scenario_name=scenario.name,
        config=config,
        backends=backends.describe(),
        started_wall=started_wall,
        clock_start=clock_start,
        clock_end=clock_end,
        calls=calls,
    )
