"""SimulatedClock: virtual time semantics the whole test suite relies on."""

from __future__ import annotations

import asyncio

import pytest

from voiceprobe.clock import MonotonicClock, SimulatedClock
from conftest import run_simulated


def test_sleep_advances_virtual_time_exactly():
    async def scenario(clock: SimulatedClock):
        start = clock.now()
        await clock.sleep(1.5)
        return clock.now() - start

    _, elapsed = run_simulated(scenario)
    assert elapsed == pytest.approx(1.5)


def test_concurrent_sleeps_wake_in_deadline_order():
    events: list[tuple[str, float]] = []

    async def sleeper(clock: SimulatedClock, label: str, duration: float):
        await clock.sleep(duration)
        events.append((label, clock.now()))

    async def scenario(clock: SimulatedClock):
        await asyncio.gather(
            sleeper(clock, "c", 3.0), sleeper(clock, "a", 1.0), sleeper(clock, "b", 2.0)
        )
        return clock.now()

    _, end = run_simulated(scenario)
    assert [label for label, _ in events] == ["a", "b", "c"]
    assert [t for _, t in events] == [pytest.approx(1.0), pytest.approx(2.0), pytest.approx(3.0)]
    # Concurrent waits overlap: total virtual time is the max, not the sum.
    assert end == pytest.approx(3.0)


def test_nested_sleeps_accumulate():
    async def scenario(clock: SimulatedClock):
        await clock.sleep(0.25)
        await clock.sleep(0.75)
        return clock.now()

    _, end = run_simulated(scenario)
    assert end == pytest.approx(1.0)


def test_zero_and_negative_sleep_do_not_advance_time():
    async def scenario(clock: SimulatedClock):
        await clock.sleep(0)
        await clock.sleep(-5)
        return clock.now()

    _, end = run_simulated(scenario)
    assert end == 0.0


def test_deadlock_without_timers_is_detected():
    async def scenario(_clock: SimulatedClock):
        await asyncio.get_running_loop().create_future()  # never resolved

    with pytest.raises(RuntimeError, match="deadlock"):
        run_simulated(scenario)


def test_monotonic_clock_measures_real_sleep():
    clock = MonotonicClock()

    async def scenario():
        start = clock.now()
        await clock.sleep(0.05)
        return clock.now() - start

    elapsed = asyncio.run(scenario())
    assert elapsed >= 0.045
