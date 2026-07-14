"""Clock abstraction: real monotonic time for measurements, virtual time for tests.

Every latency measured by voiceprobe goes through a ``Clock`` so that the
attribution logic (spans, e2e math, concurrency accounting) can be tested
deterministically against a discrete-event ``SimulatedClock`` while real
load tests run on ``MonotonicClock``.
"""

from __future__ import annotations

import asyncio
import contextlib
import heapq
import time
from typing import Any, Coroutine, Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Minimal time source used by the pipeline, runner and mock backends."""

    def now(self) -> float:
        """Return the current time in seconds (monotonic, arbitrary epoch)."""
        ...

    async def sleep(self, seconds: float) -> None:
        """Suspend the calling task for ``seconds`` of clock time."""
        ...


class MonotonicClock:
    """Wall-clock implementation backed by ``time.monotonic``."""

    def now(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, seconds))


class SimulatedClock:
    """Deterministic discrete-event clock.

    Tasks that ``await clock.sleep(d)`` are parked on a timer heap. The
    ``run`` driver advances virtual time to the earliest pending timer
    whenever every task is blocked, so a simulated 90-second load test
    finishes in milliseconds of real time and produces exact, reproducible
    latency numbers.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = float(start)
        self._heap: list[tuple[float, int, asyncio.Future[None]]] = []
        self._seq = 0

    def now(self) -> float:
        return self._now

    @property
    def pending_timers(self) -> int:
        """Number of tasks currently parked on the timer heap."""
        return sum(1 for _, _, fut in self._heap if not fut.done())

    async def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            # Yield control once so ordering matches a real event loop.
            await asyncio.sleep(0)
            return
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()
        self._seq += 1
        heapq.heappush(self._heap, (self._now + seconds, self._seq, fut))
        try:
            await fut
        except asyncio.CancelledError:
            fut.cancel()
            raise

    async def run(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Drive ``coro`` to completion, advancing virtual time when idle.

        Raises ``RuntimeError`` if the coroutine is still pending while no
        timers are scheduled (a genuine deadlock in the code under test).
        """
        task = asyncio.ensure_future(coro)
        try:
            while not task.done():
                await self._settle()
                if task.done():
                    break
                if not self._heap:
                    raise RuntimeError(
                        "SimulatedClock deadlock: task is pending but no timers are scheduled"
                    )
                due, _, fut = heapq.heappop(self._heap)
                if fut.done():
                    continue
                self._now = max(self._now, due)
                fut.set_result(None)
        except BaseException:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            raise
        return task.result()

    @staticmethod
    async def _settle() -> None:
        # Let the event loop drain every ready callback. Each sleep(0) runs
        # one pass of ready callbacks; a bounded number of passes is enough
        # for the finite await-chains used in the pipeline and tests.
        for _ in range(50):
            await asyncio.sleep(0)
