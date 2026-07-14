"""Shared test fixtures: fixed-latency backends and simulated-clock helpers.

Tests never touch the network; HTTP backend tests run against a local fake
server bound to 127.0.0.1 (see ``fake_openai.py``).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from voiceprobe.audio import AudioClip
from voiceprobe.backends.base import BackendSet, STTResult, VADResult
from voiceprobe.clock import Clock, SimulatedClock


class FixedVAD:
    """VAD that always takes exactly ``latency_s`` of clock time."""

    def __init__(self, clock: Clock, latency_s: float = 0.05, detect_speech: bool = True) -> None:
        self._clock = clock
        self._latency = latency_s
        self._detect = detect_speech
        self.name = "fixed-vad"

    async def detect(self, audio: AudioClip) -> VADResult:
        await self._clock.sleep(self._latency)
        return VADResult(speech_detected=self._detect, speech_end_ms=audio.duration_ms)


class FixedSTT:
    """STT with exact latency, echoing the transcript hint."""

    def __init__(self, clock: Clock, latency_s: float = 0.3) -> None:
        self._clock = clock
        self._latency = latency_s
        self.name = "fixed-stt"
        self.seen_audio: list[AudioClip] = []

    async def transcribe(self, audio: AudioClip) -> STTResult:
        self.seen_audio.append(audio)
        await self._clock.sleep(self._latency)
        return STTResult(text=audio.transcript_hint or "fixed transcript")


class FixedLLM:
    """LLM with exact TTFT and inter-token latency; records history."""

    def __init__(
        self,
        clock: Clock,
        ttft_s: float = 0.4,
        token_s: float = 0.02,
        tokens: tuple[str, ...] = ("hello ", "there ", "caller"),
    ) -> None:
        self._clock = clock
        self._ttft = ttft_s
        self._token = token_s
        self._tokens = tokens
        self.name = "fixed-llm"
        self.histories: list[list[dict[str, str]]] = []

    async def stream_reply(
        self, transcript: str, history: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        self.histories.append(list(history))
        await self._clock.sleep(self._ttft)
        for i, token in enumerate(self._tokens):
            if i > 0:
                await self._clock.sleep(self._token)
            yield token


class FixedTTS:
    """TTS with exact TTFB and per-chunk latency."""

    def __init__(
        self, clock: Clock, ttfb_s: float = 0.15, chunk_s: float = 0.03, chunks: int = 3
    ) -> None:
        self._clock = clock
        self._ttfb = ttfb_s
        self._chunk = chunk_s
        self._chunks = chunks
        self.name = "fixed-tts"

    async def stream_speech(self, text: str) -> AsyncIterator[bytes]:
        await self._clock.sleep(self._ttfb)
        yield b"\x00" * 64
        for _ in range(self._chunks - 1):
            await self._clock.sleep(self._chunk)
            yield b"\x00" * 64


def fixed_backend_set(clock: Clock, **overrides) -> BackendSet:
    """A BackendSet of Fixed* backends with the default exact latencies."""
    return BackendSet(
        vad=overrides.get("vad", FixedVAD(clock)),
        stt=overrides.get("stt", FixedSTT(clock)),
        llm=overrides.get("llm", FixedLLM(clock)),
        tts=overrides.get("tts", FixedTTS(clock)),
    )


def run_simulated(make_coro: Callable[[SimulatedClock], Awaitable]):
    """Run a coroutine under a fresh SimulatedClock; return (clock, result)."""
    clock = SimulatedClock()

    async def driver():
        return await clock.run(make_coro(clock))

    result = asyncio.run(driver())
    return clock, result


@pytest.fixture()
def sim_clock() -> SimulatedClock:
    return SimulatedClock()
