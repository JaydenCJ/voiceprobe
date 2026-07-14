"""The measured voice pipeline: VAD -> STT -> LLM -> TTS per turn.

Every stage call is wrapped in a :class:`Span` with timestamps taken from
the injected clock. Derived metrics:

- ``first_token_ms``  — caller stops speaking -> first LLM token
  (VAD decision + STT + LLM time-to-first-token).
- ``first_audio_ms``  — caller stops speaking -> first TTS audio byte
  (the latency the caller actually perceives).
- per-stage durations, with TTFT/TTFB markers inside the LLM/TTS spans.

Stages run sequentially (no sentence-level TTS pipelining in 0.1.0), which
matches the simplest production wiring and keeps attribution unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from voiceprobe.backends.base import BackendError, BackendSet
from voiceprobe.clock import Clock
from voiceprobe.scenario import Scenario, Turn

STAGES = ("vad", "stt", "llm", "tts")


@dataclass
class Span:
    """One measured stage execution, timestamps in seconds on the run clock."""

    stage: str
    start: float
    end: float
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end - self.start) * 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "start": round(self.start, 6),
            "end": round(self.end, 6),
            "duration_ms": round(self.duration_ms, 2),
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Span":
        return cls(
            stage=str(raw["stage"]),
            start=float(raw["start"]),
            end=float(raw["end"]),
            detail=dict(raw.get("detail") or {}),
        )


@dataclass
class TurnResult:
    """Measurements for one conversation turn."""

    index: int
    started: float
    spans: list[Span] = field(default_factory=list)
    transcript: str = ""
    reply: str = ""
    first_token_ms: float | None = None
    first_audio_ms: float | None = None
    total_ms: float = 0.0
    error: str | None = None

    def span(self, stage: str) -> Span | None:
        for s in self.spans:
            if s.stage == stage:
                return s
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "started": round(self.started, 6),
            "spans": [s.to_dict() for s in self.spans],
            "transcript": self.transcript,
            "reply": self.reply,
            "first_token_ms": None if self.first_token_ms is None else round(self.first_token_ms, 2),
            "first_audio_ms": None if self.first_audio_ms is None else round(self.first_audio_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TurnResult":
        return cls(
            index=int(raw["index"]),
            started=float(raw["started"]),
            spans=[Span.from_dict(s) for s in raw.get("spans", [])],
            transcript=str(raw.get("transcript") or ""),
            reply=str(raw.get("reply") or ""),
            first_token_ms=(
                None if raw.get("first_token_ms") is None else float(raw["first_token_ms"])
            ),
            first_audio_ms=(
                None if raw.get("first_audio_ms") is None else float(raw["first_audio_ms"])
            ),
            total_ms=float(raw.get("total_ms") or 0.0),
            error=raw.get("error"),
        )


@dataclass
class CallResult:
    """Measurements for one simulated call (a full scenario playthrough)."""

    call_id: int
    started: float
    ended: float = 0.0
    turns: list[TurnResult] = field(default_factory=list)
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        return (self.ended - self.started) * 1000.0

    @property
    def ok(self) -> bool:
        return self.error is None and all(t.error is None for t in self.turns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "started": round(self.started, 6),
            "ended": round(self.ended, 6),
            "duration_ms": round(self.duration_ms, 2),
            "turns": [t.to_dict() for t in self.turns],
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CallResult":
        return cls(
            call_id=int(raw["call_id"]),
            started=float(raw["started"]),
            ended=float(raw["ended"]),
            turns=[TurnResult.from_dict(t) for t in raw.get("turns", [])],
            error=raw.get("error"),
        )


class VoicePipeline:
    """Runs a scenario through the four stage backends, measuring each call."""

    def __init__(self, backends: BackendSet, clock: Clock) -> None:
        self._backends = backends
        self._clock = clock

    async def run_call(self, scenario: Scenario, call_id: int) -> CallResult:
        clock = self._clock
        result = CallResult(call_id=call_id, started=clock.now())
        history: list[dict[str, str]] = []
        for index, turn in enumerate(scenario.turns):
            if turn.pause_ms > 0:
                await clock.sleep(turn.pause_ms / 1000.0)
            turn_result = TurnResult(index=index, started=clock.now())
            result.turns.append(turn_result)
            try:
                await self._run_turn(scenario, turn, turn_result, history)
            except BackendError as exc:
                turn_result.error = str(exc)
                turn_result.total_ms = (clock.now() - turn_result.started) * 1000.0
                result.error = f"turn {index + 1} failed: {exc}"
                break
            turn_result.total_ms = (clock.now() - turn_result.started) * 1000.0
        result.ended = clock.now()
        return result

    async def _run_turn(
        self,
        scenario: Scenario,
        turn: Turn,
        turn_result: TurnResult,
        history: list[dict[str, str]],
    ) -> None:
        clock = self._clock
        backends = self._backends
        audio = turn.resolve_audio(scenario.base_dir, scenario.sample_rate)
        # The turn's zero point: the moment the caller finished speaking.
        turn_zero = clock.now()

        # Stage 1: VAD end-of-speech decision.
        t0 = clock.now()
        vad_result = await backends.vad.detect(audio)
        t1 = clock.now()
        detail = dict(vad_result.detail)
        detail["speech_detected"] = vad_result.speech_detected
        turn_result.spans.append(Span("vad", t0, t1, detail))
        if not vad_result.speech_detected:
            raise BackendError(
                f"vad: no speech detected in turn audio ({audio.duration_ms:.0f} ms clip)"
            )

        # Stage 2: STT final transcript.
        t0 = clock.now()
        stt_result = await backends.stt.transcribe(audio)
        t1 = clock.now()
        turn_result.spans.append(Span("stt", t0, t1, {"chars": len(stt_result.text)}))
        turn_result.transcript = stt_result.text

        # Stage 3: LLM streamed reply (records TTFT).
        t0 = clock.now()
        first_token_at: float | None = None
        tokens: list[str] = []
        async for token in backends.llm.stream_reply(stt_result.text, list(history)):
            if first_token_at is None:
                first_token_at = clock.now()
            tokens.append(token)
        t1 = clock.now()
        if first_token_at is None:
            raise BackendError("llm: backend streamed no tokens")
        reply = "".join(tokens)
        turn_result.reply = reply
        turn_result.spans.append(
            Span(
                "llm",
                t0,
                t1,
                {
                    "ttft_ms": round((first_token_at - t0) * 1000.0, 2),
                    "tokens": len(tokens),
                    "chars": len(reply),
                },
            )
        )
        turn_result.first_token_ms = (first_token_at - turn_zero) * 1000.0

        # Stage 4: TTS streamed audio (records TTFB).
        t0 = clock.now()
        first_chunk_at: float | None = None
        audio_bytes = 0
        chunks = 0
        async for chunk in backends.tts.stream_speech(reply):
            if first_chunk_at is None:
                first_chunk_at = clock.now()
            audio_bytes += len(chunk)
            chunks += 1
        t1 = clock.now()
        if first_chunk_at is None:
            raise BackendError("tts: backend streamed no audio")
        turn_result.spans.append(
            Span(
                "tts",
                t0,
                t1,
                {
                    "ttfb_ms": round((first_chunk_at - t0) * 1000.0, 2),
                    "chunks": chunks,
                    "audio_bytes": audio_bytes,
                },
            )
        )
        turn_result.first_audio_ms = (first_chunk_at - turn_zero) * 1000.0

        history.append({"role": "user", "content": stt_result.text})
        history.append({"role": "assistant", "content": reply})
