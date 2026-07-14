"""Protocol definitions for the four voice-pipeline stages.

The pipeline depends only on these Protocols. Implementations live in
sibling modules (``mock``, ``energy``, ``http``) or in user code — any
object with matching methods works, no inheritance required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol, runtime_checkable

from voiceprobe.audio import AudioClip


class BackendError(Exception):
    """Raised by backends on unrecoverable failures (bad response, timeout)."""


@dataclass(frozen=True)
class VADResult:
    """Outcome of end-of-speech detection on one utterance.

    ``speech_end_ms`` is the offset inside the clip where speech ends;
    ``speech_detected`` is False for clips that contain no speech energy.
    """

    speech_detected: bool
    speech_end_ms: float
    detail: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class STTResult:
    """Final transcription of one utterance."""

    text: str


@runtime_checkable
class VADBackend(Protocol):
    """Voice activity detection: find the end of speech in an utterance."""

    name: str

    async def detect(self, audio: AudioClip) -> VADResult: ...


@runtime_checkable
class STTBackend(Protocol):
    """Speech-to-text: produce the final transcript of an utterance."""

    name: str

    async def transcribe(self, audio: AudioClip) -> STTResult: ...


@runtime_checkable
class LLMBackend(Protocol):
    """Agent brain: stream a text reply token by token.

    ``history`` is the running conversation as ``{"role", "content"}``
    dicts (OpenAI chat format); implementations may ignore it.
    """

    name: str

    def stream_reply(
        self, transcript: str, history: list[dict[str, str]]
    ) -> AsyncIterator[str]: ...


@runtime_checkable
class TTSBackend(Protocol):
    """Text-to-speech: stream synthesized audio chunks for a reply."""

    name: str

    def stream_speech(self, text: str) -> AsyncIterator[bytes]: ...


@dataclass
class BackendSet:
    """The four stage backends a pipeline run is wired to."""

    vad: VADBackend
    stt: STTBackend
    llm: LLMBackend
    tts: TTSBackend

    def describe(self) -> dict[str, str]:
        return {
            "vad": self.vad.name,
            "stt": self.stt.name,
            "llm": self.llm.name,
            "tts": self.tts.name,
        }
