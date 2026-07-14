"""Backend abstraction layer.

voiceprobe never ships or downloads model weights. Every inference stage
(VAD, STT, LLM, TTS) is a small Protocol; you bring your own backend:

- :mod:`voiceprobe.backends.mock` — deterministic simulated backends with
  configurable latency profiles (default; zero external services).
- :mod:`voiceprobe.backends.energy` — a real, dependency-free energy-based
  VAD implementation.
- :mod:`voiceprobe.backends.http` — adapters for OpenAI-compatible
  STT / chat-completion / TTS HTTP endpoints you host or subscribe to.
"""

from voiceprobe.backends.base import (
    BackendError,
    LLMBackend,
    STTBackend,
    STTResult,
    TTSBackend,
    VADBackend,
    VADResult,
)

__all__ = [
    "BackendError",
    "LLMBackend",
    "STTBackend",
    "STTResult",
    "TTSBackend",
    "VADBackend",
    "VADResult",
]
