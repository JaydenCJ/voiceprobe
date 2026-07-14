"""voiceprobe — load testing and per-stage latency profiling for voice agents.

voiceprobe simulates concurrent calls against a voice-agent pipeline
(VAD -> STT -> LLM -> TTS), measures every stage with monotonic timestamps,
and renders the results as terminal tables and self-contained HTML
waterfall / flamegraph reports.

The package is dependency-free at runtime (Python stdlib only).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
