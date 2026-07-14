# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-08

### Added

- Concurrent call simulation: N simultaneous scenario playthroughs on
  asyncio with optional linear ramp-up (`--calls`, `--ramp`) and a
  per-call safety timeout.
- Per-stage latency breakdown across the full voice pipeline
  (VAD -> STT -> LLM -> TTS): measured spans per turn with mean / p50 /
  p95 / p99 / max and per-stage share of turn latency.
- End-to-end metrics per turn: time to first LLM token and time to first
  TTS audio byte (caller-perceived response latency), with TTFT/TTFB
  markers recorded inside the LLM/TTS spans.
- Scenario files (JSON): text turns with synthesized deterministic
  speech-shaped audio, or real WAV recordings via `audio_file`;
  `voiceprobe init` writes an editable example.
- Backend abstraction (Python Protocols) for all four stages — no model
  weights shipped or downloaded:
  - deterministic mock stack with `fast` / `typical` / `slow` latency
    profiles and seeded jitter,
  - real dependency-free energy-based VAD,
  - HTTP adapters for OpenAI-compatible STT / chat-completions (SSE
    streaming) / TTS endpoints, with API keys taken from environment
    variables only.
- Self-contained HTML report (inline SVG, no JavaScript, no external
  assets): per-call latency waterfalls, a whole-run flame chart showing
  call concurrency, a first-audio latency histogram and stage tables.
- Raw results export (`--out results.json`, schema_version 1) and
  offline re-rendering via `voiceprobe report`.
- Terminal summary table with per-stage share bars.
- Deterministic virtual-time clock (`SimulatedClock`) used by the test
  suite to verify latency attribution exactly.
- Test suite (101 pytest cases) and `scripts/smoke.sh`; HTTP backends are
  tested against a local fake server on 127.0.0.1 only.
