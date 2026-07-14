"""Backend selection and configuration loading for the CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from voiceprobe.backends.base import BackendSet
from voiceprobe.backends.energy import EnergyVAD
from voiceprobe.backends.http import HttpLLM, HttpSTT, HttpTTS, PassthroughVAD
from voiceprobe.backends.mock import LatencyProfile, MockLLM, MockSTT, MockTTS, MockVAD
from voiceprobe.clock import Clock


class ConfigError(Exception):
    """Raised for invalid backend configuration with a readable message."""


def load_backend_config(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate an HTTP backend config file."""
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"backend config file not found: {p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ConfigError(f"backend config {p} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"backend config {p} must be a JSON object")
    for stage in ("stt", "llm", "tts"):
        section = raw.get(stage)
        if not isinstance(section, dict):
            raise ConfigError(
                f"backend config {p} needs an object section {stage!r} "
                "(with at least a 'url' field)"
            )
        if any(k in section for k in ("api_key", "apikey", "token", "secret")):
            raise ConfigError(
                f"backend config {p}, section {stage!r}: never put secrets in the "
                "config file; use 'api_key_env' with an environment variable name"
            )
    return raw


def build_backends(
    kind: str,
    clock: Clock,
    profile_name: str = "typical",
    seed: int = 0,
    vad_kind: str = "auto",
    backend_config: dict[str, Any] | None = None,
) -> BackendSet:
    """Assemble the four stage backends for a run.

    ``vad_kind='auto'`` picks mock VAD for the mock stack (so all four
    stages carry the profile's simulated latency) and the real energy VAD
    for HTTP stacks (which have no probeable remote VAD endpoint).
    """
    if kind == "mock":
        profile = LatencyProfile.named(profile_name)
        resolved_vad = "mock" if vad_kind == "auto" else vad_kind
        vad = _make_vad(resolved_vad, clock, profile, seed)
        return BackendSet(
            vad=vad,
            stt=MockSTT(clock, profile, seed),
            llm=MockLLM(clock, profile, seed),
            tts=MockTTS(clock, profile, seed),
        )
    if kind == "http":
        if backend_config is None:
            raise ConfigError(
                "the http backend needs --backend-config pointing at a JSON file "
                "with 'stt', 'llm' and 'tts' sections"
            )
        resolved_vad = "energy" if vad_kind == "auto" else vad_kind
        profile = LatencyProfile.named(profile_name)
        vad = _make_vad(resolved_vad, clock, profile, seed)
        return BackendSet(
            vad=vad,
            stt=HttpSTT(backend_config["stt"]),
            llm=HttpLLM(backend_config["llm"]),
            tts=HttpTTS(backend_config["tts"]),
        )
    raise ConfigError(f"unknown backend kind {kind!r}; choose 'mock' or 'http'")


def _make_vad(vad_kind: str, clock: Clock, profile: LatencyProfile, seed: int):
    if vad_kind == "mock":
        return MockVAD(clock, profile, seed)
    if vad_kind == "energy":
        return EnergyVAD()
    if vad_kind == "passthrough":
        return PassthroughVAD()
    raise ConfigError(
        f"unknown vad kind {vad_kind!r}; choose 'auto', 'energy', 'mock' or 'passthrough'"
    )
