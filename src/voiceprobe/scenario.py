"""Scenario model: the conversation a simulated caller plays against the agent.

A scenario is a JSON file with a list of turns. Each turn is what the
caller says — either as text (voiceprobe synthesizes deterministic
speech-shaped audio of a realistic duration) or as a path to a real WAV
recording. Example:

    {
      "name": "billing-support",
      "turns": [
        {"user_text": "Hi, I was double charged on my last invoice."},
        {"user_text": "The invoice number is 4 4 2 1.", "pause_ms": 800},
        {"audio_file": "recordings/complaint.wav", "user_text": "reference transcript"}
      ]
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from voiceprobe import audio as audio_mod
from voiceprobe.audio import AudioClip


class ScenarioError(Exception):
    """Raised for missing/invalid scenario files with a readable message."""


@dataclass(frozen=True)
class Turn:
    """One caller utterance.

    ``pause_ms`` is think-time before the caller starts speaking.
    ``audio_ms`` overrides the estimated duration of synthesized speech.
    """

    user_text: str | None = None
    audio_file: str | None = None
    audio_ms: float | None = None
    pause_ms: float = 250.0

    def resolve_audio(self, base_dir: Path, sample_rate: int) -> AudioClip:
        """Materialize the utterance audio for this turn."""
        if self.audio_file:
            clip = audio_mod.read_wav(base_dir / self.audio_file)
            return clip.with_hint(self.user_text)
        assert self.user_text is not None  # guaranteed by validation
        duration = self.audio_ms or audio_mod.estimate_speech_ms(self.user_text)
        clip = audio_mod.synthesize_speech_like(
            duration_ms=duration, sample_rate=sample_rate, seed=self.user_text
        )
        return clip.with_hint(self.user_text)


@dataclass(frozen=True)
class Scenario:
    """A named list of caller turns plus audio parameters."""

    name: str
    turns: tuple[Turn, ...]
    sample_rate: int = audio_mod.DEFAULT_SAMPLE_RATE
    base_dir: Path = field(default_factory=Path)


def default_scenario() -> Scenario:
    """Built-in three-turn support call used when no scenario file is given."""
    return Scenario(
        name="builtin-support-call",
        turns=(
            Turn(user_text="Hi, I was double charged on my last invoice."),
            Turn(user_text="The invoice number is 4 4 2 1 , from March.", pause_ms=600.0),
            Turn(user_text="Great, please send the refund confirmation by email.", pause_ms=400.0),
        ),
    )


def example_scenario_json() -> str:
    """JSON text written by ``voiceprobe init`` as a starting point."""
    doc = {
        "name": "billing-support",
        "sample_rate": 16000,
        "turns": [
            {"user_text": "Hi, I was double charged on my last invoice."},
            {"user_text": "The invoice number is 4 4 2 1 , from March.", "pause_ms": 800},
            {
                "user_text": "Great, please send the refund confirmation by email.",
                "audio_ms": 2600,
            },
        ],
    }
    return json.dumps(doc, indent=2) + "\n"


def _parse_turn(raw: object, index: int) -> Turn:
    if not isinstance(raw, dict):
        raise ScenarioError(f"turn #{index + 1} must be an object, got {type(raw).__name__}")
    user_text = raw.get("user_text")
    audio_file = raw.get("audio_file")
    if user_text is not None and not isinstance(user_text, str):
        raise ScenarioError(f"turn #{index + 1}: 'user_text' must be a string")
    if audio_file is not None and not isinstance(audio_file, str):
        raise ScenarioError(f"turn #{index + 1}: 'audio_file' must be a string path")
    if not user_text and not audio_file:
        raise ScenarioError(
            f"turn #{index + 1} needs 'user_text' (synthesized) or 'audio_file' (real WAV)"
        )
    audio_ms = raw.get("audio_ms")
    if audio_ms is not None:
        if not isinstance(audio_ms, (int, float)) or audio_ms <= 0:
            raise ScenarioError(f"turn #{index + 1}: 'audio_ms' must be a positive number")
    pause_ms = raw.get("pause_ms", 250.0)
    if not isinstance(pause_ms, (int, float)) or pause_ms < 0:
        raise ScenarioError(f"turn #{index + 1}: 'pause_ms' must be a non-negative number")
    unknown = set(raw) - {"user_text", "audio_file", "audio_ms", "pause_ms"}
    if unknown:
        raise ScenarioError(
            f"turn #{index + 1} has unknown fields: {', '.join(sorted(unknown))}"
        )
    return Turn(
        user_text=user_text,
        audio_file=audio_file,
        audio_ms=float(audio_ms) if audio_ms is not None else None,
        pause_ms=float(pause_ms),
    )


def load_scenario(path: str | Path) -> Scenario:
    """Load and validate a scenario JSON file."""
    p = Path(path)
    if not p.exists():
        raise ScenarioError(f"scenario file not found: {p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ScenarioError(f"scenario file {p} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ScenarioError(f"scenario file {p} must contain a JSON object at the top level")
    turns_raw = raw.get("turns")
    if not isinstance(turns_raw, list) or not turns_raw:
        raise ScenarioError(f"scenario file {p} needs a non-empty 'turns' array")
    turns = tuple(_parse_turn(t, i) for i, t in enumerate(turns_raw))
    sample_rate = raw.get("sample_rate", audio_mod.DEFAULT_SAMPLE_RATE)
    if not isinstance(sample_rate, int) or sample_rate < 8000:
        raise ScenarioError(f"scenario file {p}: 'sample_rate' must be an integer >= 8000")
    name = raw.get("name") or p.stem
    if not isinstance(name, str):
        raise ScenarioError(f"scenario file {p}: 'name' must be a string")
    return Scenario(name=name, turns=turns, sample_rate=sample_rate, base_dir=p.parent)
