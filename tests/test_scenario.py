"""Scenario loading, validation and audio resolution."""

from __future__ import annotations

import json

import pytest

from voiceprobe.audio import synthesize_speech_like, write_wav
from voiceprobe.scenario import (
    ScenarioError,
    Turn,
    default_scenario,
    example_scenario_json,
    load_scenario,
)


def write_scenario(tmp_path, doc) -> str:
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


def test_load_valid_scenario_with_defaults(tmp_path):
    path = write_scenario(
        tmp_path,
        {
            "name": "demo",
            "turns": [
                {"user_text": "hello"},
                {"user_text": "goodbye", "pause_ms": 800, "audio_ms": 1500},
            ],
        },
    )
    scenario = load_scenario(path)
    assert scenario.name == "demo"
    assert len(scenario.turns) == 2
    assert scenario.turns[0].pause_ms == 250.0
    assert scenario.turns[1].audio_ms == 1500.0
    assert scenario.sample_rate == 16000


def test_name_defaults_to_file_stem(tmp_path):
    path = write_scenario(tmp_path, {"turns": [{"user_text": "hi"}]})
    assert load_scenario(path).name == "scenario"


def test_missing_file_error():
    with pytest.raises(ScenarioError, match="not found"):
        load_scenario("/nonexistent/path/scenario.json")


def test_invalid_json_error(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ScenarioError, match="not valid JSON"):
        load_scenario(path)


def test_empty_turns_rejected(tmp_path):
    path = write_scenario(tmp_path, {"turns": []})
    with pytest.raises(ScenarioError, match="non-empty 'turns'"):
        load_scenario(path)


def test_turn_needs_text_or_audio(tmp_path):
    path = write_scenario(tmp_path, {"turns": [{"pause_ms": 100}]})
    with pytest.raises(ScenarioError, match="turn #1"):
        load_scenario(path)


def test_unknown_turn_fields_rejected(tmp_path):
    path = write_scenario(tmp_path, {"turns": [{"user_text": "hi", "speaker": "bob"}]})
    with pytest.raises(ScenarioError, match="unknown fields: speaker"):
        load_scenario(path)


def test_negative_pause_rejected(tmp_path):
    path = write_scenario(tmp_path, {"turns": [{"user_text": "hi", "pause_ms": -5}]})
    with pytest.raises(ScenarioError, match="pause_ms"):
        load_scenario(path)


def test_bad_sample_rate_rejected(tmp_path):
    path = write_scenario(tmp_path, {"turns": [{"user_text": "hi"}], "sample_rate": 4000})
    with pytest.raises(ScenarioError, match="sample_rate"):
        load_scenario(path)


def test_resolve_audio_synthesizes_with_hint(tmp_path):
    turn = Turn(user_text="please check my booking", audio_ms=1200.0)
    clip = turn.resolve_audio(tmp_path, 16000)
    assert clip.transcript_hint == "please check my booking"
    assert clip.duration_ms == pytest.approx(1200, abs=120)


def test_resolve_audio_loads_real_wav_relative_to_scenario(tmp_path):
    recorded = synthesize_speech_like(700, seed="recorded")
    write_wav(tmp_path / "utt.wav", recorded)
    turn = Turn(user_text="reference transcript", audio_file="utt.wav")
    clip = turn.resolve_audio(tmp_path, 16000)
    assert clip.pcm == recorded.pcm
    assert clip.transcript_hint == "reference transcript"


def test_default_scenario_and_example_json_are_valid(tmp_path):
    scenario = default_scenario()
    assert len(scenario.turns) >= 3
    path = tmp_path / "example.json"
    path.write_text(example_scenario_json(), encoding="utf-8")
    loaded = load_scenario(path)
    assert loaded.name == "billing-support"
    assert all(t.user_text for t in loaded.turns)
