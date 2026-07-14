"""CLI end-to-end: real runs through the public entry point.

``main`` is invoked in-process; runs use the mock stack on a real clock
(short scenarios keep them fast) or the local fake HTTP server.
"""

from __future__ import annotations

import json

import pytest

from voiceprobe.cli import main
from fake_openai import FakeOpenAIServer

SHORT_SCENARIO = {
    "name": "cli-test",
    "turns": [{"user_text": "Hello there, quick check.", "audio_ms": 700, "pause_ms": 0}],
}


def write_short_scenario(tmp_path) -> str:
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(SHORT_SCENARIO), encoding="utf-8")
    return str(path)


def test_run_writes_results_and_report(tmp_path, capsys):
    scenario = write_short_scenario(tmp_path)
    out = tmp_path / "results.json"
    html = tmp_path / "report.html"
    code = main(
        [
            "run",
            "--scenario",
            scenario,
            "--calls",
            "2",
            "--profile",
            "fast",
            "--out",
            str(out),
            "--html",
            str(html),
        ]
    )
    assert code == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["scenario"] == "cli-test"
    assert len(doc["calls"]) == 2
    assert doc["ok_calls"] == 2
    stages = [s["stage"] for s in doc["calls"][0]["turns"][0]["spans"]]
    assert stages == ["vad", "stt", "llm", "tts"]
    report = html.read_text(encoding="utf-8")
    assert report.startswith("<!DOCTYPE html>") and "<svg" in report
    captured = capsys.readouterr()
    assert "stage" in captured.out and "llm" in captured.out
    assert "first audio (e2e)" in captured.out


def test_run_is_deterministic_with_seed_for_transcripts(tmp_path):
    scenario = write_short_scenario(tmp_path)
    out_a, out_b = tmp_path / "a.json", tmp_path / "b.json"
    for out in (out_a, out_b):
        assert (
            main(
                [
                    "run",
                    "--scenario",
                    scenario,
                    "--profile",
                    "fast",
                    "--seed",
                    "9",
                    "--quiet",
                    "--out",
                    str(out),
                ]
            )
            == 0
        )
    turn_a = json.loads(out_a.read_text())["calls"][0]["turns"][0]
    turn_b = json.loads(out_b.read_text())["calls"][0]["turns"][0]
    assert turn_a["transcript"] == turn_b["transcript"] == "Hello there, quick check."
    assert turn_a["reply"] == turn_b["reply"]


def test_run_with_missing_scenario_is_usage_error(tmp_path, capsys):
    code = main(["run", "--scenario", str(tmp_path / "missing.json"), "--quiet"])
    assert code == 2
    assert "not found" in capsys.readouterr().err


def test_run_http_backend_without_config_is_usage_error(capsys):
    code = main(["run", "--backend", "http", "--quiet"])
    assert code == 2
    assert "--backend-config" in capsys.readouterr().err


def test_run_invalid_calls_value(capsys):
    code = main(["run", "--calls", "0", "--quiet"])
    assert code == 2
    assert "--calls" in capsys.readouterr().err


def test_run_against_local_http_stack(tmp_path):
    scenario = write_short_scenario(tmp_path)
    out = tmp_path / "http-results.json"
    with FakeOpenAIServer() as server:
        config_path = tmp_path / "backends.json"
        config_path.write_text(
            json.dumps(
                {
                    "stt": {"url": server.url("/v1/audio/transcriptions")},
                    "llm": {"url": server.url("/v1/chat/completions"), "model": "m"},
                    "tts": {"url": server.url("/v1/audio/speech")},
                }
            ),
            encoding="utf-8",
        )
        code = main(
            [
                "run",
                "--scenario",
                scenario,
                "--backend",
                "http",
                "--backend-config",
                str(config_path),
                "--quiet",
                "--out",
                str(out),
            ]
        )
    assert code == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["backends"]["stt"].startswith("http-stt(")
    assert doc["backends"]["vad"].startswith("energy-vad")  # auto VAD for http
    turn = doc["calls"][0]["turns"][0]
    assert turn["transcript"] == "fake transcript from server"
    assert turn["reply"] == "Hello from the fake agent."
    assert turn["first_audio_ms"] > 0


def test_report_command_rerenders_from_results(tmp_path, capsys):
    scenario = write_short_scenario(tmp_path)
    out = tmp_path / "results.json"
    assert main(["run", "--scenario", scenario, "--profile", "fast", "--quiet", "--out", str(out)]) == 0
    html = tmp_path / "again.html"
    assert main(["report", str(out), "--html", str(html)]) == 0
    assert "<svg" in html.read_text(encoding="utf-8")


def test_report_command_on_missing_file(capsys, tmp_path):
    assert main(["report", str(tmp_path / "nope.json")]) == 2
    assert "not found" in capsys.readouterr().err


def test_report_command_on_non_results_json(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text('{"hello": 1}', encoding="utf-8")
    assert main(["report", str(bad)]) == 2
    assert "does not look like voiceprobe results" in capsys.readouterr().err


def test_init_writes_editable_scenario(tmp_path, capsys):
    target = tmp_path / "new-scenario.json"
    assert main(["init", "--out", str(target)]) == 0
    doc = json.loads(target.read_text(encoding="utf-8"))
    assert doc["turns"]
    # Refuses to clobber without --force.
    assert main(["init", "--out", str(target)]) == 2
    assert "already exists" in capsys.readouterr().err
    assert main(["init", "--out", str(target), "--force"]) == 0


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "voiceprobe 0.1.0" in capsys.readouterr().out
