"""HTTP backends against a local fake OpenAI-compatible server (127.0.0.1).

No test in this module talks to anything outside the loopback interface.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from voiceprobe.audio import synthesize_speech_like
from voiceprobe.backends.base import BackendError
from voiceprobe.backends.http import HttpLLM, HttpSTT, HttpTTS, PassthroughVAD
from voiceprobe.config import ConfigError, load_backend_config
from fake_openai import FakeOpenAIServer

AUDIO = synthesize_speech_like(600, seed="http test")


def test_http_stt_uploads_multipart_and_parses_text():
    with FakeOpenAIServer() as server:
        stt = HttpSTT({"url": server.url("/v1/audio/transcriptions"), "model": "whisper-1"})
        result = asyncio.run(stt.transcribe(AUDIO))
        assert result.text == "fake transcript from server"
        req = server.requests[0]
        assert "multipart/form-data" in req.headers.get("Content-Type", "")
        assert b'name="model"' in req.body and b"whisper-1" in req.body
        assert b'filename="utterance.wav"' in req.body
        assert b"RIFF" in req.body  # a real WAV payload was sent


def test_http_stt_sends_bearer_token_from_env(monkeypatch):
    monkeypatch.setenv("VOICEPROBE_TEST_KEY", "sk-unit-test-value")
    with FakeOpenAIServer() as server:
        stt = HttpSTT(
            {"url": server.url("/v1/audio/transcriptions"), "api_key_env": "VOICEPROBE_TEST_KEY"}
        )
        asyncio.run(stt.transcribe(AUDIO))
        assert server.requests[0].headers.get("Authorization") == "Bearer sk-unit-test-value"


def test_http_stt_missing_env_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("VOICEPROBE_MISSING_KEY", raising=False)
    stt = HttpSTT({"url": "http://127.0.0.1:1/x", "api_key_env": "VOICEPROBE_MISSING_KEY"})
    with pytest.raises(BackendError, match="VOICEPROBE_MISSING_KEY"):
        asyncio.run(stt.transcribe(AUDIO))


def test_http_llm_streams_sse_deltas_in_order():
    with FakeOpenAIServer() as server:
        llm = HttpLLM(
            {
                "url": server.url("/v1/chat/completions"),
                "model": "test-model",
                "system_prompt": "You are a support agent.",
            }
        )

        async def collect():
            return [t async for t in llm.stream_reply("hi", [{"role": "user", "content": "x"}])]

        tokens = asyncio.run(collect())
        assert "".join(tokens) == "Hello from the fake agent."
        sent = json.loads(server.requests[0].body)
        assert sent["stream"] is True
        assert sent["model"] == "test-model"
        assert sent["messages"][0] == {"role": "system", "content": "You are a support agent."}
        assert sent["messages"][-1] == {"role": "user", "content": "hi"}


def test_http_tts_streams_binary_chunks():
    with FakeOpenAIServer(tts_chunks=4) as server:
        tts = HttpTTS({"url": server.url("/v1/audio/speech"), "voice": "alloy"})

        async def collect():
            return [c async for c in tts.stream_speech("hello caller")]

        chunks = asyncio.run(collect())
        assert len(b"".join(chunks)) == 4 * 256
        sent = json.loads(server.requests[0].body)
        assert sent["input"] == "hello caller"
        assert sent["voice"] == "alloy"


def test_http_error_status_becomes_backend_error():
    with FakeOpenAIServer() as server:
        stt = HttpSTT({"url": server.url("/boom")})
        with pytest.raises(BackendError, match="HTTP 500"):
            asyncio.run(stt.transcribe(AUDIO))


def test_connection_refused_becomes_backend_error():
    # Port 1 on loopback is never listening.
    stt = HttpSTT({"url": "http://127.0.0.1:1/v1/audio/transcriptions", "timeout_s": 2})
    with pytest.raises(BackendError, match="failed"):
        asyncio.run(stt.transcribe(AUDIO))


def test_url_validation():
    with pytest.raises(BackendError, match="'url'"):
        HttpSTT({})
    with pytest.raises(BackendError, match="http"):
        HttpLLM({"url": "ftp://example.invalid/llm"})


def test_backend_names_never_contain_secrets(monkeypatch):
    monkeypatch.setenv("VOICEPROBE_TEST_KEY", "sk-super-secret-0123456789")
    stt = HttpSTT(
        {"url": "http://127.0.0.1:9/v1/audio/transcriptions", "api_key_env": "VOICEPROBE_TEST_KEY"}
    )
    assert "sk-super-secret" not in stt.name
    assert "sk-super-secret" not in repr(vars(stt))


def test_passthrough_vad_accepts_speech_and_flags_empty():
    result = asyncio.run(PassthroughVAD().detect(AUDIO))
    assert result.speech_detected
    from voiceprobe.audio import AudioClip

    empty = asyncio.run(PassthroughVAD().detect(AudioClip(pcm=b"")))
    assert not empty.speech_detected


def test_load_backend_config_rejects_inline_secrets(tmp_path):
    path = tmp_path / "backends.json"
    path.write_text(
        json.dumps(
            {
                "stt": {"url": "http://127.0.0.1:9/stt", "api_key": "sk-oops"},
                "llm": {"url": "http://127.0.0.1:9/llm"},
                "tts": {"url": "http://127.0.0.1:9/tts"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="api_key_env"):
        load_backend_config(path)


def test_load_backend_config_requires_all_sections(tmp_path):
    path = tmp_path / "backends.json"
    path.write_text(json.dumps({"stt": {"url": "http://127.0.0.1:9/stt"}}), encoding="utf-8")
    with pytest.raises(ConfigError, match="llm"):
        load_backend_config(path)
