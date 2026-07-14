"""HTTP backends for OpenAI-compatible STT / chat / TTS endpoints.

voiceprobe ships no models. These adapters let you point each pipeline
stage at a service you already run or subscribe to (OpenAI, a local
whisper.cpp / llama.cpp / vLLM / Piper server, or your own gateway) —
anything that speaks the OpenAI-style routes:

- STT:  ``POST /v1/audio/transcriptions`` (multipart)  -> ``{"text": ...}``
- LLM:  ``POST /v1/chat/completions`` with ``stream: true`` (SSE deltas)
- TTS:  ``POST /v1/audio/speech`` -> streamed binary audio

Design notes:

- Connections are direct (environment HTTP proxies are ignored) so that
  measured latencies reflect the target service, not a proxy hop.
- API keys are read from environment variables named in the config
  (``api_key_env``); keys never appear in config files, logs or reprs.
- Blocking I/O runs on a bounded thread pool; response bytes are handed
  to the event loop as they arrive so time-to-first-token/byte is real.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator

from voiceprobe.audio import AudioClip, write_wav
from voiceprobe.backends.base import BackendError, STTResult, VADResult

_MAX_WORKERS = 64
_executor: ThreadPoolExecutor | None = None

# Direct opener: never route load-test traffic through env proxies.
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=_MAX_WORKERS, thread_name_prefix="voiceprobe-http"
        )
    return _executor


def _resolve_api_key(config: dict[str, Any], stage: str) -> str | None:
    """Resolve the API key for a stage from the environment, never inline.

    Config carries only the *name* of an environment variable so that
    secrets stay out of scenario/config files and result artifacts.
    """
    env_name = config.get("api_key_env")
    if not env_name:
        return None
    value = os.environ.get(str(env_name))
    if not value:
        raise BackendError(
            f"{stage}: environment variable {env_name!r} named by 'api_key_env' is not set"
        )
    return value


def _build_headers(config: dict[str, Any], stage: str, content_type: str) -> dict[str, str]:
    headers = {"Content-Type": content_type, "Accept": "*/*"}
    extra = config.get("headers") or {}
    if not isinstance(extra, dict):
        raise BackendError(f"{stage}: 'headers' must be an object of string pairs")
    headers.update({str(k): str(v) for k, v in extra.items()})
    api_key = _resolve_api_key(config, stage)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _require_url(config: dict[str, Any], stage: str) -> str:
    url = config.get("url")
    if not url or not isinstance(url, str):
        raise BackendError(f"{stage}: backend config needs a 'url' field")
    if not url.startswith(("http://", "https://")):
        raise BackendError(f"{stage}: 'url' must start with http:// or https:// (got {url!r})")
    return url


async def _stream_response(
    request: urllib.request.Request,
    timeout: float,
    stage: str,
    by_line: bool,
    chunk_size: int = 4096,
) -> AsyncIterator[bytes]:
    """Yield response bytes (or lines) as they arrive over the wire."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    def put(kind: str, payload: Any) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (kind, payload))

    def worker() -> None:
        try:
            with _opener.open(request, timeout=timeout) as resp:
                if by_line:
                    for line in resp:
                        put("data", line)
                else:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        put("data", chunk)
            put("end", None)
        except urllib.error.HTTPError as exc:
            body = b""
            try:
                body = exc.read(500)
            except Exception:
                body = b""
            put(
                "error",
                BackendError(
                    f"{stage}: HTTP {exc.code} from {request.full_url}: "
                    f"{body.decode('utf-8', 'replace')[:200]}"
                ),
            )
        except Exception as exc:
            put("error", BackendError(f"{stage}: request to {request.full_url} failed: {exc}"))

    _get_executor().submit(worker)
    while True:
        kind, payload = await queue.get()
        if kind == "data":
            yield payload
        elif kind == "end":
            return
        else:
            raise payload


async def _request_bytes(request: urllib.request.Request, timeout: float, stage: str) -> bytes:
    chunks = []
    async for chunk in _stream_response(request, timeout, stage, by_line=False):
        chunks.append(chunk)
    return b"".join(chunks)


def _encode_multipart(fields: dict[str, str], file_field: str, filename: str, file_bytes: bytes) -> tuple[bytes, str]:
    """Encode a multipart/form-data body (stdlib has no helper for this)."""
    boundary = f"voiceprobe-{uuid.uuid4().hex}"
    out = io.BytesIO()
    for key, value in fields.items():
        out.write(f"--{boundary}\r\n".encode())
        out.write(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
        out.write(f"{value}\r\n".encode())
    out.write(f"--{boundary}\r\n".encode())
    out.write(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'.encode()
    )
    out.write(b"Content-Type: audio/wav\r\n\r\n")
    out.write(file_bytes)
    out.write(f"\r\n--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"


class PassthroughVAD:
    """Treats the full utterance as speech; use when the target stack has no
    separately probeable VAD stage (common for HTTP-only setups)."""

    name = "passthrough-vad"

    async def detect(self, audio: AudioClip) -> VADResult:
        return VADResult(speech_detected=audio.num_samples > 0, speech_end_ms=audio.duration_ms)


class HttpSTT:
    """OpenAI-compatible transcription endpoint adapter."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._url = _require_url(config, "stt")
        self._timeout = float(config.get("timeout_s", 30.0))
        self.name = f"http-stt({self._url})"

    async def transcribe(self, audio: AudioClip) -> STTResult:
        buf = io.BytesIO()
        write_wav(buf, audio)
        fields = {"model": str(self._config.get("model", "whisper-1"))}
        if self._config.get("language"):
            fields["language"] = str(self._config["language"])
        body, content_type = _encode_multipart(fields, "file", "utterance.wav", buf.getvalue())
        headers = _build_headers(self._config, "stt", content_type)
        request = urllib.request.Request(self._url, data=body, headers=headers, method="POST")
        raw = await _request_bytes(request, self._timeout, "stt")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise BackendError(f"stt: response is not valid JSON: {exc}") from exc
        text = payload.get("text")
        if not isinstance(text, str):
            raise BackendError("stt: response JSON has no 'text' string field")
        return STTResult(text=text)


class HttpLLM:
    """OpenAI-compatible chat-completions adapter with SSE streaming."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._url = _require_url(config, "llm")
        self._timeout = float(config.get("timeout_s", 60.0))
        self.name = f"http-llm({self._url})"

    async def stream_reply(
        self, transcript: str, history: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        messages: list[dict[str, str]] = []
        system_prompt = self._config.get("system_prompt")
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.extend(history)
        messages.append({"role": "user", "content": transcript})
        payload = {
            "model": str(self._config.get("model", "gpt-4o-mini")),
            "messages": messages,
            "stream": True,
        }
        if "max_tokens" in self._config:
            payload["max_tokens"] = int(self._config["max_tokens"])
        body = json.dumps(payload).encode("utf-8")
        headers = _build_headers(self._config, "llm", "application/json")
        request = urllib.request.Request(self._url, data=body, headers=headers, method="POST")
        got_any = False
        async for line in _stream_response(request, self._timeout, "llm", by_line=True):
            text = line.decode("utf-8", "replace").strip()
            if not text.startswith("data:"):
                continue
            data = text[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except ValueError as exc:
                raise BackendError(f"llm: malformed SSE JSON event: {data[:120]}") from exc
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content:
                got_any = True
                yield content
        if not got_any:
            raise BackendError("llm: stream ended without any content deltas")


class HttpTTS:
    """OpenAI-compatible speech-synthesis adapter (streamed binary body)."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._url = _require_url(config, "tts")
        self._timeout = float(config.get("timeout_s", 60.0))
        self.name = f"http-tts({self._url})"

    async def stream_speech(self, text: str) -> AsyncIterator[bytes]:
        payload = {
            "model": str(self._config.get("model", "tts-1")),
            "input": text,
            "voice": str(self._config.get("voice", "alloy")),
            "response_format": str(self._config.get("response_format", "wav")),
        }
        body = json.dumps(payload).encode("utf-8")
        headers = _build_headers(self._config, "tts", "application/json")
        request = urllib.request.Request(self._url, data=body, headers=headers, method="POST")
        got_any = False
        async for chunk in _stream_response(request, self._timeout, "tts", by_line=False):
            got_any = True
            yield chunk
        if not got_any:
            raise BackendError("tts: response body was empty")
