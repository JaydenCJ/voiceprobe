"""Local fake OpenAI-compatible server for HTTP backend tests.

Binds to 127.0.0.1 on an ephemeral port; tests never leave the loopback
interface. Implements just enough of the three routes voiceprobe talks to
and records every request for assertions.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@dataclass
class RecordedRequest:
    path: str
    headers: dict[str, str]
    body: bytes


@dataclass
class FakeOpenAIServer:
    """Threaded fake server; use as a context manager."""

    sse_tokens: tuple[str, ...] = ("Hello", " from", " the", " fake", " agent.")
    tts_chunks: int = 3
    stt_text: str = "fake transcript from server"
    inter_chunk_delay_s: float = 0.01
    requests: list[RecordedRequest] = field(default_factory=list)

    def __post_init__(self) -> None:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *args: object) -> None:
                # Keep pytest output clean.
                return

            def _read_body(self) -> bytes:
                length = int(self.headers.get("Content-Length") or 0)
                return self.rfile.read(length) if length else b""

            def do_POST(self) -> None:  # noqa: N802 (http.server API name)
                body = self._read_body()
                outer.requests.append(
                    RecordedRequest(
                        path=self.path,
                        headers={k: v for k, v in self.headers.items()},
                        body=body,
                    )
                )
                if self.path == "/v1/audio/transcriptions":
                    payload = json.dumps({"text": outer.stt_text}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                elif self.path == "/v1/chat/completions":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for token in outer.sse_tokens:
                        event = {"choices": [{"delta": {"content": token}}]}
                        self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                        self.wfile.flush()
                        time.sleep(outer.inter_chunk_delay_s)
                    self.wfile.write(b"data: [DONE]\n\n")
                elif self.path == "/v1/audio/speech":
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/wav")
                    self.end_headers()
                    for i in range(outer.tts_chunks):
                        self.wfile.write(bytes([i]) * 256)
                        self.wfile.flush()
                        time.sleep(outer.inter_chunk_delay_s)
                elif self.path == "/boom":
                    payload = b'{"error": "kaboom"}'
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                else:
                    self.send_response(404)
                    self.end_headers()

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def url(self, path: str) -> str:
        return self.base_url + path

    def __enter__(self) -> "FakeOpenAIServer":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
