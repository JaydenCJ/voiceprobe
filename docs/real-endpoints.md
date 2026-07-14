# Profiling real endpoints

voiceprobe ships no models. The `--backend http` adapters speak the three
OpenAI-style routes, so any service that implements them — hosted OpenAI, a
gateway of your own, or fully local servers — can be profiled without code
changes. This page documents the exact wire contract the adapters implement
(source of truth: [`src/voiceprobe/backends/http.py`](../src/voiceprobe/backends/http.py))
and gives two ready-to-adapt configurations.

## The adapter contract

The `--backend-config` file is a JSON object with three object sections —
`stt`, `llm`, `tts` — each needing at least a `url`
(validated by `load_backend_config` in
[`src/voiceprobe/config.py`](../src/voiceprobe/config.py)). Putting
`api_key` / `apikey` / `token` / `secret` keys in any section is rejected;
secrets are referenced by environment-variable *name* via `api_key_env`.

| Stage | Request | Expected response | Config keys (defaults) |
|---|---|---|---|
| `stt` | `POST url`, `multipart/form-data` with form fields `model` (default `whisper-1`), optional `language`, and the utterance as file field `file` (`utterance.wav`, `audio/wav`) | JSON body with a string `text` field | `url` (required), `model`, `language`, `timeout_s` (30), `api_key_env`, `headers` |
| `llm` | `POST url`, JSON `{"model", "messages", "stream": true}` plus `max_tokens` if configured; `system_prompt` is prepended as a `system` message | SSE stream of `data:` lines with `choices[0].delta.content` deltas, terminated by `data: [DONE]` | `url` (required), `model` (`gpt-4o-mini`), `system_prompt`, `max_tokens`, `timeout_s` (60), `api_key_env`, `headers` |
| `tts` | `POST url`, JSON `{"model", "input", "voice", "response_format"}` | streamed binary audio body (bytes are timed as they arrive) | `url` (required), `model` (`tts-1`), `voice` (`alloy`), `response_format` (`wav`), `timeout_s` (60), `api_key_env`, `headers` |

Notes that matter when pointing these at real services:

- **Auth**: if `api_key_env` is set, the adapter sends
  `Authorization: Bearer $VALUE` where `$VALUE` is read from that environment
  variable at runtime; a missing variable is a hard error. Extra static
  headers can be added per stage via a `headers` object.
- **Proxies are bypassed**: connections are made directly (environment
  `HTTP(S)_PROXY` is deliberately ignored) so measured latencies reflect the
  target service, not a proxy hop. The target URL must be reachable directly
  from the machine running voiceprobe.
- **VAD stays local**: HTTP stacks have no probeable remote VAD, so
  `--vad auto` resolves to the built-in energy VAD; use `--vad passthrough`
  to treat each utterance as all speech and exclude VAD from the comparison.
- **TTFT/TTFB are real**: the LLM adapter times the first SSE content delta
  and the TTS adapter times the first body chunk as they arrive over the
  socket, so the endpoint must actually stream (OpenAI-compatible servers
  listed below all do).

## Example A — hosted OpenAI endpoints

[`docs/backends.example.json`](backends.example.json) targets
`api.openai.com` for all three stages and only needs `OPENAI_API_KEY`
exported:

```bash
export OPENAI_API_KEY=sk-...
voiceprobe run --scenario scenario.json --calls 10 --backend http \
  --backend-config docs/backends.example.json --out results.json --html report.html
```

## Example B — fully local, no API keys

A common self-hosted trio that keeps audio on your machine. Each server
implements the OpenAI-compatible route the corresponding adapter expects:

- **LLM — [Ollama](https://ollama.com)**: exposes OpenAI-compatible
  streaming chat completions at `http://127.0.0.1:11434/v1/chat/completions`.
  Setup: `ollama pull llama3.2:1b` (any pulled model name works as `model`).
- **STT — [whisper.cpp](https://github.com/ggml-org/whisper.cpp) server**:
  `whisper-server -m models/ggml-tiny.en.bin --port 8081` accepts multipart
  `file` uploads on `/inference` and answers `{"text": "..."}`; the extra
  `model` form field the adapter always sends is ignored by the server.
- **TTS — [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI)**:
  serves OpenAI-compatible `POST /v1/audio/speech` (streamed WAV) on port
  8880, e.g. `docker run -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu`.

Save as `backends.local.json` (no `api_key_env` — none of these need keys):

```json
{
  "stt": {
    "url": "http://127.0.0.1:8081/inference",
    "model": "ggml-tiny.en",
    "timeout_s": 60
  },
  "llm": {
    "url": "http://127.0.0.1:11434/v1/chat/completions",
    "model": "llama3.2:1b",
    "system_prompt": "You are a concise phone support agent.",
    "max_tokens": 200,
    "timeout_s": 120
  },
  "tts": {
    "url": "http://127.0.0.1:8880/v1/audio/speech",
    "model": "kokoro",
    "voice": "af_heart",
    "response_format": "wav",
    "timeout_s": 120
  }
}
```

```bash
voiceprobe run --scenario scenario.json --calls 10 --ramp 2 \
  --backend http --backend-config backends.local.json \
  --out results.json --html report.html
```

Expect the summary table's `stt`/`llm`/`tts` rows to now show your servers'
real latencies (first cold call is typically slower while models load;
increase `timeout_s` or discard the first call when comparing runs).

## What is verified, and how

- The request/response contract in the table above is taken line-by-line
  from `src/voiceprobe/backends/http.py`; the config-file structure and the
  inline-secret rejection from `src/voiceprobe/config.py`.
- The full HTTP path — multipart STT upload, SSE delta parsing with
  `[DONE]`, streamed TTS bytes, error propagation — is exercised end-to-end
  against a real loopback HTTP server in `tests/test_http_backends.py`
  (server implementation: `tests/fake_openai.py`), which is also how the
  documented `--backend http --backend-config` command line is executed in
  CI-less test runs.
- The third-party servers in Example B follow their published
  OpenAI-compatible routes; their APIs can drift between releases, so treat
  `url` / `model` / `voice` values as starting points and check the
  server's own docs if a stage errors out.
