# Contributing to voiceprobe

Thanks for your interest in improving voiceprobe. Bug reports, latency
methodology discussions and pull requests are all welcome.

## Development setup

Requirements: Python 3.10+ (no other runtime dependencies — the package
is stdlib-only by design).

```bash
git clone https://github.com/JaydenCJ/voiceprobe.git
cd voiceprobe
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```bash
python -m pytest          # full unit/integration suite
bash scripts/smoke.sh     # CLI end-to-end smoke test (prints SMOKE OK)
```

Test rules:

- Tests must never touch the network. HTTP backends are tested against
  the local fake server in `tests/fake_openai.py` (127.0.0.1 only).
- Latency-attribution tests must run on `SimulatedClock` and assert
  exact virtual-time numbers — avoid sleeps and wall-clock tolerances
  wherever a simulated clock can express the behavior.
- Never add model weights or large binary fixtures to the repository.

## Project principles

- **Stdlib-only runtime.** New runtime dependencies need a strong
  justification; prefer optional extras if something heavy is genuinely
  useful.
- **Measured, not fabricated.** Every number voiceprobe prints or renders
  must come from a measured span. Simulation parameters (mock latency
  profiles) must be clearly labeled as such.
- **Bring your own backend.** Inference stays behind the Protocols in
  `src/voiceprobe/backends/base.py`; voiceprobe itself never downloads
  or bundles models.
- Code comments and docstrings are written in English.

## Pull request guidelines

1. Open an issue first for anything larger than a small fix, so the
   approach can be discussed before you invest time.
2. Keep PRs focused: one logical change per PR.
3. Add or update tests for any behavior change; `python -m pytest` and
   `bash scripts/smoke.sh` must both pass.
4. Update `CHANGELOG.md` under an `Unreleased` heading and the README(s)
   if user-visible behavior changes. The three README files (`README.md`,
   `README.zh.md`, `README.ja.md`) must stay in sync.
5. If you touch the results JSON schema, bump `SCHEMA_VERSION` and keep
   `voiceprobe report` able to read the previous version, or state the
   break clearly in the changelog.

## Reporting issues

Please include: your Python version, the exact command you ran, the
scenario file (or a minimal reproduction), and the full stderr output.
For latency questions, attach the `results.json` produced with `--out`.
