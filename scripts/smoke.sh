#!/usr/bin/env bash
# Smoke test: exercise the voiceprobe CLI end to end with the mock stack.
# Self-asserting, offline (no network at all), idempotent, < 1 minute.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workdir="$(mktemp -d "${TMPDIR:-/tmp}/voiceprobe-smoke.XXXXXX")"
trap 'rm -rf "$workdir"' EXIT

PYTHON="${PYTHON:-python3}"
if [ -x "$here/.venv/bin/python" ]; then
  PYTHON="$here/.venv/bin/python"
fi

run_cli() {
  (cd "$workdir" && PYTHONPATH="$here/src" "$PYTHON" -m voiceprobe.cli "$@")
}

fail() {
  echo "SMOKE FAIL: $*" >&2
  exit 1
}

echo "[smoke] python: $PYTHON"
run_cli --version | grep -q "voiceprobe 0.1.0" || fail "--version output unexpected"

echo "[smoke] 1/4 init writes an editable scenario"
run_cli init --out scenario.json
[ -f "$workdir/scenario.json" ] || fail "scenario.json was not created"
"$PYTHON" -c "import json,sys; json.load(open(sys.argv[1]))" "$workdir/scenario.json" \
  || fail "scenario.json is not valid JSON"

echo "[smoke] 2/4 run a 4-call mock load test on the generated scenario"
run_cli run --scenario scenario.json --calls 4 --profile fast \
  --out results.json --html report.html > "$workdir/summary.txt"
[ -f "$workdir/results.json" ] || fail "results.json was not written"
[ -f "$workdir/report.html" ] || fail "report.html was not written"

echo "[smoke] 3/4 assert measured results"
"$PYTHON" - "$workdir/results.json" <<'PYEOF'
import json, sys
doc = json.load(open(sys.argv[1]))
assert doc["schema_version"] == 1, "unexpected schema version"
assert doc["ok_calls"] == 4 and doc["failed_calls"] == 0, "calls failed"
turn = doc["calls"][0]["turns"][0]
stages = [s["stage"] for s in turn["spans"]]
assert stages == ["vad", "stt", "llm", "tts"], f"unexpected stages: {stages}"
assert all(s["duration_ms"] > 0 for s in turn["spans"]), "non-positive span duration"
assert turn["first_audio_ms"] and turn["first_audio_ms"] > 0, "missing e2e latency"
print("[smoke] results.json ok:", doc["ok_calls"], "calls,",
      sum(len(c["turns"]) for c in doc["calls"]), "turns")
PYEOF

grep -q "first audio (e2e)" "$workdir/summary.txt" || fail "summary table missing e2e line"
grep -q "llm" "$workdir/summary.txt" || fail "summary table missing llm stage"

echo "[smoke] 4/4 assert the HTML report is a self-contained visual"
grep -q "<svg" "$workdir/report.html" || fail "report has no SVG charts"
grep -q "Flame chart" "$workdir/report.html" || fail "report has no flame chart section"
grep -q "Waterfalls" "$workdir/report.html" || fail "report has no waterfall section"
if grep -q "<script" "$workdir/report.html"; then fail "report contains scripts"; fi
if grep -q 'src="http' "$workdir/report.html"; then fail "report references external assets"; fi

run_cli report results.json --html report2.html
grep -q "<svg" "$workdir/report2.html" || fail "re-rendered report has no SVG"

echo "SMOKE OK"
