"""voiceprobe command-line interface.

Subcommands:

- ``voiceprobe run``    — simulate concurrent calls, print the per-stage
  latency breakdown and optionally write results JSON + HTML report.
- ``voiceprobe report`` — re-render an HTML report from a results JSON.
- ``voiceprobe init``   — write an editable example scenario file.

Exit codes: 0 success, 1 runtime failure (e.g. every call failed),
2 invalid input or usage.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from voiceprobe import __version__
from voiceprobe.audio import AudioError
from voiceprobe.backends.base import BackendError
from voiceprobe.clock import MonotonicClock
from voiceprobe.config import ConfigError, build_backends, load_backend_config
from voiceprobe.metrics import compute_metrics, format_summary
from voiceprobe.report import render_report
from voiceprobe.runner import LoadTestConfig, run_load_test
from voiceprobe.scenario import (
    ScenarioError,
    default_scenario,
    example_scenario_json,
    load_scenario,
)

_USAGE_ERROR = 2
_RUNTIME_ERROR = 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voiceprobe",
        description=(
            "Load testing and per-stage latency profiling (VAD/STT/LLM/TTS) "
            "for voice agents."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"voiceprobe {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser(
        "run", help="simulate concurrent calls and profile per-stage latency"
    )
    run_p.add_argument(
        "--scenario",
        metavar="FILE",
        help="scenario JSON (default: a built-in 3-turn support call)",
    )
    run_p.add_argument(
        "--calls", type=int, default=1, metavar="N", help="concurrent calls (default 1)"
    )
    run_p.add_argument(
        "--ramp",
        type=float,
        default=0.0,
        metavar="SEC",
        help="ramp calls up linearly over SEC seconds (default 0: all at once)",
    )
    run_p.add_argument(
        "--backend",
        choices=("mock", "http"),
        default="mock",
        help="stage backends: deterministic mock stack or your own HTTP endpoints",
    )
    run_p.add_argument(
        "--backend-config",
        metavar="FILE",
        help="JSON config for --backend http (sections: stt, llm, tts)",
    )
    run_p.add_argument(
        "--profile",
        choices=("fast", "typical", "slow"),
        default="typical",
        help="latency profile for the mock stack (default typical)",
    )
    run_p.add_argument(
        "--vad",
        choices=("auto", "energy", "mock", "passthrough"),
        default="auto",
        help="VAD stage implementation (default auto: mock stack -> mock, http -> energy)",
    )
    run_p.add_argument(
        "--seed", type=int, default=0, help="seed for deterministic mock jitter (default 0)"
    )
    run_p.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        metavar="SEC",
        help="per-call safety timeout in seconds (default 300)",
    )
    run_p.add_argument("--out", metavar="FILE", help="write raw results JSON to FILE")
    run_p.add_argument("--html", metavar="FILE", help="write self-contained HTML report to FILE")
    run_p.add_argument(
        "--quiet", action="store_true", help="suppress the terminal summary table"
    )

    report_p = sub.add_parser("report", help="render an HTML report from results JSON")
    report_p.add_argument("results", metavar="RESULTS_JSON", help="file written by 'run --out'")
    report_p.add_argument(
        "--html", default="report.html", metavar="FILE", help="output path (default report.html)"
    )

    init_p = sub.add_parser("init", help="write an example scenario file to edit")
    init_p.add_argument(
        "--out", default="scenario.json", metavar="FILE", help="output path (default scenario.json)"
    )
    init_p.add_argument(
        "--force", action="store_true", help="overwrite FILE if it already exists"
    )
    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    if args.calls < 1:
        print("error: --calls must be >= 1", file=sys.stderr)
        return _USAGE_ERROR
    if args.ramp < 0:
        print("error: --ramp must be >= 0", file=sys.stderr)
        return _USAGE_ERROR
    if args.timeout <= 0:
        print("error: --timeout must be > 0", file=sys.stderr)
        return _USAGE_ERROR
    scenario = load_scenario(args.scenario) if args.scenario else default_scenario()
    backend_config = (
        load_backend_config(args.backend_config) if args.backend_config else None
    )
    clock = MonotonicClock()
    backends = build_backends(
        kind=args.backend,
        clock=clock,
        profile_name=args.profile,
        seed=args.seed,
        vad_kind=args.vad,
        backend_config=backend_config,
    )
    config = LoadTestConfig(calls=args.calls, ramp_s=args.ramp, call_timeout_s=args.timeout)
    if not args.quiet:
        print(
            f"voiceprobe {__version__} — scenario '{scenario.name}', "
            f"{config.calls} call(s), backend {args.backend}",
            file=sys.stderr,
        )
    result = asyncio.run(run_load_test(scenario, backends, config, clock))
    doc = result.to_dict()
    if args.out:
        Path(args.out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        if not args.quiet:
            print(f"results written to {args.out}", file=sys.stderr)
    if args.html:
        Path(args.html).write_text(render_report(doc), encoding="utf-8")
        if not args.quiet:
            print(f"HTML report written to {args.html}", file=sys.stderr)
    metrics = compute_metrics(result.calls)
    if not args.quiet:
        print(format_summary(metrics, result.duration_s, len(result.calls), result.failed_calls))
    if result.failed_calls:
        for call in result.calls:
            if not call.ok:
                print(
                    f"warning: call {call.call_id} failed: {call.error or 'turn error'}",
                    file=sys.stderr,
                )
    if result.ok_calls == 0:
        print("error: every simulated call failed", file=sys.stderr)
        return _RUNTIME_ERROR
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    path = Path(args.results)
    if not path.exists():
        print(f"error: results file not found: {path}", file=sys.stderr)
        return _USAGE_ERROR
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        print(f"error: {path} is not valid JSON: {exc}", file=sys.stderr)
        return _USAGE_ERROR
    if not isinstance(doc, dict) or "calls" not in doc:
        print(
            f"error: {path} does not look like voiceprobe results "
            "(expected an object with a 'calls' array)",
            file=sys.stderr,
        )
        return _USAGE_ERROR
    Path(args.html).write_text(render_report(doc), encoding="utf-8")
    print(f"HTML report written to {args.html}", file=sys.stderr)
    return 0


def _cmd_init(args: argparse.Namespace) -> int:
    path = Path(args.out)
    if path.exists() and not args.force:
        print(
            f"error: {path} already exists; pass --force to overwrite", file=sys.stderr
        )
        return _USAGE_ERROR
    path.write_text(example_scenario_json(), encoding="utf-8")
    print(f"example scenario written to {path} — edit the turns, then run:", file=sys.stderr)
    print(f"  voiceprobe run --scenario {path} --calls 5 --html report.html", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "report":
            return _cmd_report(args)
        if args.command == "init":
            return _cmd_init(args)
    except (ScenarioError, ConfigError, AudioError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _USAGE_ERROR
    except BackendError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _RUNTIME_ERROR
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _RUNTIME_ERROR
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    parser.error(f"unknown command {args.command!r}")
    return _USAGE_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
