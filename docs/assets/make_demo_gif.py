"""Regenerate docs/assets/demo.gif from a real voiceprobe run.

Development-only script (not part of the package). It executes an actual
load test with the CLI, captures the terminal output, and renders it as
GIF frames — every number in the demo comes from that live run.

Usage (Pillow required, e.g. ``pip install pillow``):

    python3 docs/assets/make_demo_gif.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ASSETS_DIR = Path(__file__).parent
PROJECT_ROOT = ASSETS_DIR.parent.parent
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
]

BG = (18, 21, 28)
FG = (214, 219, 230)
DIM = (122, 129, 148)
GREEN = (98, 200, 130)
STAGE_COLORS = {
    "vad": (110, 150, 240),
    "stt": (85, 190, 140),
    "llm": (235, 180, 80),
    "tts": (205, 125, 220),
}
ACCENT = (140, 170, 255)

CMD = "voiceprobe run --scenario clinic.json --calls 10 --ramp 2 --vad energy --html report.html"


def run_real_load_test(workdir: Path) -> tuple[str, str]:
    """Execute the CLI for real and return (stderr_banner, stdout_summary)."""
    scenario = {
        "name": "clinic-appointment",
        "turns": [
            {"user_text": "Good morning, I need to reschedule my dental cleaning."},
            {
                "user_text": "My patient number is seven three five nine.",
                "pause_ms": 700,
            },
            {"user_text": "Is there anything available next Tuesday afternoon?", "pause_ms": 500},
            {"user_text": "Perfect, please book the three thirty slot.", "pause_ms": 400},
        ],
    }
    (workdir / "clinic.json").write_text(json.dumps(scenario), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "voiceprobe.cli",
            "run",
            "--scenario",
            "clinic.json",
            "--calls",
            "10",
            "--ramp",
            "2",
            "--vad",
            "energy",
            "--seed",
            "42",
            "--html",
            "report.html",
        ],
        cwd=workdir,
        env={"PYTHONPATH": str(PROJECT_ROOT / "src"), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stderr.strip(), proc.stdout.rstrip()


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise SystemExit("no monospace TTF font found; install fonts-dejavu")


def line_color(line: str) -> tuple[int, int, int]:
    stripped = line.strip()
    for stage, color in STAGE_COLORS.items():
        if stripped.startswith(stage):
            return color
    if stripped.startswith(("first token", "first audio", "turn total")):
        return FG
    if stripped.startswith(("stage", "-")):
        return DIM
    if stripped.startswith(("calls:", "voiceprobe")):
        return DIM
    return FG


def render_frame(
    lines: list[tuple[str, tuple[int, int, int]]],
    font: ImageFont.FreeTypeFont,
    width: int,
    height: int,
    cursor: bool = False,
) -> Image.Image:
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)
    # Window chrome dots.
    for i, color in enumerate(((236, 106, 94), (245, 191, 79), (98, 197, 84))):
        draw.ellipse((14 + i * 22, 12, 26 + i * 22, 24), fill=color)
    draw.text((width - 210, 10), "voiceprobe demo", font=font, fill=DIM)
    y = 40
    line_h = 19
    for text, color in lines:
        draw.text((16, y), text, font=font, fill=color)
        y += line_h
    if cursor and lines:
        last_text = lines[-1][0]
        x = 16 + draw.textlength(last_text, font=font)
        draw.rectangle((x + 2, y - line_h + 2, x + 11, y - 3), fill=FG)
    return img


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        banner, summary = run_real_load_test(Path(tmp))
    summary_lines = summary.splitlines()
    font = load_font(14)
    n_rows = len(summary_lines) + 4
    width, height = 900, 40 + 19 * n_rows + 14

    frames: list[Image.Image] = []
    durations: list[int] = []

    prompt = "$ "
    # Typing animation for the command, in a few keystroke groups.
    for cut in (18, 40, len(CMD)):
        frames.append(
            render_frame([(prompt + CMD[:cut], FG)], font, width, height, cursor=True)
        )
        durations.append(420)

    base: list[tuple[str, tuple[int, int, int]]] = [(prompt + CMD, FG)]
    base.append((banner.splitlines()[0], DIM))
    frames.append(render_frame(base, font, width, height))
    durations.append(700)

    shown = list(base)
    for line in summary_lines:
        shown.append((line, line_color(line)))
        frames.append(render_frame(shown, font, width, height))
        durations.append(240)

    shown.append(("", FG))
    shown.append(("report.html written — waterfall + flame chart, self-contained", GREEN))
    final = render_frame(shown, font, width, height)
    frames.append(final)
    durations.append(900)
    frames.append(final)
    durations.append(3600)

    out = ASSETS_DIR / "demo.gif"
    # Quantize to a small palette to keep the file well under 1 MB.
    quantized = [f.quantize(colors=64, dither=Image.Dither.NONE) for f in frames]
    quantized[0].save(
        out,
        save_all=True,
        append_images=quantized[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"wrote {out} ({out.stat().st_size} bytes, {len(frames)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
