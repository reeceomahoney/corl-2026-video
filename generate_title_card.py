"""Render the intro title card for the DistAL video (§1 of script.md).

Produces a 1920x1080 PNG on an Oxford-blue background with the project title,
authors, and institute, laid out as a centred title block.

Usage:
    uv run python generate_title_card.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# --- Content ----------------------------------------------------------------

TITLE = "DistAL: Distance-based Advantage Learning for VLA Fine-tuning"
AUTHORS = "Reece O'Mahoney · Ioannis Havoutis"
INSTITUTE = "Oxford Robotics Institute"

# --- Configuration ----------------------------------------------------------

OUTPUT_PATH = Path(__file__).parent / "outputs" / "title_card.png"

WIDTH, HEIGHT = 1920, 1080
SCALE = 2  # supersample then downscale for crisp, anti-aliased text

# Palette (RGBA).
WHITE = (245, 247, 250, 255)
OXFORD_BLUE = (0, 33, 71, 255)
ACCENT = (120, 178, 255, 255)  # light Oxford blue
DIVIDER = (110, 130, 160, 255)

# Font weights to search for, in order of preference.
FONT_CANDIDATES = {
    "bold": ["NotoSans-Bold.ttf", "LiberationSans-Bold.ttf"],
    "semibold": ["NotoSans-SemiBold.ttf", "NotoSans-Bold.ttf", "LiberationSans-Bold.ttf"],
    "medium": ["NotoSans-Medium.ttf", "NotoSans-Regular.ttf", "LiberationSans-Regular.ttf"],
}
FONT_DIRS = [Path("/usr/share/fonts/noto"), Path("/usr/share/fonts/liberation")]


# --- Fonts ------------------------------------------------------------------

def find_font_file(weight: str) -> Path:
    for name in FONT_CANDIDATES[weight]:
        for directory in FONT_DIRS:
            candidate = directory / name
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"No font found for weight '{weight}'. Looked in {[str(d) for d in FONT_DIRS]}."
    )


def font(weight: str, px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(find_font_file(weight)), px * SCALE)


# --- Layout helpers ---------------------------------------------------------

@dataclass
class Line:
    text: str
    font: ImageFont.FreeTypeFont
    color: tuple[int, int, int, int]
    tracking: int = 0  # extra px between glyphs (scaled units)
    space_before: int = 0  # extra gap above this line (scaled units)


def line_height(line: Line) -> int:
    ascent, descent = line.font.getmetrics()
    return ascent + descent


def draw_centered(draw: ImageDraw.ImageDraw, line: Line, cx: int, top: int) -> None:
    if not line.tracking:
        draw.text((cx, top), line.text, font=line.font, fill=line.color, anchor="ma")
        return
    total = int(draw.textlength(line.text, font=line.font))
    total += line.tracking * max(len(line.text) - 1, 0)
    x = cx - total // 2
    for char in line.text:
        draw.text((x, top), char, font=line.font, fill=line.color, anchor="la")
        x += int(draw.textlength(char, font=line.font)) + line.tracking


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=fnt) <= max_w or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


# --- Render -----------------------------------------------------------------

def build_lines(draw: ImageDraw.ImageDraw, max_text_w: int) -> list[Line]:
    title_font = font("bold", 78)
    lines: list[Line] = []
    for i, chunk in enumerate(wrap(draw, TITLE, title_font, max_text_w)):
        lines.append(Line(chunk, title_font, WHITE, space_before=0 if i == 0 else 16))
    lines.append(Line(AUTHORS, font("medium", 32), WHITE, space_before=90))
    lines.append(Line(INSTITUTE, font("semibold", 28), ACCENT, tracking=3, space_before=20))
    return lines


def render() -> Image.Image:
    w, h = WIDTH * SCALE, HEIGHT * SCALE
    img = Image.new("RGBA", (w, h), OXFORD_BLUE)
    draw = ImageDraw.Draw(img)
    cx = w // 2
    max_text_w = int(w * 0.82)
    lines = build_lines(draw, max_text_w)

    # Total block height to centre the group vertically.
    block_h = sum(line_height(line) + line.space_before for line in lines)
    y = (h - block_h) // 2

    for line in lines:
        y += line.space_before
        # Thin divider rule above the authors line.
        if line.text == AUTHORS:
            rule_w = 90 * SCALE
            ry = y - line.space_before // 2
            draw.line([(cx - rule_w, ry), (cx + rule_w, ry)], fill=DIVIDER, width=max(1, SCALE))
        draw_centered(draw, line, cx, y)
        y += line_height(line)

    return img.resize((WIDTH, HEIGHT), Image.LANCZOS)


# --- Main -------------------------------------------------------------------

def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    img = render()
    img.save(OUTPUT_PATH)
    print(f"✓ title card -> {OUTPUT_PATH} ({img.width}x{img.height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
