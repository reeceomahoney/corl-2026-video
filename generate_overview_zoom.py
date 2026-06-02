"""Animate a guided tour of the method overview figure for the DistAL video.

Takes ``overview.png`` (the three-step "Score → Compute advantage → Train VLA"
schematic) and:

    1. Fades in the whole figure so the viewer sees all three steps at once.
    2. Zooms into each step in turn, dimming the other two so the focused panel
       reads as a spotlight.
    3. Pulls back out to the full figure to close.

The motion is synced to the §4 (Method) narration clip
``outputs/voice_clips/004_distal_is_a_drop_in_replacement_for_the_reward_in.mp3``:
each zoom-in fires as the voiceover says "Step one / two / three", and the
camera pulls back out on "The encoder is the ...". The cue times below were
measured from that clip with ``ffmpeg ... silencedetect`` (each cue is the start
of the named sentence). The render itself is silent, like the other graphics —
the editor lays the voiceover track over it — and ``CLIP_END`` makes the clip's
length match the narration so it drops in aligned.

The figure is very wide (~2.89:1) while each panel is near-square, so a fixed
16:9 frame can't fill on both the wide overview and a single panel. We sidestep
that by loading the three panels as separate images at their true layout
positions on the Oxford-blue background: the overview shows all three, and a
zoom moves the camera onto one panel while the siblings fade back.

Renders 1280x720 on the Oxford-blue background shared by the title card and the
other graphics, so it drops straight into the edit.

Usage:
    uv run manim -qh generate_overview_zoom.py OverviewZoom
    # add -p to preview
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from manim import (
    ImageMobject,
    ManimColor,
    MovingCameraScene,
    config,
    smooth,
)

# --- Visual identity (matches generate_title_card.py / the other graphics) ---

OXFORD_BLUE = ManimColor("#002147")

config.background_color = OXFORD_BLUE
config.pixel_width = 1280
config.pixel_height = 720
config.frame_rate = 25

# --- Source figure ----------------------------------------------------------

SOURCE = Path(__file__).with_name("overview.png")
CACHE = Path(__file__).with_name("media") / "overview_zoom"
AUDIO = (
    Path(__file__).with_name("outputs")
    / "voice_clips"
    / "004_distal_is_a_drop_in_replacement_for_the_reward_in.mp3"
)

# How tightly a zoomed panel fills the frame height, how tightly the full figure
# fills the frame width (< 1 leaves a border around it), and how far the siblings
# fade back while one panel is in focus.
PANEL_FILL = 0.92
OVERVIEW_FILL = 0.9
DIM_OPACITY = 0.12

# --- Narration cues (seconds into the §4 clip) ------------------------------
# Sentence starts measured from AUDIO via `ffmpeg -af silencedetect`. The three
# zoom-ins fire on "Step one / two / three"; the pull-back fires on "The encoder
# is the ...". CLIP_END is the clip's full duration, so the animation length
# matches the narration exactly.
STEP_CUES = [6.34, 10.58, 13.90]  # "Step one" / "Step two" / "Step three"
ZOOM_OUT_CUE = 23.61  # "The encoder is the ..."
CLIP_END = 35.04

REVEAL = 1.4  # fade-in of the full figure (starts at t=0)
MOVE = 1.1  # seconds for each zoom-in camera move
ZOOM_OUT = 1.4  # seconds for the closing pull-back


def detect_panels(png: Path) -> list[tuple[int, int]]:
    """Find the x-pixel extents of each panel from the white gaps between them."""
    arr = np.asarray(Image.open(png).convert("RGB")).astype(int)
    w = arr.shape[1]
    white = (arr > 240).all(axis=2)
    gap = white.mean(axis=0) > 0.97  # near-fully-white columns separate panels
    xs = np.where(~gap)[0]
    segments: list[tuple[int, int]] = []
    start = prev = xs[0]
    for x in xs[1:]:
        if x - prev > 5:
            segments.append((start, prev))
            start = x
        prev = x
    segments.append((start, prev))
    # Keep the real panels (drop any stray specks narrower than 5% of the width).
    return [(a, b) for a, b in segments if b - a > w * 0.05]


def crop_panels(png: Path, bounds: list[tuple[int, int]]) -> list[Path]:
    """Save each full-height panel slice as its own PNG."""
    img = Image.open(png).convert("RGBA")
    h = img.height
    paths = []
    for i, (a, b) in enumerate(bounds):
        out = CACHE / f"panel_{i}.png"
        img.crop((a, 0, b, h)).save(out)
        paths.append(out)
    return paths


class OverviewZoom(MovingCameraScene):
    def construct(self) -> None:
        CACHE.mkdir(parents=True, exist_ok=True)
        png = SOURCE
        bounds = detect_panels(png)
        panel_paths = crop_panels(png, bounds)

        full_w = Image.open(png).width
        fw, fh = config.frame_width, config.frame_height

        # Map the full figure to fit the frame width; lay the cropped panels at
        # their true positions so together they reproduce the original figure.
        units_per_px = fw / full_w
        panels = []
        centres = []
        for (a, b), path in zip(bounds, panel_paths):
            panel = ImageMobject(str(path))
            panel.width = (b - a) * units_per_px
            cx = ((a + b) / 2 - full_w / 2) * units_per_px
            panel.move_to([cx, 0.0, 0.0])
            panels.append(panel)
            centres.append(cx)

        # The cue timeline assumes one zoom per step, so the figure must split
        # into exactly as many panels as there are step cues.
        if len(panels) != len(STEP_CUES):
            raise RuntimeError(
                f"detected {len(panels)} panels but have {len(STEP_CUES)} step "
                "cues; re-check overview.png / STEP_CUES."
            )

        camera = self.camera.frame
        # Frame the full figure with a margin (it's mapped to the frame width, so
        # pull the camera back a touch to keep it off the edges).
        overview_h = fh / OVERVIEW_FILL
        camera.set(height=overview_h)

        # --- 1) Reveal the whole figure, then hold until "Step one". --------
        for panel in panels:
            self.add(panel)
            panel.set_opacity(0.0)
        self.play(*[p.animate.set_opacity(1.0) for p in panels], run_time=REVEAL)
        self.wait(STEP_CUES[0] - REVEAL)

        # --- 2) Spotlight each step, the zoom firing on its cue. ------------
        # A zoomed panel fills (most of) the frame height.
        zoom_h = panels[0].height / PANEL_FILL
        # When to leave each panel: the next step's cue, then the pull-back cue.
        leave_at = STEP_CUES[1:] + [ZOOM_OUT_CUE]

        for i, panel in enumerate(panels):
            self.play(
                camera.animate.move_to([centres[i], 0.0, 0.0]).set(height=zoom_h),
                panel.animate.set_opacity(1.0),
                *[
                    p.animate.set_opacity(DIM_OPACITY)
                    for j, p in enumerate(panels)
                    if j != i
                ],
                run_time=MOVE,
                rate_func=smooth,
            )
            self.wait(leave_at[i] - STEP_CUES[i] - MOVE)

        # --- 3) Pull back out to the full figure on "The encoder is the". ---
        self.play(
            camera.animate.move_to([0.0, 0.0, 0.0]).set(height=overview_h),
            *[p.animate.set_opacity(1.0) for p in panels],
            run_time=ZOOM_OUT,
            rate_func=smooth,
        )
        self.wait(CLIP_END - ZOOM_OUT_CUE - ZOOM_OUT)


if __name__ == "__main__":
    raise SystemExit(
        "Render with: uv run manim -qh generate_overview_zoom.py OverviewZoom"
    )
