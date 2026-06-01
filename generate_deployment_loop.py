"""Animate the deployment-data loop graphic for the DistAL video (§2 of script.md).

The opening motivation beat — "enabling VLAs to learn from their own deployment
data ... a promising path to improving robustness" — is visualised as a closed
flywheel:

    [ rollout video ]  --collect-->  [ deployment data ]  --improve-->  back

A real base-policy ethernet-unplug rollout plays on the left. Frame thumbnails fly
along the top arc into a small stack ("deployment data"); a return arc loops
that data back into the policy that produced the rollout. Pulsing dots travel
both arcs continuously so the loop reads as self-improving.

Renders 1920x1080 on the Oxford-blue background used by the title card and
motivation panels, so it drops straight into the edit.

Usage:
    uv run manim -qh generate_deployment_loop.py DeploymentLoop
    # add -p to preview
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    ArcBetweenPoints,
    Create,
    CurvedArrow,
    Dot,
    FadeIn,
    Group,
    GrowFromCenter,
    ImageMobject,
    ManimColor,
    RoundedRectangle,
    Scene,
    Text,
    Write,
    config,
)

# --- Visual identity (matches generate_title_card.py / panels) --------------

OXFORD_BLUE = ManimColor("#002147")
WHITE = ManimColor("#F5F7FA")
ACCENT = ManimColor("#78B2FF")  # light Oxford blue
DIVIDER = ManimColor("#6E82A0")

config.background_color = OXFORD_BLUE
config.pixel_width = 1280
config.pixel_height = 720
config.frame_rate = 25
config.disable_caching = True

# --- Source clip ------------------------------------------------------------

CLIP = Path(__file__).parent / "outputs" / "remove-ethernet" / "base" / "top.mp4"

# Decode parameters: a short, looping, downscaled window of the rollout.
CLIP_START = 6.0  # seconds into the base rollout (skip the initial reach)
CLIP_DURATION = 10.0
DECODE_FPS = 24
DECODE_HEIGHT = 360  # pixels; width follows the source 4:3 aspect ratio


def load_video_rgba(
    path: Path,
    *,
    height: int = DECODE_HEIGHT,
    fps: int = DECODE_FPS,
    start: float = CLIP_START,
    duration: float = CLIP_DURATION,
) -> np.ndarray:
    """Decode ``path`` into an (N, H, W, 4) uint8 array via an ffmpeg rawvideo pipe.

    Manim has no native video support; we feed the frames into an ImageMobject
    that swaps its ``pixel_array`` each scene frame (see VideoMobject).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Source clip not found: {path}\n"
            "Run `uv run python save_episodes.py` first to export the rollouts."
        )
    width = int(round(height * 4 / 3))
    width -= width % 2  # keep even for the scaler
    cmd = [
        "ffmpeg", "-v", "error",
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(path),
        "-vf", f"scale={width}:{height},fps={fps}",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    rgb = np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)
    alpha = np.full((*rgb.shape[:3], 1), 255, np.uint8)
    return np.concatenate([rgb, alpha], axis=-1).copy()


# --- A minimal looping video mobject ----------------------------------------


class VideoMobject(ImageMobject):
    """An ImageMobject that cycles through pre-decoded RGBA frames, looping."""

    def __init__(self, frames: np.ndarray, fps: int = DECODE_FPS, **kwargs) -> None:
        self.frames = frames
        self.fps = fps
        self._elapsed = 0.0
        super().__init__(frames[0], **kwargs)
        self.add_updater(VideoMobject._advance)

    @staticmethod
    def _advance(mob: "VideoMobject", dt: float) -> None:
        mob._elapsed += dt
        idx = int(mob._elapsed * mob.fps) % len(mob.frames)
        mob.pixel_array = np.ascontiguousarray(mob.frames[idx])


# --- Dataset icon ------------------------------------------------------------


def make_dataset(frames: np.ndarray, *, height: float) -> Group:
    """A fanned stack of frame thumbnails — the growing 'deployment data' set."""
    # Sample a few spread-out frames as static thumbnails.
    picks = np.linspace(0, len(frames) - 1, 4).astype(int)
    offsets = [(-0.18, -0.18), (-0.06, -0.06), (0.06, 0.06), (0.18, 0.18)]
    rotations = [6, 2, -2, -6]

    stack = Group()
    for pick, (dx, dy), rot in zip(picks, offsets, rotations):
        thumb = ImageMobject(frames[pick])
        thumb.height = height
        border = RoundedRectangle(
            width=thumb.width + 0.08,
            height=thumb.height + 0.08,
            corner_radius=0.06,
            color=WHITE,
            stroke_width=3,
        ).set_fill(OXFORD_BLUE, opacity=1.0)
        card = Group(border, thumb)
        card.rotate(rot * np.pi / 180).shift(np.array([dx, dy, 0]))
        stack.add(card)
    return stack


# --- Scene -------------------------------------------------------------------


class DeploymentLoop(Scene):
    def construct(self) -> None:
        frames = load_video_rgba(CLIP)

        # --- Left: the live rollout, in a rounded frame. ---------------------
        video = VideoMobject(frames)
        video.height = 3.7
        video.move_to([-3.5, 0.25, 0])
        video_frame = RoundedRectangle(
            width=video.width + 0.16,
            height=video.height + 0.16,
            corner_radius=0.12,
            color=WHITE,
            stroke_width=4,
        ).move_to(video.get_center())
        video_label = Text(
            "Deployment rollout", font="Noto Sans", weight="MEDIUM",
            color=WHITE, font_size=30,
        ).next_to(video_frame, DOWN, buff=0.3)

        # --- Right: the deployment-data stack. -------------------------------
        dataset = make_dataset(frames, height=1.7)
        dataset.move_to([3.7, 0.25, 0])
        data_frame = RoundedRectangle(
            width=3.4, height=3.4, corner_radius=0.14,
            color=DIVIDER, stroke_width=3,
        ).move_to(dataset.get_center())
        data_label = Text(
            "Deployment data", font="Noto Sans", weight="MEDIUM",
            color=ACCENT, font_size=30,
        ).next_to(data_frame, DOWN, buff=0.3)

        # --- Loop arcs between the two nodes (clockwise). --------------------
        top_arrow = CurvedArrow(
            video_frame.get_corner(UP + RIGHT) + RIGHT * 0.1,
            data_frame.get_corner(UP + LEFT) + LEFT * 0.1,
            angle=-0.9, color=ACCENT, stroke_width=6, tip_length=0.28,
        )
        bot_arrow = CurvedArrow(
            data_frame.get_corner(DOWN + LEFT) + LEFT * 0.1,
            video_frame.get_corner(DOWN + RIGHT) + RIGHT * 0.1,
            angle=-0.9, color=ACCENT, stroke_width=6, tip_length=0.28,
        )
        # Invisible guide paths for the travelling flow dots (no tip to skew
        # point_from_proportion near the ends).
        top_path = ArcBetweenPoints(
            video_frame.get_corner(UP + RIGHT) + RIGHT * 0.1,
            data_frame.get_corner(UP + LEFT) + LEFT * 0.1,
            angle=-0.9,
        )
        bot_path = ArcBetweenPoints(
            data_frame.get_corner(DOWN + LEFT) + LEFT * 0.1,
            video_frame.get_corner(DOWN + RIGHT) + RIGHT * 0.1,
            angle=-0.9,
        )
        collect_label = Text(
            "collect", font="Noto Sans", weight="MEDIUM", color=WHITE, font_size=26,
        ).move_to(top_path.point_from_proportion(0.5) + UP * 0.32)
        improve_label = Text(
            "improve", font="Noto Sans", weight="MEDIUM", color=WHITE, font_size=26,
        ).move_to(bot_path.point_from_proportion(0.5) + DOWN * 0.32)

        # --- Timeline --------------------------------------------------------
        # 1) Rollout comes up and starts playing.
        self.add(video)
        self.play(FadeIn(video_frame), run_time=0.7)
        self.play(Write(video_label), run_time=0.5)
        self.wait(0.4)

        # 2) Collect: top arc draws and the dataset appears in place.
        self.play(Create(top_arrow), FadeIn(collect_label), run_time=0.8)
        self.play(
            FadeIn(data_frame), GrowFromCenter(dataset), Write(data_label),
            run_time=0.8,
        )
        self.wait(0.3)

        # 3) Improve: return arc closes the loop.
        self.play(Create(bot_arrow), FadeIn(improve_label), run_time=0.8)

        # 4) Continuous flow dots while the loop settles.
        for dot in self._make_flow_dots(top_path) + self._make_flow_dots(bot_path):
            self.add(dot)
        self.wait(4.0)

    @staticmethod
    def _make_flow_dots(path, *, speed: float = 0.32, count: int = 2) -> list[Dot]:
        """Dots that travel ``path`` on a loop, evenly phased, to show flow."""
        dots: list[Dot] = []
        for i in range(count):
            dot = Dot(color=ACCENT, radius=0.09)
            dot.set_fill(WHITE, opacity=1.0)
            state = {"t": i / count}

            def updater(mob, dt, _path=path, _state=state):
                _state["t"] = (_state["t"] + dt * speed) % 1.0
                mob.move_to(_path.point_from_proportion(_state["t"]))

            dot.add_updater(updater)
            dots.append(dot)
        return dots


if __name__ == "__main__":
    raise SystemExit(
        "Render with: uv run manim -qh generate_deployment_loop.py DeploymentLoop"
    )
