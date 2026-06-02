"""Animate the 2D embedding scatter for the DistAL video (§3 of script.md).

Visualises the key insight — "as a policy drifts toward failure, its observations
move away from the data the VLA was trained on" — as a 2D SigLIP feature space:

    1. A cloud of "training successes" (a tight green cluster).
    2. A deployment trajectory that starts inside the cloud and drifts outward.
    3. Live k-nearest-neighbour spokes from the moving frame back into the cloud,
       growing longer as it drifts; a side bar reads off the resulting dense,
       per-step reward (∝ kNN distance), and the frame reddens as it goes OOD.

Renders 1920x1080 on the Oxford-blue background used by the title card and the
other §1-§2 graphics, so it drops straight into the edit.

Usage:
    uv run manim -qh generate_key_insight.py KeyInsightScatter
    # add -p to preview
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Create,
    DecimalNumber,
    Dot,
    FadeIn,
    GrowFromCenter,
    LaggedStart,
    Line,
    ManimColor,
    MoveAlongPath,
    Rectangle,
    Scene,
    Text,
    TracedPath,
    VGroup,
    VMobject,
    Write,
    always_redraw,
    config,
    interpolate_color,
    linear,
)

# --- Visual identity (matches generate_title_card.py / panels) --------------

OXFORD_BLUE = ManimColor("#002147")
WHITE = ManimColor("#F5F7FA")
ACCENT = ManimColor("#78B2FF")  # light Oxford blue
DIVIDER = ManimColor("#6E82A0")
GOOD = ManimColor("#5CD08A")  # green — in-distribution / training successes
BAD = ManimColor("#FF5A5A")  # red — out-of-distribution / drifting to failure

config.background_color = OXFORD_BLUE
config.pixel_width = 1920
config.pixel_height = 1080
config.frame_rate = 25

# --- Geometry ---------------------------------------------------------------

CLOUD_CENTER = np.array([-2.9, 0.3, 0.0])
CLOUD_RADIUS = 1.6  # successes are sampled uniformly within this disk
N_TRAIN = 50
K = 5  # neighbours drawn as spokes / averaged for the reward

# The deployment frame meanders inside the training cluster first (staying
# in-distribution, reward ~0) before curving out toward the failure region.
# These waypoints are fed through a smooth spline so the motion wanders and
# loops rather than tracing a straight beeline.
TRAJ_START = np.array([-2.9, 0.3, 0.0])
TRAJ_END = np.array([2.8, -1.0, 0.0])
TRAJ_WAYPOINTS = np.array(
    [
        TRAJ_START,
        [-3.7, 1.1, 0.0],   # drift up-left, still well inside the cloud
        [-2.0, 1.0, 0.0],   # arc across the top
        [-2.4, -0.8, 0.0],  # loop back down through the centre
        [-3.5, -0.2, 0.0],  # tight loop back toward the left edge
        [-1.6, 0.6, 0.0],   # begin escaping to the right
        [-0.2, -0.5, 0.0],  # cross open space, dipping down
        [1.3, 0.5, 0.0],    # swing back up
        [2.1, -0.3, 0.0],   # final approach to the failure region
        TRAJ_END,
    ]
)

# kNN distance is mapped through this range into the reward: ~0 in-distribution,
# falling to -1 once the frame has fully drifted out.
DIST_MIN = 0.65  # typical in-cluster nearest-neighbour distance
DIST_MAX = 4.6  # roughly the distance once fully drifted out

# Reward bar, parked on the right.
BAR_X = 5.5
BAR_BOTTOM = -2.1
BAR_H = 4.0
BAR_W = 0.55


def make_training_cloud() -> tuple[np.ndarray, VGroup]:
    """Sample 'training successes' uniformly within a disk (points + dots)."""
    rng = np.random.default_rng(7)
    # Uniform-in-disk: r = R*sqrt(u) compensates for the area growing with r.
    theta = rng.uniform(0.0, 2.0 * np.pi, N_TRAIN)
    r = CLOUD_RADIUS * np.sqrt(rng.uniform(0.0, 1.0, N_TRAIN))
    offsets = np.stack([r * np.cos(theta), r * np.sin(theta), np.zeros(N_TRAIN)], 1)
    pts = CLOUD_CENTER + offsets
    pts[:, 2] = 0.0
    dots = VGroup(
        *[
            Dot(p, radius=0.055, color=GOOD).set_fill(GOOD, opacity=0.85)
            for p in pts
        ]
    )
    return pts, dots


def reward_from_point(p: np.ndarray, train: np.ndarray) -> float:
    """Dense reward at ``p``: ~0 inside the distribution, -1 once fully OOD.

    The mean distance to the K nearest training points is normalised to [0, 1]
    and negated, so the reward starts high (near 0) and falls to -1 as the
    frame drifts away from the training successes.
    """
    d = np.sort(np.linalg.norm(train - p, axis=1))[:K]
    norm = (d.mean() - DIST_MIN) / (DIST_MAX - DIST_MIN)
    return -float(np.clip(norm, 0.0, 1.0))


OUTPUT_PATH = Path(__file__).parent / "outputs" / "knn.mp4"


class KeyInsightScatter(Scene):
    def render(self, *args, **kwargs) -> None:
        """Render as usual, then copy the finished movie into outputs/."""
        super().render(*args, **kwargs)
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.renderer.file_writer.movie_file_path, OUTPUT_PATH)
        print(f"✓ copied render -> {OUTPUT_PATH}")

    def construct(self) -> None:
        train, cloud = make_training_cloud()

        # --- Feature-space frame + caption. ---------------------------------
        space_label = Text(
            "SigLIP feature space", font="Noto Sans", weight="MEDIUM",
            color=DIVIDER, font_size=28,
        ).to_edge(UP, buff=0.5)

        cloud_label = Text(
            "Training successes", font="Noto Sans", weight="MEDIUM",
            color=GOOD, font_size=28,
        ).move_to(CLOUD_CENTER + DOWN * 2.25)

        # --- Reward bar (right). --------------------------------------------
        bar_track = Rectangle(
            width=BAR_W, height=BAR_H, color=DIVIDER, stroke_width=2,
        ).move_to([BAR_X, BAR_BOTTOM + BAR_H / 2, 0])
        bar_caption = Text(
            "dense reward rₜ", font="Noto Sans", weight="MEDIUM",
            color=WHITE, font_size=26,
        ).next_to(bar_track, DOWN, buff=0.22)
        bar_sub = Text(
            "−kNN distance", font="Noto Sans", weight="MEDIUM",
            color=ACCENT, font_size=22,
        ).next_to(bar_caption, DOWN, buff=0.12)
        # 0 at the top (in-distribution), -1 at the bottom (fully OOD).
        tick_0 = Text("0", font="Noto Sans", color=DIVIDER, font_size=22)
        tick_0.next_to(bar_track.get_corner(UP + LEFT), LEFT, buff=0.18)
        tick_neg = Text("−1", font="Noto Sans", color=DIVIDER, font_size=22)
        tick_neg.next_to(bar_track.get_corner(DOWN + LEFT), LEFT, buff=0.18)

        # --- The drifting deployment frame. ---------------------------------
        # Smooth spline through the waypoints — a wandering, looping path
        # rather than a straight drift.
        path = VMobject().set_points_smoothly(TRAJ_WAYPOINTS)
        agent = Dot(TRAJ_START, radius=0.14, color=GOOD)
        agent.set_fill(GOOD, opacity=1.0).set_stroke(WHITE, width=2)
        trail = TracedPath(
            agent.get_center, stroke_color=ACCENT, stroke_width=4,
        )

        # Live kNN spokes from the agent back into the cloud.
        def make_spokes() -> VGroup:
            p = agent.get_center()
            idx = np.argsort(np.linalg.norm(train - p, axis=1))[:K]
            return VGroup(
                *[
                    Line(p, train[i], color=ACCENT, stroke_width=2.5)
                    .set_stroke(opacity=0.55)
                    for i in idx
                ]
            )

        spokes = always_redraw(make_spokes)

        # The agent reddens as it leaves the distribution (reward -> -1).
        def recolor(mob: Dot) -> None:
            frac = -reward_from_point(mob.get_center(), train)
            mob.set_fill(interpolate_color(GOOD, BAD, frac), opacity=1.0)

        # Reward bar hangs down from the zero line at the top toward -1.
        def make_bar() -> Rectangle:
            frac = -reward_from_point(agent.get_center(), train)
            h = max(1e-3, frac * BAR_H)
            top = BAR_BOTTOM + BAR_H
            return (
                Rectangle(width=BAR_W, height=h, stroke_width=0)
                .set_fill(interpolate_color(GOOD, BAD, frac), opacity=0.9)
                .move_to([BAR_X, top - h / 2, 0])
            )

        bar_fill = always_redraw(make_bar)
        readout = DecimalNumber(
            0.0, num_decimal_places=2, color=WHITE, font_size=30,
        ).next_to(bar_track, UP, buff=0.2)
        readout.add_updater(
            lambda m: m.set_value(reward_from_point(agent.get_center(), train))
        )

        # --- Timeline -------------------------------------------------------
        # 1) Feature space + the training cluster.
        self.play(Write(space_label), run_time=1.2)
        self.play(Write(cloud_label), run_time=1.2)
        self.play(
            LaggedStart(
                *[GrowFromCenter(d) for d in cloud],
                lag_ratio=0.01, run_time=2.8,
            )
        )
        self.wait(0.6)

        # 2) The reward gauge.
        self.play(
            Create(bar_track), Write(bar_caption), Write(bar_sub),
            FadeIn(tick_0), FadeIn(tick_neg), FadeIn(readout), run_time=1.4,
        )

        # 3) Drop the agent into the cloud and let the loop run.
        self.play(GrowFromCenter(agent), run_time=1.0)
        agent.add_updater(recolor)
        self.add(trail, spokes, bar_fill)
        self.wait(0.8)

        # 4) Drift outward — spokes stretch, bar fills, frame reddens.
        self.play(MoveAlongPath(agent, path), run_time=13.0, rate_func=linear)

        # 5) Land the punchline at the OOD end of the trajectory.
        agent.remove_updater(recolor)
        readout.clear_updaters()
        ood_label = Text(
            "Out of distribution\n= low reward", font="Noto Sans",
            weight="MEDIUM", color=BAD, font_size=28, line_spacing=0.6,
        ).next_to(agent, UP, buff=1.8)
        self.play(Write(ood_label), run_time=1.4)
        self.play(agent.animate.scale(1.25), run_time=0.6)
        self.play(agent.animate.scale(1 / 1.25), run_time=0.6)
        self.wait(3.0)


if __name__ == "__main__":
    raise SystemExit(
        "Render with: uv run manim -qh generate_key_insight.py KeyInsightScatter"
    )
