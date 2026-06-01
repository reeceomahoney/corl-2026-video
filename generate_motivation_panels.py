"""Animate the motivation panels for the DistAL video (§2 of script.md).

Three panels reveal in sequence, synced to the voiceover beat:

    1. Teleoperator        -> "interventions: expensive"   (red ✗)
    2. Reward curve        -> "online RL: sample inefficient" (red ✗)
    3. Advantage conditioning -> green ✓ (the alternative without the drawbacks)

Renders a 1920x1080 clip on the Oxford-blue background used by the title card,
so it drops straight into the edit.

Usage:
    uv run manim -qh generate_motivation_panels.py MotivationPanels
    # add -p to preview, or --format=mov --transparent for an alpha overlay
"""

from __future__ import annotations

import numpy as np
from manim import (
    DOWN,
    UP,
    Circle,
    Create,
    Cross,
    Ellipse,
    FadeIn,
    GrowFromCenter,
    Line,
    ManimColor,
    RoundedRectangle,
    Scene,
    Text,
    VGroup,
    VMobject,
    Write,
    config,
)

# --- Visual identity (matches generate_title_card.py) -----------------------

OXFORD_BLUE = ManimColor("#002147")
WHITE = ManimColor("#F5F7FA")
ACCENT = ManimColor("#78B2FF")  # light Oxford blue
DIVIDER = ManimColor("#6E82A0")
BAD = ManimColor("#FF5A5A")  # red — the drawback panels
GOOD = ManimColor("#5CD08A")  # green — advantage conditioning

config.background_color = OXFORD_BLUE
config.pixel_width = 1280
config.pixel_height = 720
config.frame_rate = 25

PANEL_W = 3.9
PANEL_H = 4.4
PANEL_GAP = 0.55
ICON_BOX = 2.0  # square region icons are drawn into, in scene units


# --- Icon builders ----------------------------------------------------------
#
# Each returns a VMobject sized to roughly fit inside an ICON_BOX square,
# centred on the origin. The panel layout repositions them.


def teleoperator_icon() -> VGroup:
    """A joystick: elliptical base, an angled shaft, and a ball grip on top."""
    base = Ellipse(width=1.4, height=0.42, color=WHITE, stroke_width=6)
    base.set_fill(WHITE, opacity=0.12).move_to([0, -0.62, 0])
    # Shaft leans slightly right for a bit of life.
    shaft_top = np.array([0.22, 0.34, 0])
    shaft = Line([0, -0.58, 0], shaft_top, color=WHITE, stroke_width=12)
    knob = Circle(radius=0.32, color=ACCENT, stroke_width=6)
    knob.set_fill(ACCENT, opacity=1.0).move_to(shaft_top + np.array([0, 0.18, 0]))
    return VGroup(base, shaft, knob)


def reward_curve_icon() -> VGroup:
    """Axes with a noisy, slowly-rising reward curve — the RL sample cost."""
    x0, y0 = -0.9, -0.8
    axes = VGroup(
        Line([x0, y0, 0], [x0 + 1.9, y0, 0], color=DIVIDER, stroke_width=4),  # x
        Line([x0, y0, 0], [x0, y0 + 1.7, 0], color=DIVIDER, stroke_width=4),  # y
    )
    # Noisy rising curve: a smooth log-ish trend plus jitter.
    n = 90
    xs = np.linspace(0.0, 1.85, n)
    trend = 1.45 * (1 - np.exp(-2.4 * xs / 1.85))
    rng = np.sin(xs * 23.0) * 0.10 + np.sin(xs * 41.0) * 0.05
    ys = np.clip(trend + rng, 0, None)
    points = [np.array([x0 + x, y0 + y, 0]) for x, y in zip(xs, ys)]
    curve = VMobject(color=ACCENT, stroke_width=6).set_points_smoothly(points)
    return VGroup(axes, curve)


def advantage_icon() -> VGroup:
    """A bold green tick to mark advantage conditioning as the good path."""
    check = VMobject(color=GOOD, stroke_width=12)
    check.set_points_as_corners(
        [
            np.array([-0.55, 0.05, 0]),
            np.array([-0.12, -0.45, 0]),
            np.array([0.62, 0.6, 0]),
        ]
    )
    ring = Circle(radius=0.95, color=GOOD, stroke_width=6)
    return VGroup(ring, check)


# --- Panel assembly ----------------------------------------------------------


def make_panel(icon: VGroup, title: str, subtitle: str, *, good: bool) -> VGroup:
    accent = GOOD if good else BAD
    frame = RoundedRectangle(
        width=PANEL_W,
        height=PANEL_H,
        corner_radius=0.22,
        color=DIVIDER,
        stroke_width=3,
    )

    icon.move_to(frame.get_center() + UP * 0.95)

    title_t = Text(title, font="Noto Sans", weight="BOLD", color=WHITE, font_size=34)
    sub_t = Text(subtitle, font="Noto Sans", weight="MEDIUM", color=accent, font_size=26)

    # Keep text inside the panel: scale down anything wider than the inner width.
    inner_w = PANEL_W * 0.84
    for t in (title_t, sub_t):
        if t.width > inner_w:
            t.scale_to_fit_width(inner_w)

    title_t.move_to(frame.get_center() + DOWN * 0.85)
    sub_t.next_to(title_t, DOWN, buff=0.22)

    return VGroup(frame, icon, title_t, sub_t)


def stamp(panel: VGroup, *, good: bool) -> VMobject:
    """Build the ✗ / ✓ stamp over a panel's icon (returned, not added)."""
    icon = panel[1]
    if good:
        return panel  # the advantage panel already carries its own tick
    cross = Cross(
        stroke_color=BAD, stroke_width=14
    ).scale(0.5).move_to(icon.get_center())
    return cross


# --- Scene -------------------------------------------------------------------


class MotivationPanels(Scene):
    def construct(self) -> None:
        panels = VGroup(
            make_panel(
                teleoperator_icon(),
                "Interventions",
                "expensive",
                good=False,
            ),
            make_panel(
                reward_curve_icon(),
                "Online RL",
                "sample inefficient",
                good=False,
            ),
            make_panel(
                advantage_icon(),
                "Advantage Conditioning",
                "no teleop · no online RL",
                good=True,
            ),
        )
        panels.arrange(buff=PANEL_GAP).move_to([0, 0, 0])

        # Reveal each panel in turn, synced to the VO beat.
        # 1) Interventions: expensive.
        p = panels[0]
        self.play(FadeIn(p[0], shift=UP * 0.2), run_time=0.5)
        self.play(GrowFromCenter(p[1]), run_time=0.6)
        self.play(Write(p[2]), Write(p[3]), run_time=0.7)
        self.play(Create(stamp(p, good=False)), run_time=0.5)
        self.wait(0.4)

        # 2) Online RL: sample inefficient.
        p = panels[1]
        self.play(FadeIn(p[0], shift=UP * 0.2), run_time=0.5)
        self.play(Create(p[1]), run_time=0.9)
        self.play(Write(p[2]), Write(p[3]), run_time=0.7)
        self.play(Create(stamp(p, good=False)), run_time=0.5)
        self.wait(0.4)

        # 3) Advantage conditioning — the green tick.
        p = panels[2]
        self.play(FadeIn(p[0], shift=UP * 0.2), run_time=0.5)
        self.play(Create(p[1]), run_time=0.8)
        self.play(Write(p[2]), Write(p[3]), run_time=0.7)
        # Let the winning panel pop.
        self.play(p.animate.scale(1.06), run_time=0.4)
        self.play(p.animate.scale(1 / 1.06), run_time=0.3)
        self.wait(1.0)


if __name__ == "__main__":
    raise SystemExit(
        "Render with: uv run manim -qh generate_motivation_panels.py MotivationPanels"
    )
