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
    MathTex,
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
    """The advantage definition, split over two lines, in green."""
    line1 = MathTex(r"A(s,a) =", color=GOOD)
    line2 = MathTex(r"\sum_t r_t - V(s)", color=GOOD)
    eq = VGroup(line1, line2).arrange(DOWN, buff=0.22)
    eq.scale_to_fit_width(ICON_BOX * 1.25)
    return eq


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

    icon.move_to(frame.get_center() + UP * (0.5 if good else 0.95))

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
    """Build the ✗ / ✓ corner badge for a panel (returned, not added)."""
    color = GOOD if good else BAD
    ring = Circle(radius=0.92, color=color, stroke_width=6)
    ring.set_fill(OXFORD_BLUE, opacity=1.0)
    if good:
        mark = VMobject(color=color, stroke_width=12)
        mark.set_points_as_corners(
            [
                np.array([-0.5, 0.0, 0]),
                np.array([-0.12, -0.42, 0]),
                np.array([0.58, 0.55, 0]),
            ]
        )
    else:
        mark = Cross(stroke_color=color, stroke_width=12)
        mark.scale_to_fit_width(1.05)
    badge = VGroup(ring, mark).scale(0.5)
    frame = panel[0]
    badge.move_to(
        frame.get_center() + np.array([PANEL_W / 2 - 0.62, PANEL_H / 2 - 0.62, 0])
    )
    return badge


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

        # Cadence is synced to outputs/voice_clips/002_*.mp3 (§2 of script.md,
        # ≈148 wpm). t=0 here is aligned to the word "Human" in the edit, i.e.
        # audio 8.65s; the clip then runs to the end of the VO (~21s later).
        # Phrase onsets, expressed in this clip's local timeline:
        #   Panel 1  "Human interventions need an expert tele-operator..."  t=0.0
        #   Panel 2  "and online RL is sample inefficient..."               t≈4.4
        #   Panel 3  "Offline RL via advantage conditioning..."             t≈8.9
        #   tail     "but prior work only uses sparse rewards..."           t≈15.4

        # 1) Interventions: expensive. [0.0 - 4.4s]
        p = panels[0]
        self.play(FadeIn(p[0], shift=UP * 0.2), run_time=0.5)
        self.play(GrowFromCenter(p[1]), run_time=0.6)
        self.play(Write(p[2]), Write(p[3]), run_time=0.9)
        self.play(Create(stamp(p, good=False)), run_time=0.5)
        self.wait(1.9)

        # 2) Online RL: sample inefficient. [4.4 - 8.9s]
        p = panels[1]
        self.play(FadeIn(p[0], shift=UP * 0.2), run_time=0.5)
        self.play(Create(p[1]), run_time=0.9)
        self.play(Write(p[2]), Write(p[3]), run_time=0.9)
        self.play(Create(stamp(p, good=False)), run_time=0.5)
        self.wait(1.7)

        # 3) Advantage conditioning — equation written on "advantage
        # conditioning", green tick on "shown some success". [8.9s - end]
        p = panels[2]
        self.play(FadeIn(p[0], shift=UP * 0.2), run_time=0.5)
        self.play(Create(p[1]), run_time=1.4)
        self.play(Write(p[2]), Write(p[3]), run_time=0.9)
        # The green tick stamps on top of the equation.
        self.play(Create(stamp(p, good=True)), run_time=0.5)
        # Let the winning panel pop.
        self.play(p.animate.scale(1.06), run_time=0.4)
        self.play(p.animate.scale(1 / 1.06), run_time=0.3)
        # Hold on the winner through the "but prior work..." limitation,
        # which sets up §3.
        self.wait(8.1)


if __name__ == "__main__":
    raise SystemExit(
        "Render with: uv run manim -qh generate_motivation_panels.py MotivationPanels"
    )
