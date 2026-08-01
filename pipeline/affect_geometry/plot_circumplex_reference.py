"""Reference diagram of Russell's (1980) circumplex for the paper.

Places the 28 circumplex words at their target angles on the unit circle
(angles from anchors_russell.json). Labels are spread along the circle with
a minimum angular separation and, near the vertical poles (where horizontal
text crowds fastest), staggered over two radii; a leader line connects a
label to its point whenever the label moved. Output:
figures/russell_circumplex_reference.{pdf,png}.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

PACKAGE = Path(__file__).resolve().parent
MIN_SEPARATION_DEG = 10.5
LABEL_RADIUS = 1.10
STAGGER_EXTRA = 0.135
POLE_BAND = 0.42  # |cos(angle)| below this counts as near a pole
LEADER_THRESHOLD_DEG = 2.0

INK = "#1a1a1a"
MUTED = "#8a8a8a"
FAINT = "#c9c9c9"


def spread_angles(angles: np.ndarray, minimum: float) -> np.ndarray:
    """Enforce a minimum circular separation, preserving the circular order.

    Works on the sorted, unwrapped sequence so pairs can never invert, which
    would stall a naive modular relaxation.
    """
    order = np.argsort(angles)
    unwrapped = angles[order].astype(float).copy()
    for _ in range(2000):
        moved = False
        for i in range(len(unwrapped) - 1):
            gap = unwrapped[i + 1] - unwrapped[i]
            if gap < minimum - 1e-9:
                shift = (minimum - gap) / 2.0
                unwrapped[i] -= shift
                unwrapped[i + 1] += shift
                moved = True
        wrap_gap = unwrapped[0] + 360.0 - unwrapped[-1]
        if wrap_gap < minimum - 1e-9:
            shift = (minimum - wrap_gap) / 2.0
            unwrapped[0] += shift
            unwrapped[-1] -= shift
            moved = True
        if not moved:
            break
    result = np.empty_like(unwrapped)
    result[order] = unwrapped % 360.0
    return result


def unit(angle_deg: float) -> np.ndarray:
    return np.array([np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))])


def main() -> None:
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    anchors = json.loads((PACKAGE / "anchors_russell.json").read_text())
    words = list(anchors["angles_degrees"])
    angles = np.asarray([anchors["angles_degrees"][word] for word in words])
    label_angles = spread_angles(angles, MIN_SEPARATION_DEG)

    fig, ax = plt.subplots(figsize=(3.3, 3.3))
    circle_theta = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(np.cos(circle_theta), np.sin(circle_theta), color=FAINT, lw=0.8, zorder=1)
    for x, y in ((1.0, 0.0), (0.0, 1.0)):
        ax.annotate("", xy=(1.02 * x, 1.02 * y), xytext=(-1.02 * x, -1.02 * y),
                    arrowprops=dict(arrowstyle="<->", color=FAINT, lw=0.7,
                                    shrinkA=0.0, shrinkB=0.0), zorder=1)
    for x, y, text in ((0.52, 0.062, "pleasant"), (-0.52, 0.062, "unpleasant"),
                       (0.0, 0.30, "high arousal"), (0.0, -0.30, "low arousal")):
        ax.text(x, y, text, ha="center", va="center", fontsize=6.0, color=MUTED,
                style="italic", zorder=2,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6))

    # Alternate label radii within each pole band so horizontal text can't collide.
    pole_parity: dict[bool, int] = {}
    radii = np.full(len(words), LABEL_RADIUS)
    for index in np.argsort(label_angles):
        cosine = np.cos(np.radians(label_angles[index]))
        if abs(cosine) < POLE_BAND:
            is_top = np.sin(np.radians(label_angles[index])) > 0
            parity = pole_parity.get(is_top, 0)
            radii[index] += STAGGER_EXTRA * parity
            pole_parity[is_top] = 1 - parity

    for word, angle, label_angle, radius in zip(words, angles, label_angles, radii):
        point = unit(angle)
        direction = unit(label_angle)
        label = radius * direction
        ax.plot(*point, marker="o", ms=2.6, color=INK, zorder=3)
        moved = abs((label_angle - angle + 180.0) % 360.0 - 180.0) > LEADER_THRESHOLD_DEG
        if moved or radius > LABEL_RADIUS:
            ax.plot(*zip(1.015 * point, label - 0.045 * direction),
                    color=FAINT, lw=0.5, zorder=2)
        cosine = direction[0]
        alignment = "left" if cosine > 0.25 else "right" if cosine < -0.25 else "center"
        ax.text(*label, word, ha=alignment, va="center", fontsize=7.0, color=INK, zorder=4)

    ax.set_xlim(-1.66, 1.66)
    ax.set_ylim(-1.46, 1.46)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    figures = PACKAGE / "figures"
    figures.mkdir(exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(figures / f"russell_circumplex_reference.{suffix}", dpi=300,
                    bbox_inches="tight", pad_inches=0.02)
    print("wrote", figures / "russell_circumplex_reference.pdf")


if __name__ == "__main__":
    main()
