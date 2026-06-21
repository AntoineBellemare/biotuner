"""
Concept stills for the "ride the line" harmonograph reel: a camera travels
ALONG the harmonograph curve (the pen/camera IS a moving point on the line),
leaving a glowing comet-trail, so the full figure emerges from the camera's
motion — then a pull-back reveals the whole harmonograph.

Renders three moments (riding 30%, riding 65%, full reveal) → out/proto/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
OUT = HERE / "out" / "proto"
OUT.mkdir(parents=True, exist_ok=True)
BG = "#05070d"


def harmonograph_xy():
    """Curve from the biotuner module if possible, else a faithful fallback."""
    try:
        from biotuner.harmonic_input import HarmonicInput
        from biotuner.harmonic_geometry.harmonograph import harmonograph_from_peaks
        g = harmonograph_from_peaks(
            peaks=[2.0, 3.0, 2.005, 3.002],
            amps=[1.0, 1.0, 0.8, 0.8],
            phases=[0.0, np.pi / 2, np.pi / 5, 0.0],
            damping=[0.011, 0.011, 0.011, 0.011],
            duration=70, sr=260,
        )
        xy = np.asarray(g.coordinates, dtype=float)
        return xy[:, 0], xy[:, 1]
    except Exception as e:  # noqa: BLE001
        print(f"  (module path failed: {e}; using numpy fallback)")
        t = np.linspace(0, 70, 70 * 260)
        d = np.exp(-0.011 * t)
        x = d * (np.sin(2.0 * 2 * np.pi * t) + 0.8 * np.sin(2.005 * 2 * np.pi * t + np.pi / 5))
        y = d * (np.sin(3.0 * 2 * np.pi * t + np.pi / 2) + 0.8 * np.sin(3.002 * 2 * np.pi * t))
        return x, y


def comet(ax, x, y, head, window=None, head_frac=None):
    """Draw the traveled portion [0:head] as a glowing comet trail, camera
    centred (and zoomed) on the head when `window` is given."""
    n = head
    xs, ys = x[:n], y[:n]
    if n < 2:
        return
    pts = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    # brightness + width ramp toward the head (comet)
    f = np.linspace(0, 1, len(segs)) ** 2.2
    # teal → gold gradient
    cols = np.zeros((len(segs), 4))
    cols[:, 0] = 0.45 + 0.5 * f      # R
    cols[:, 1] = 0.78 - 0.1 * f      # G
    cols[:, 2] = 0.72 - 0.45 * f     # B
    cols[:, 3] = 0.05 + 0.95 * f     # alpha (faint tail → bright head)
    lc = LineCollection(segs, colors=cols, linewidths=0.6 + 2.6 * f)
    ax.add_collection(lc)
    # glowing head
    hx, hy = x[n - 1], y[n - 1]
    for rad, a in [(120, 0.12), (60, 0.20), (24, 0.5), (8, 1.0)]:
        ax.scatter([hx], [hy], s=rad, c="#f4fbff", alpha=a, linewidths=0, zorder=5)
    if window is not None:
        ax.set_xlim(hx - window, hx + window)
        ax.set_ylim(hy - window, hy + window)


def main():
    x, y = harmonograph_xy()
    span = max(np.ptp(x), np.ptp(y)) / 2 * 1.1
    cx, cy = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2
    N = len(x)

    fig, axes = plt.subplots(1, 3, figsize=(4.6 * 3, 5.4), facecolor=BG)
    fig.suptitle("⟜  RIDE THE LINE — camera travels the harmonograph",
                 color="#e6ecf5", fontsize=21, y=0.98, fontweight="bold")

    # Panel A — riding, zoomed in, 30%
    comet(axes[0], x, y, int(N * 0.30), window=span * 0.16)
    axes[0].set_title("riding · the line builds ahead", color="#aeb9cc", fontsize=13)

    # Panel B — riding, medium zoom, 65%
    comet(axes[1], x, y, int(N * 0.65), window=span * 0.42)
    axes[1].set_title("riding · structure emerges", color="#aeb9cc", fontsize=13)

    # Panel C — full reveal, zoomed out
    comet(axes[2], x, y, N)
    axes[2].set_xlim(cx - span, cx + span)
    axes[2].set_ylim(cy - span, cy + span)
    axes[2].set_title("pull back · the whole figure", color="#aeb9cc", fontsize=13)

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_facecolor(BG)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    p = OUT / "harmonograph_camera.png"
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(p, dpi=120, facecolor=BG)
    print(f"  wrote {p.relative_to(HERE)}")


if __name__ == "__main__":
    main()
