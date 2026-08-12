"""Static visualisations of the scale labyrinth and its inhabitants.

The centrepiece is :func:`plot_labyrinth`, a faithful rendering of Milne et al.
(2011) Figures 1--2.  Reading it, per their §4:

- **angle** is the generator/period ratio -- top of the circle is 0, the bottom
  is 1/2, so 700 cents against a 1200-cent period sits at ``7/12`` of the way
  round.  The picture is left--right symmetric because a generator and its
  complement within the period build the same scale.
- **ring** is cardinality.  Ring ``N`` carries every ``N``-note MOS.
- **spokes** are equal temperaments.  Each spoke runs inward from the rim and
  *touches without crossing* the ring giving its number of notes.
- **arcs** are valid tuning ranges.  An arc on ring ``N`` spans the generators
  over which some ``N``-note MOS keeps its identity; the darker inner band is
  where it is also coherent.

Every function takes ``ax=None`` and returns ``(fig, ax)``, so figures compose.
"""

from __future__ import annotations

import math
import warnings
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Wedge

from biotuner.mos import theory as T
from biotuner.mos.scale import MOSScale, mos_family

__all__ = [
    "plot_labyrinth",
    "plot_stern_brocot",
    "plot_scale_wheel",
    "plot_step_covariation",
    "plot_modes",
    "plot_mode_lattice",
    "plot_mos_family",
    "plot_mos_fit",
    "plot_mos_trajectory",
    "plot_forward_vs_inverse",
    "plot_ji_landscape",
    "plot_fit_field",
    "plot_play_state",
    "LARGE_COLOR",
    "SMALL_COLOR",
    "INVERSE_COLOR",
    "FORWARD_CMAP",
]

# --- palette --------------------------------------------------------------- #
INK, MUTED, GRIDC = "#22252b", "#6a6f78", "#dfe1e4"
LARGE_COLOR = "#2E86AB"  # biotuner primary -- large steps
SMALL_COLOR = "#F18F01"  # biotuner accent  -- small steps
SIGNAL_COLOR = "#C73E1D"
TEMPERAMENT_COLOR = "#A23B72"
RING_CMAP = "viridis"
#: Colours for `highlight=`, deliberately excluding red so highlighted
#: generators never read as signal peaks.
HIGHLIGHT_COLORS = ("#1b3a6b", "#06A77D", "#7B3FA0", "#B8860B", "#2F4F4F")

# --- the two directions, drawn together ------------------------------------ #
#: Forward readings are shaded by how well they explain the signal: dark is
#: accurate, as in :func:`plot_ji_landscape`.  The ramp is truncated before its
#: pale end by :func:`_forward_colormap`, because a near-white marker on a white
#: page is a marker you cannot see.
FORWARD_CMAP = "magma"
#: The inverse fit's colour.  Blue appears nowhere on ``FORWARD_CMAP``, so the
#: one latent generator can never be mistaken for one more observed one -- and
#: it carries a star rather than a disc, so the distinction survives printing in
#: grey.
INVERSE_COLOR = "#1B6CA8"
#: The labyrinth when it is background rather than subject.
SCENERY_COLOR = "#8d94a1"

#: 5-limit consonances used as the default just-intonation target set.
JI_5_LIMIT: Tuple[float, ...] = (
    16 / 15, 9 / 8, 6 / 5, 5 / 4, 4 / 3, 3 / 2, 8 / 5, 5 / 3, 9 / 5, 15 / 8
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _polar_axes(ax, figsize):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    else:
        fig = ax.figure
        if ax.name != "polar":
            raise ValueError(
                "this plot needs a polar axes; create it with "
                "subplot_kw={'projection': 'polar'}"
            )
    # Milne et al. §4: zero at the top, increasing clockwise, 1/2 at the bottom.
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    return fig, ax


def _theta(fraction) -> float:
    """Generator fraction -> angle in radians."""
    return 2.0 * math.pi * float(fraction)


def _arc(t0: float, t1: float, n: int = 64) -> np.ndarray:
    return np.linspace(t0, t1, n)


def _sb_parent(node: T.SBNode) -> Optional[Fraction]:
    """The tree node this one branched from, or ``None`` for the root ``1/2``.

    The bracketing endpoint with the larger denominator is always the one that
    was added most recently, hence the parent; ``0/1`` and ``1/1`` are the
    boundary pair, not nodes.
    """
    candidates = [f for f in (node.left, node.right) if f.denominator >= 2]
    if not candidates:
        return None
    return max(candidates, key=lambda f: f.denominator)


def _as_scale(obj) -> MOSScale:
    """Accept a MOSScale or a Mode and return the underlying scale."""
    return obj.scale if hasattr(obj, "scale") else obj


def _bright(generator: float) -> float:
    """Fold a generator fraction into the bright half ``[0.5, 1)``.

    The drawing counterpart of :func:`biotuner.mos.derive._fold_bright`, and it
    must agree with it: a generator and its complement build the same scale, so
    both directions of the fit report the bright spelling only, and a picture
    that placed one of them in the dark half would invent a difference that is
    not there.  Where the fitting helper *rejects* the degenerate values, this
    one clamps them -- a plot has to draw whatever it is handed.
    """
    g = float(generator) % 1.0
    return 1.0 - g if g < 0.5 else g


def _forward_colormap(name: str = FORWARD_CMAP, hi: float = 0.80):
    """``name`` with its palest end cut off, so every marker stays visible."""
    base = plt.get_cmap(name)
    return ListedColormap(base(np.linspace(0.0, hi, 256)), name=f"{name}_trunc")


def _dim_to_scenery(ax, factor: float = 0.62, color: str = SCENERY_COLOR) -> None:
    """Fade an already-drawn labyrinth so an overlay reads as the subject.

    Called *after* :func:`plot_labyrinth` and *before* anything is overlaid, so
    only the scenery is touched.  Existing alphas are *scaled* rather than
    replaced, which keeps the labyrinth's own hierarchy -- the coherent band
    still stands out from the valid range, the spokes still recede -- while
    putting the whole of it behind the data.  The ring colouring does not
    survive: a viridis ramp underneath a data colourmap is two colour scales
    competing for one meaning.
    """
    for line in ax.get_lines():
        line.set_color(color)
        line.set_alpha((line.get_alpha() if line.get_alpha() is not None else 1.0)
                       * factor)
        line.set_linewidth(line.get_linewidth() * 0.85)
        line.set_zorder(min(line.get_zorder(), 1.5))
    for text in ax.texts:
        text.set_alpha(factor)


def _source_size(n_sources: int) -> float:
    """Marker area for a forward reading proposed by ``n`` observed intervals.

    Linear in the count and generous in its step, because the difference
    between "one peak pair happens to state this" and "three of them agree" is
    the difference between a coincidence and a finding, and a two-point size
    increment does not say that.  Capped so a densely corroborated generator
    cannot swallow its neighbours on the ring.
    """
    return min(44.0 + 78.0 * (max(int(n_sources), 1) - 1), 320.0)


def _tag_label(index: int) -> str:
    """Letters for the generator tags, so they cannot be read as ring numbers.

    The radial axis is already labelled with integers -- the cardinalities --
    and a numbered tag beside them is an invitation to read "3" as three notes.
    """
    return chr(ord("A") + index) if index < 26 else str(index + 1)


# --------------------------------------------------------------------------- #
# The labyrinth
# --------------------------------------------------------------------------- #
def plot_labyrinth(
    max_cardinality: int = 18,
    *,
    period: float = 2.0,
    periods_per_octave: int = 1,
    highlight: Union[None, float, MOSScale, Sequence] = None,
    peaks: Optional[Sequence[float]] = None,
    peak_weights: Optional[Sequence[float]] = None,
    temperaments: bool = False,
    temperament_tuning: str = "pote",
    label: str = "cents",
    n_labels: int = 12,
    show_arcs: bool = True,
    show_spokes: bool = True,
    show_coherence: bool = True,
    annotate: Optional[int] = None,
    generator_range: Optional[Tuple[float, float]] = None,
    ax=None,
    figsize: Tuple[float, float] = (9.0, 9.0),
):
    """The scale labyrinth: every well-formed scale up to ``max_cardinality``.

    Parameters
    ----------
    max_cardinality : int, default 18
        Outermost ring.  Milne et al. Fig. 1 shows 18.
    period : float, default 2.0
        The interval the whole circle spans, as a frequency ratio.
    periods_per_octave : int, default 1
        Draw a scale whose period is ``period ** (1 / n)`` -- a *fractional*
        period, as srutal (2), augmented (3), blackwood (5) and compton (12)
        have.  The scale then repeats ``n`` times around the circle, giving the
        labyrinth ``n``-fold rotational symmetry, and rings count notes per
        octave rather than per period, so only multiples of ``n`` appear.
        ``1`` leaves the ordinary labyrinth untouched.
    highlight : float, MOSScale, or sequence, optional
        A generator fraction, a scale, or several of either.  Each gets a bold
        radial line, and every MOS in its family gets a marker on its ring --
        so you can read a generator's whole scale series off the picture.
    peaks : sequence of float, optional
        Biosignal peak ratios to overlay, drawn just outside the rim at the
        angle each one occupies in the labyrinth.
    peak_weights : sequence of float, optional
        Marker areas, e.g. peak amplitudes.
    temperaments : bool, default False
        Overlay named rank-2 temperaments as radial lines at their optimal
        generators -- the red lines of Milne et al. Fig. 1.  Only those whose
        period is the whole octave are drawn.
    temperament_tuning : {'pote', 'cte'}, default 'pote'
        Which optimum to place the lines at.  POTE is what published
        temperament tables quote, so it is the default and the one to use when
        cross-checking against them; CTE constrains the period pure from the
        outset instead.  See :mod:`biotuner.mos.temperaments`.
    label : {'cents', 'fraction', 'none'}, default 'cents'
    n_labels : int, default 12
    show_arcs, show_spokes, show_coherence : bool
    annotate : int, optional
        Write the ``nLms`` signature beside every arc up to this cardinality.
        Unreadable above about 9 rings.
    generator_range : (float, float), optional
        Restrict the angular window to these generator fractions -- the zoom of
        Milne et al. Fig. 6.  ``(0.5, 0.62)`` frames the diatonic region.
    ax : matplotlib polar axes, optional
    figsize : tuple

    Returns
    -------
    (fig, ax)

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> fig, ax = plot_labyrinth(12)
    >>> bool(ax.get_ylim()[1] > 12)
    True
    >>> plt.close(fig)
    """
    if max_cardinality < 2:
        raise ValueError(f"max_cardinality must be at least 2, got {max_cardinality}")
    if label not in ("cents", "fraction", "none"):
        raise ValueError(
            f"label must be 'cents', 'fraction' or 'none', got {label!r}"
        )
    n_per = int(periods_per_octave)
    if n_per < 1:
        raise ValueError(
            f"periods_per_octave must be at least 1, got {periods_per_octave!r}"
        )
    per_period_max = max_cardinality // n_per
    if per_period_max < 2:
        raise ValueError(
            f"max_cardinality={max_cardinality} leaves only {per_period_max} "
            f"notes per period at periods_per_octave={n_per}; raise one or "
            "lower the other"
        )
    fig, ax = _polar_axes(ax, figsize)

    nodes = T.sb_tree_nodes(per_period_max)
    rim = max_cardinality + 0.8
    cmap = plt.get_cmap(RING_CMAP)

    def copies(fraction) -> List[float]:
        """Angles of one period-fraction, once per period in the circle.

        With a whole-octave period this is a single angle and everything below
        reduces to the ordinary labyrinth.  With ``n`` periods per octave the
        scale simply repeats, so each feature is drawn ``n`` times a turn --
        which is what gives those temperaments their rotational symmetry.
        """
        v = float(fraction)
        return [2.0 * math.pi * (v + k) / n_per for k in range(n_per)]

    for node in nodes:
        # Rings count notes per octave, so a fractional period multiplies up.
        card = node.cardinality * n_per
        colour = cmap((card - 1) / max(1, max_cardinality - 1))
        c_lo = T.mediant(node.left, node.node)
        c_hi = T.mediant(node.node, node.right)

        for k, t_mid in enumerate(copies(node.node)):
            t_left = copies(node.left)[k]
            t_right = copies(node.right)[k]

            if show_arcs:
                # Full valid range of the MOS pair living on this ring.
                arc = _arc(t_left, t_right)
                ax.plot(arc, np.full_like(arc, card), color=colour, lw=2.0,
                        alpha=0.55, solid_capstyle="butt", zorder=2)
                if show_coherence:
                    # Coherent while Blackwood's R < 2: between the embedding
                    # tunings on either side of the equalized node.
                    band = _arc(copies(c_lo)[k], copies(c_hi)[k])
                    ax.plot(band, np.full_like(band, card), color=colour, lw=4.5,
                            alpha=0.95, solid_capstyle="butt", zorder=3)
                # The equalized landmark: L and s the same size, MOS meets inverse.
                ax.plot([t_mid], [card], marker="|", ms=7, color=INK, alpha=0.7,
                        zorder=4)

            if show_spokes:
                # An equal temperament: in from the rim, stopping on its ring.
                ax.plot([t_mid, t_mid], [rim, card], color=MUTED, lw=0.6,
                        alpha=0.45, zorder=1)

            if annotate and card <= annotate:
                b, d = node.left.denominator, node.right.denominator
                ax.text(copies(T.mediant(node.left, node.node))[k], card + 0.28,
                        f"{b}L{d}s", fontsize=6.5, color=INK, ha="center",
                        va="bottom")
                ax.text(copies(T.mediant(node.node, node.right))[k], card + 0.28,
                        f"{d}L{b}s", fontsize=6.5, color=INK, ha="center",
                        va="bottom")

    # --- overlays --------------------------------------------------------- #
    if temperaments:
        _overlay_temperaments(ax, rim, temperament_tuning)

    if highlight is not None:
        items = highlight if isinstance(highlight, (list, tuple)) else [highlight]
        for i, item in enumerate(items):
            g = item.generator if isinstance(item, MOSScale) else float(item)
            colour = HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)]
            per = item.period if isinstance(item, MOSScale) else period
            fam = mos_family(
                T.fraction_to_generator(g, per), per_period_max, per
            )
            for theta in copies(g):
                ax.plot([theta, theta], [0, rim], color=colour, lw=1.8,
                        alpha=0.9, zorder=6)
                for scale in fam:
                    ax.plot([theta], [scale.cardinality * n_per], "o", ms=6,
                            color=colour, mec=INK, mew=0.5, zorder=7)

    if peaks is not None:
        pos = np.mod(np.log(np.asarray(peaks, float)) / math.log(period), 1.0)
        if peak_weights is None:
            sizes = np.full(len(pos), 55.0)
        else:
            w = np.asarray(peak_weights, float)
            w = w / (w.max() or 1.0)
            sizes = 25.0 + 110.0 * w
        for t in pos:
            ax.plot([_theta(t), _theta(t)], [0, rim], color=SIGNAL_COLOR, lw=0.9,
                    alpha=0.35, ls="--", zorder=5)
        ax.scatter([_theta(t) for t in pos], np.full(len(pos), rim), s=sizes,
                   color=SIGNAL_COLOR, edgecolor="white", linewidth=0.6, zorder=8)

    # --- frame ------------------------------------------------------------ #
    ax.set_rlim(0, rim + 0.6)
    ax.set_rticks(list(range(1, max_cardinality + 1)))
    ax.set_yticklabels([str(n) if n % 2 == 0 or n <= 7 else ""
                        for n in range(1, max_cardinality + 1)], fontsize=6.5,
                       color=MUTED)
    ax.set_rlabel_position(0.0)
    ax.grid(color=GRIDC, lw=0.4, alpha=0.5)

    if label == "none":
        ax.set_xticks([])
    else:
        ticks = np.linspace(0, 2 * math.pi, n_labels, endpoint=False)
        ax.set_xticks(ticks)
        pc = T.PERIOD_CENTS * math.log2(period)
        if label == "cents":
            ax.set_xticklabels(
                [f"{t / (2 * math.pi) * pc:.0f}" for t in ticks], fontsize=7.5,
                color=MUTED
            )
        else:
            ax.set_xticklabels(
                [str(Fraction(t / (2 * math.pi)).limit_denominator(n_labels))
                 for t in ticks], fontsize=7.5, color=MUTED
            )

    # Applied last: set_xticks re-autoscales the angular view, so narrowing the
    # window before laying out the tick labels silently un-narrows it again.
    if generator_range is not None:
        lo, hi = generator_range
        ax.set_thetamin(360.0 * lo)
        ax.set_thetamax(360.0 * hi)

    handles = [
        Line2D([], [], color=MUTED, lw=0.6, label="equal temperament (spoke)"),
        Line2D([], [], color="#4c8f7d", lw=2.0, alpha=0.55, label="valid tuning range"),
        Line2D([], [], color="#4c8f7d", lw=4.5, label="coherent (R < 2)"),
    ]
    if peaks is not None:
        handles.append(
            Line2D([], [], color=SIGNAL_COLOR, marker="o", ls="--", lw=0.9,
                   label="signal peaks")
        )
    if temperaments:
        handles.append(
            Line2D([], [], color=TEMPERAMENT_COLOR, lw=1.0, label="temperament")
        )
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=8, frameon=False)
    ax.set_title(
        f"Scale labyrinth — MOS scales to {max_cardinality} notes",
        fontsize=13, pad=18, color=INK,
    )
    return fig, ax


def _overlay_temperaments(ax, rim: float, tuning: str = "pote") -> None:
    """Radial lines at named temperaments' optimal generators (Fig. 1's red lines)."""
    if tuning not in ("pote", "cte"):
        raise ValueError(
            f"temperament_tuning must be 'pote' or 'cte', got {tuning!r}"
        )
    try:
        from biotuner.mos.temperaments import all_temperaments
    except Exception as exc:  # pragma: no cover - optional overlay
        warnings.warn(
            f"temperament overlay unavailable ({exc.__class__.__name__}: {exc}); "
            "drawing the labyrinth without it",
            stacklevel=3,
        )
        return
    for name, temp in all_temperaments().items():
        if getattr(temp, "periods_per_octave", 1) != 1:
            # A fractional period would need its own set of spokes.
            continue
        cents = (temp.pote_generator_cents if tuning == "pote"
                 else temp.generator_cents)
        theta = _theta((cents / T.PERIOD_CENTS) % 1.0)
        ax.plot([theta, theta], [0, rim], color=TEMPERAMENT_COLOR, lw=1.0,
                alpha=0.8, zorder=6)
        ax.text(theta, rim + 0.35, name, fontsize=6, color=TEMPERAMENT_COLOR,
                ha="center", va="bottom", rotation=0)


# --------------------------------------------------------------------------- #
# The tree itself
# --------------------------------------------------------------------------- #
def plot_stern_brocot(
    max_cardinality: int = 12,
    *,
    highlight_generator: Optional[float] = None,
    annotate_signatures: bool = True,
    ax=None,
    figsize: Tuple[float, float] = (12.0, 6.0),
):
    """The Stern-Brocot tree, with each node's MOS signatures.

    The labyrinth is this tree bent into a circle (Milne et al. §1).  Drawn
    flat, the parent/child structure is explicit: each node is the mediant of
    the pair bracketing it, and its denominator is the cardinality of the MOS
    whose two step sizes equalise there.

    Parameters
    ----------
    max_cardinality : int, default 12
    highlight_generator : float, optional
        Generator fraction whose path down the tree to trace.
    annotate_signatures : bool, default True
        Label each node with the ``bLds`` / ``dLbs`` pair its two sub-ranges host.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> fig, ax = plot_stern_brocot(7, highlight_generator=math.log2(3 / 2))
    >>> plt.close(fig)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    nodes = T.sb_tree_nodes(max_cardinality)
    xy: Dict[Fraction, Tuple[float, float]] = {
        n.node: (float(n.node), -float(n.depth)) for n in nodes
    }

    path = set()
    if highlight_generator is not None:
        path = {
            n.node
            for n in T.sb_walk(highlight_generator, max_cardinality=max_cardinality)
        }

    segments, colours, widths = [], [], []
    for node in nodes:
        parent = _sb_parent(node)
        if parent is None or parent not in xy:
            continue
        segments.append([xy[parent], xy[node.node]])
        on_path = node.node in path and parent in path
        colours.append(SIGNAL_COLOR if on_path else GRIDC)
        widths.append(2.0 if on_path else 0.8)
    if segments:
        ax.add_collection(
            LineCollection(segments, colors=colours, linewidths=widths, zorder=1)
        )

    for node in nodes:
        x, y = xy[node.node]
        on_path = node.node in path
        ax.plot([x], [y], "o", ms=6 if on_path else 4,
                color=SIGNAL_COLOR if on_path else LARGE_COLOR,
                mec=INK, mew=0.4, zorder=3)
        ax.text(x, y + 0.16, f"{node.node.numerator}/{node.node.denominator}",
                fontsize=7.5 if on_path else 6.5, ha="center", va="bottom",
                color=INK, fontweight="bold" if on_path else "normal")
        if annotate_signatures:
            b, d = node.left.denominator, node.right.denominator
            ax.text(x, y - 0.20, f"{b}L{d}s | {d}L{b}s", fontsize=5.5,
                    ha="center", va="top", color=MUTED)

    ax.set_xlabel("generator / period", fontsize=10, color=INK)
    ax.set_ylabel("tree depth", fontsize=10, color=INK)
    ax.set_yticks(sorted({-n.depth for n in nodes}))
    ax.set_yticklabels([str(int(-t)) for t in sorted({-n.depth for n in nodes})],
                       fontsize=8)
    ax.set_xlim(-0.04, 1.04)
    ax.spines[["top", "right"]].set_visible(False)
    title = f"Stern-Brocot tree to {max_cardinality} notes"
    if highlight_generator is not None:
        title += (
            f"   —   path of g = {highlight_generator:.6f} "
            f"({highlight_generator * T.PERIOD_CENTS:.1f} c)"
        )
    ax.set_title(title, fontsize=12, color=INK)
    return fig, ax


# --------------------------------------------------------------------------- #
# One scale, up close
# --------------------------------------------------------------------------- #
def plot_scale_wheel(
    scale,
    *,
    show_labels: bool = True,
    ax=None,
    figsize: Tuple[float, float] = (7.0, 7.0),
):
    """A scale as a circular keyboard, keys as wide as the steps above them.

    Milne et al. §5 and Fig. 8: "the keyboard layout for a finite scale can have
    specific key widths which are proportional to the sizes of the step
    intervals above each tone".  Large and small steps are coloured apart, so
    the maximally even distribution of the MOS word is visible at a glance.

    Accepts an :class:`~biotuner.mos.scale.MOSScale` or a
    :class:`~biotuner.mos.modes.Mode`.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> fig, ax = plot_scale_wheel(MOSScale.from_signature(5, 2, tuning=12))
    >>> plt.close(fig)
    """
    fig, ax = _polar_axes(ax, figsize)
    base = _as_scale(scale)
    degrees = list(scale.degrees)
    word = scale.word
    cents = list(scale.cents)

    for i, d in enumerate(degrees):
        nxt = degrees[i + 1] if i + 1 < len(degrees) else 1.0
        t0, t1 = math.degrees(_theta(d)), math.degrees(_theta(nxt))
        colour = LARGE_COLOR if word[i] == "L" else SMALL_COLOR
        # Wedge works in the data's own angular convention; the axes handle the
        # zero-location and direction, so plot in raw degrees.
        ax.bar(
            x=_theta((d + nxt) / 2.0),
            height=1.0,
            width=_theta(nxt - d),
            bottom=0.0,
            color=colour,
            edgecolor="white",
            linewidth=1.2,
            alpha=0.85,
        )
        if show_labels:
            ax.text(_theta(d), 1.12, f"{cents[i]:.0f}", fontsize=8, ha="center",
                    va="center", color=INK)

    ax.set_rlim(0, 1.3)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    name = getattr(scale, "name", base.signature)
    ax.set_title(
        f"{name}   {word}\n"
        f"L = {base.step_cents[0]:.1f} c,  s = {base.step_cents[1]:.1f} c,  "
        f"R = {base.hardness:.2f}",
        fontsize=11, pad=16, color=INK,
    )
    ax.legend(
        handles=[
            Line2D([], [], color=LARGE_COLOR, lw=8, label="large step"),
            Line2D([], [], color=SMALL_COLOR, lw=8, label="small step"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=9,
        frameon=False,
    )
    return fig, ax


def plot_step_covariation(
    n_large: int,
    n_small: int,
    *,
    bright: bool = True,
    period: float = 2.0,
    mark: Optional[float] = None,
    ax=None,
    figsize: Tuple[float, float] = (10.0, 6.5),
):
    """How the two step sizes trade off as the generator moves.

    This draws the paragraph Milne et al. §2 calls "Landmark equal tunings":
    across the valid range the large and small steps co-vary, always summing to
    the period, and they pass through three distinguished tunings -- one where
    they become equal (and the scale meets its inverse), and one on each side
    where a step size reaches zero.

    The span shown covers the MOS *and* its inverse, meeting at the equalized
    landmark, so the whole story is in one frame.  The lower panel tracks
    Blackwood's ``R``; the scale is coherent below ``R = 2``.

    Parameters
    ----------
    n_large, n_small : int
    bright : bool, default True
        Which of the two mirror ranges to show.
    period : float, default 2.0
    mark : float, optional
        A generator fraction to mark with a vertical line, e.g. the tuning you
        actually fitted.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> fig, axes = plot_step_covariation(5, 2)
    >>> plt.close(fig)
    """
    if ax is None:
        fig, axes = plt.subplots(
            2, 1, figsize=figsize, sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.12},
        )
    else:
        fig, axes = ax.figure, np.atleast_1d(ax)

    lm = T.mos_landmarks(n_large, n_small, bright=bright)
    card = n_large + n_small
    pc = T.PERIOD_CENTS * math.log2(period)

    lo = float(min(lm.large_vanishes, lm.small_vanishes))
    hi = float(max(lm.large_vanishes, lm.small_vanishes))
    inset = (hi - lo) * 1e-4
    gs = np.linspace(lo + inset, hi - inset, 900)

    large = np.empty_like(gs)
    small = np.empty_like(gs)
    for i, g in enumerate(gs):
        a, b = T.step_sizes(float(g), card)
        large[i], small[i] = a * pc, b * pc

    top = axes[0]
    top.plot(gs * pc, large, color=LARGE_COLOR, lw=2.0, label="large step")
    top.plot(gs * pc, small, color=SMALL_COLOR, lw=2.0, label="small step")
    top.axhline(pc / card, color=MUTED, lw=0.8, ls=":",
                label=f"{card}-EDO step ({pc / card:.1f} c)")

    for frac, tag in (
        (lm.equalized, f"equalized · {lm.equalized_edo}-EDO"),
        (lm.small_vanishes, f"s → 0 · {lm.small_vanishes_edo}-EDO"),
        (lm.large_vanishes, f"L → 0 · {lm.large_vanishes_edo}-EDO"),
    ):
        x = float(frac) * pc
        for a in axes:
            a.axvline(x, color=INK, lw=0.9, ls="--", alpha=0.6)
        top.text(x, top.get_ylim()[1], f" {frac}\n {tag}", fontsize=7.5,
                 color=INK, rotation=90, va="top", ha="left")

    c_lo, c_hi = T.coherence_range(n_large, n_small, bright=bright)
    for a in axes:
        a.axvspan(float(c_lo) * pc, float(c_hi) * pc, color=LARGE_COLOR,
                  alpha=0.08, lw=0)
    if mark is not None:
        for a in axes:
            a.axvline(float(mark) * pc, color=SIGNAL_COLOR, lw=1.6)

    top.set_ylabel("step size (cents)", fontsize=10, color=INK)
    top.legend(fontsize=8, frameon=False, loc="center left")
    top.spines[["top", "right"]].set_visible(False)
    top.set_title(
        f"{n_large}L{n_small}s and its inverse {n_small}L{n_large}s — "
        f"step sizes across the valid generator range",
        fontsize=12, color=INK,
    )

    bot = axes[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(small > 0, large / small, np.inf)
        r = np.maximum(r, 1.0 / np.maximum(r, 1e-12))  # report L/s >= 1 either side
    bot.plot(gs * pc, r, color=INK, lw=1.6)
    bot.axhline(2.0, color=SIGNAL_COLOR, lw=1.0, ls="--")
    bot.text(gs[0] * pc, 2.05, " R = 2: edge of coherence", fontsize=7.5,
             color=SIGNAL_COLOR, va="bottom")
    bot.set_ylim(1.0, 5.0)
    bot.set_ylabel("hardness R", fontsize=10, color=INK)
    bot.set_xlabel("generator (cents)", fontsize=10, color=INK)
    bot.spines[["top", "right"]].set_visible(False)
    return fig, axes


def plot_modes(
    scale: MOSScale,
    *,
    ax=None,
    figsize: Tuple[float, float] = (11.0, 6.0),
):
    """Every mode of a scale, stacked brightest to darkest.

    Each row is one mode's step pattern drawn to scale.  Read down the stack and
    the parsimony of Milne et al. §4 is visible directly: between neighbouring
    rows exactly one boundary moves, by the chroma ``L - s``.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> fig, ax = plot_modes(MOSScale.from_signature(5, 2, tuning=12))
    >>> plt.close(fig)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    modes = scale.modes()
    pc = scale.period_cents
    for row, mode in enumerate(modes):
        y = len(modes) - row - 1
        cents = list(mode.cents) + [pc]
        for i in range(mode.cardinality):
            width = cents[i + 1] - cents[i]
            ax.barh(y, width, left=cents[i], height=0.72,
                    color=LARGE_COLOR if mode.word[i] == "L" else SMALL_COLOR,
                    edgecolor="white", linewidth=1.0, alpha=0.85)
        for c in cents[:-1]:
            ax.plot([c], [y], marker="|", color=INK, ms=10, mew=0.8, zorder=3)

    ax.set_yticks(range(len(modes)))
    ax.set_yticklabels([m.name for m in reversed(modes)], fontsize=9)
    ax.set_xlabel("cents", fontsize=10, color=INK)
    ax.set_xlim(0, pc)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    chroma = modes[0].chroma
    ax.set_title(
        f"Modes of {scale.signature} at {scale.generator_cents:.1f} c "
        f"— brightest at top; one tone moves by the chroma ({chroma:.1f} c) per step",
        fontsize=11.5, color=INK,
    )
    ax.legend(
        handles=[
            Line2D([], [], color=LARGE_COLOR, lw=8, label="large step"),
            Line2D([], [], color=SMALL_COLOR, lw=8, label="small step"),
        ],
        loc="lower right", fontsize=9, frameon=False,
    )
    return fig, ax


def plot_mode_lattice(
    scale: MOSScale,
    *,
    width: int = 4,
    height: int = 3,
    base: int = 0,
    ax=None,
    figsize: Tuple[float, float] = (12.0, 5.5),
):
    """The modal ℤ² lattice and one mode's fundamental frame (Fig. 7).

    Left panel: a patch of the free commutative group generated by the two
    commuting transformations -- chromatic transposition ``τ`` rightwards
    (same finalis, origin a generator sharper) and diatonic transposition ``σ``
    downwards (same collection, finalis a step higher).

    Right panel: the base mode's own frame, each degree placed at its
    (generator, period) lattice coordinates.  The zig-zag is the step pattern.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> fig, axes = plot_mode_lattice(MOSScale.from_signature(5, 2, tuning=12))
    >>> plt.close(fig)
    """
    from biotuner.mos.modes import Mode, mode_lattice

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize,
                                 gridspec_kw={"width_ratios": [1.35, 1.0]})
    else:
        fig, axes = ax.figure, np.atleast_1d(ax)

    left = axes[0]
    grid = mode_lattice(scale, width=width, height=height, base=base)
    for j, row in enumerate(grid):
        for i, mode in enumerate(row):
            left.add_patch(
                plt.Rectangle((i - 0.45, -j - 0.35), 0.9, 0.7, facecolor="white",
                              edgecolor=GRIDC, lw=1.0, zorder=1)
            )
            left.text(i, -j + 0.06, mode.name, fontsize=8.5, ha="center",
                      va="center", color=INK, zorder=2)
            left.text(i, -j - 0.16, mode.word, fontsize=7, ha="center",
                      va="center", color=MUTED, family="monospace", zorder=2)
    if width > 1:
        left.annotate("", xy=(1.45, 0.55), xytext=(-0.45, 0.55),
                      arrowprops=dict(arrowstyle="->", color=LARGE_COLOR, lw=1.6))
        left.text(0.5, 0.68, "τ  chromatic: origin one generator flatwards",
                  fontsize=8, ha="center", color=LARGE_COLOR)
    if height > 1:
        left.annotate("", xy=(-0.75, -1.35), xytext=(-0.75, 0.35),
                      arrowprops=dict(arrowstyle="->", color=SMALL_COLOR, lw=1.6))
        left.text(-0.85, -0.5, "σ  diatonic: finalis one step higher", fontsize=8,
                  rotation=90, va="center", ha="right", color=SMALL_COLOR)
    left.set_xlim(-1.6, width - 0.2)
    left.set_ylim(-height + 0.2, 1.0)
    left.axis("off")
    left.set_title("modal ℤ², freely generated by σ and τ", fontsize=11, color=INK)

    right = axes[1]
    coords = Mode(scale, base).lattice_coords()
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    right.plot(xs, ys, "-o", color=LARGE_COLOR, lw=1.6, ms=7, mec=INK, mew=0.5)
    for (x, y), c in zip(coords, Mode(scale, base).cents):
        right.annotate(f"{c:.0f}", (x, y), textcoords="offset points",
                       xytext=(0, 9), fontsize=7.5, ha="center", color=INK)
    right.set_xlabel("generators from the finalis (width)", fontsize=9.5, color=INK)
    right.set_ylabel("periods (height)", fontsize=9.5, color=INK)
    right.set_yticks(sorted(set(ys)))
    right.set_xticks(xs)
    right.grid(color=GRIDC, lw=0.6)
    right.set_axisbelow(True)
    right.spines[["top", "right"]].set_visible(False)
    right.set_title(f"fundamental frame of {Mode(scale, base).name}", fontsize=11,
                    color=INK)
    return fig, axes


def plot_mos_family(
    generator: float,
    *,
    max_cardinality: int = 24,
    period: float = 2.0,
    ax=None,
    figsize: Tuple[float, float] = (11.0, 6.0),
):
    """Every MOS a generator produces, stacked by cardinality.

    The nesting Milne et al. §2 call embedding: each scale's tones are a subset
    of the next one down, because they are all the same generator chain cut at
    different lengths.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> fig, ax = plot_mos_family(3 / 2, max_cardinality=17)
    >>> plt.close(fig)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    scales = mos_family(generator, max_cardinality=max_cardinality, period=period)
    if not scales:
        raise ValueError(
            f"generator {generator!r} produces no MOS at or below "
            f"{max_cardinality} notes"
        )
    pc = scales[0].period_cents
    for row, scale in enumerate(scales):
        y = len(scales) - row - 1
        cents = list(scale.cents) + [pc]
        for i in range(scale.cardinality):
            ax.barh(y, cents[i + 1] - cents[i], left=cents[i], height=0.7,
                    color=LARGE_COLOR if scale.word[i] == "L" else SMALL_COLOR,
                    edgecolor="white", linewidth=0.9, alpha=0.85)

    ax.set_yticks(range(len(scales)))
    ax.set_yticklabels(
        [f"{s.signature}  ({s.cardinality})" for s in reversed(scales)], fontsize=9
    )
    ax.set_xlim(0, pc)
    ax.set_xlabel("cents", fontsize=10, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title(
        f"MOS family of generator {scales[0].generator_cents:.2f} c "
        f"(ratio {generator:.6f})",
        fontsize=12, color=INK,
    )
    return fig, ax


# --------------------------------------------------------------------------- #
# Fits and trajectories
# --------------------------------------------------------------------------- #
def plot_mos_fit(
    fit,
    ratios: Optional[Sequence[float]] = None,
    *,
    weights: Optional[Sequence[float]] = None,
    ax=None,
    figsize: Tuple[float, float] = (11.0, 6.0),
):
    """A fitted MOS against the ratios it was fitted to.

    Top: the scale's degrees as vertical lines, with each target ratio placed
    where it actually falls.  Bottom: the signed residual per target, against
    the error a scale of this size would get on random input -- the band inside
    which a fit is not evidence of anything.

    Parameters
    ----------
    fit : MOSFit
    ratios : sequence of float, optional
        The targets.  Taken from ``fit`` alone if omitted, in which case only
        the residuals are drawn.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> from biotuner.mos.derive import best_mos
    >>> r = MOSScale.from_signature(4, 3, tuning=19).ratios
    >>> fig, axes = plot_mos_fit(best_mos(r, max_cardinality=12), r)
    >>> plt.close(fig)
    """
    if ax is None:
        fig, axes = plt.subplots(
            2, 1, figsize=figsize, sharex=True,
            gridspec_kw={"height_ratios": [1.6, 1.0], "hspace": 0.15},
        )
    else:
        fig, axes = ax.figure, np.atleast_1d(ax)

    scale = fit.scale
    pc = scale.period_cents
    if ratios is not None and fit.targets and len(ratios) != fit.n_targets:
        # The fit folded octave-equivalent ratios together, so the caller's list
        # no longer runs parallel to the residuals; draw what was fitted.
        ratios = list(fit.targets)
        weights = None
    top = axes[0]
    # Draw the scale where the fit actually put it. A scale and its
    # transpositions are one scale, so `fit` is free to rotate it onto the data
    # and records that rotation in `offset`; `scale.cents` is rooted on the
    # generator chain's arbitrary origin. Drawing the unrotated degrees puts the
    # lines up to a whole step away from the very ratios they matched -- a fit
    # with 0.000 cents of error rendered with a 77-cent gap between every dot
    # and its line.
    #
    # Not `fit.aligned_cents`: that re-roots the scale onto the data's own
    # reference to report the mode, which is a different question and shifts
    # the degrees again.
    shift = fit.offset * pc
    for i, c in enumerate(scale.cents):
        used = i in set(fit.assignments)
        x = (c + shift) % pc
        top.axvline(x, color=LARGE_COLOR if used else GRIDC,
                    lw=1.8 if used else 1.0, alpha=0.9 if used else 0.8)
        top.text(x, 1.02, f"{i}", fontsize=7, ha="center", va="bottom",
                 color=INK if used else MUTED, transform=top.get_xaxis_transform())

    if ratios is not None:
        pos = np.mod(np.log(np.asarray(ratios, float)) / math.log(scale.period), 1.0)
        target_cents = pos * pc
        if weights is None:
            sizes = np.full(len(pos), 70.0)
        else:
            w = np.asarray(weights, float)
            sizes = 25.0 + 130.0 * (w / (w.max() or 1.0))
        top.scatter(target_cents, np.full(len(pos), 0.5), s=sizes,
                    color=SIGNAL_COLOR, zorder=5, edgecolor="white", linewidth=0.7)
    top.set_ylim(0, 1)
    top.set_yticks([])
    top.spines[["top", "right", "left"]].set_visible(False)
    top.set_title(
        f"{scale.signature} at {scale.generator_cents:.2f} c   —   "
        f"weighted error {fit.error_cents:.2f} c, "
        f"{fit.improvement:.1f}× better than chance",
        fontsize=12, color=INK,
    )

    bot = axes[-1]
    res = np.asarray(fit.residuals, float)
    if ratios is not None:
        x = np.mod(np.log(np.asarray(ratios, float)) / math.log(scale.period), 1.0) * pc
    else:
        x = np.array([scale.cents[i] for i in fit.assignments], float)
    bot.axhspan(-fit.chance_error_cents, fit.chance_error_cents, color=MUTED,
                alpha=0.13, lw=0, label="chance-level error")
    bot.axhline(0, color=INK, lw=0.8)
    bot.vlines(x, 0, res, color=SIGNAL_COLOR, lw=1.6)
    bot.plot(x, res, "o", color=SIGNAL_COLOR, ms=6, mec="white", mew=0.7)
    bot.set_xlabel("cents", fontsize=10, color=INK)
    bot.set_ylabel("residual (c)", fontsize=10, color=INK)
    bot.set_xlim(0, pc)
    bot.legend(fontsize=8, frameon=False, loc="upper right")
    bot.spines[["top", "right"]].set_visible(False)
    return fig, axes


def plot_mos_trajectory(
    trajectory: Sequence,
    *,
    times: Optional[Sequence[float]] = None,
    max_cardinality: int = 18,
    figsize: Tuple[float, float] = (15.0, 6.5),
):
    """A signal's path through the labyrinth over time.

    Left: the labyrinth with the trajectory drawn through it, each window a
    point at (its generator, its cardinality) and consecutive windows joined,
    coloured by time.  Right: the same three coordinates as time series --
    generator, cardinality, and fit error.

    Windows where no scale could be fitted (``None`` entries) break the path
    rather than being interpolated across.

    Parameters
    ----------
    trajectory : sequence of MOSFit or None
        As returned by :func:`~biotuner.mos.derive.mos_trajectory`.
    times : sequence of float, optional
        Window times; window index is used if omitted.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> from biotuner.mos.derive import trajectory_from_windows
    >>> a = MOSScale.from_signature(5, 2, tuning=12).ratios
    >>> b = MOSScale.from_signature(4, 3, tuning=19).ratios
    >>> traj = trajectory_from_windows([a, b, a], max_cardinality=12)
    >>> fig, axes = plot_mos_trajectory(traj)
    >>> plt.close(fig)
    """
    fig = plt.figure(figsize=figsize)
    lab = fig.add_subplot(1, 2, 1, projection="polar")
    plot_labyrinth(max_cardinality, ax=lab, label="cents", annotate=None)
    lab.get_legend().remove()

    idx = [i for i, f in enumerate(trajectory) if f is not None]
    if not idx:
        raise ValueError("trajectory contains no successful fits to plot")
    t = np.asarray(times, float) if times is not None else np.arange(len(trajectory),
                                                                     dtype=float)
    gens = np.array([trajectory[i].scale.generator for i in idx])
    cards = np.array([trajectory[i].scale.cardinality for i in idx])
    errs = np.array([trajectory[i].error_cents for i in idx])
    tt = t[idx]

    # Break the path wherever a window failed, so gaps read as gaps.
    runs, current = [], [0]
    for k in range(1, len(idx)):
        if idx[k] == idx[k - 1] + 1:
            current.append(k)
        else:
            runs.append(current)
            current = [k]
    runs.append(current)
    for run in runs:
        if len(run) > 1:
            lab.plot([_theta(gens[k]) for k in run], [cards[k] for k in run],
                     "-", color=SIGNAL_COLOR, lw=1.4, alpha=0.75, zorder=9)
    sc = lab.scatter([_theta(g) for g in gens], cards, c=tt, cmap="plasma", s=55,
                     edgecolor="white", linewidth=0.7, zorder=10)
    fig.colorbar(sc, ax=lab, pad=0.11, shrink=0.7, label="time")
    lab.set_title("path through the labyrinth", fontsize=12, color=INK, pad=18)

    gs = fig.add_gridspec(3, 2, hspace=0.12)
    axes = [fig.add_subplot(gs[r, 1]) for r in range(3)]
    axes[0].plot(tt, gens * T.PERIOD_CENTS, "-o", color=LARGE_COLOR, ms=4)
    axes[0].set_ylabel("generator (c)", fontsize=9.5, color=INK)
    axes[1].step(tt, cards, where="mid", color=SMALL_COLOR, lw=1.8)
    axes[1].plot(tt, cards, "o", color=SMALL_COLOR, ms=4)
    axes[1].set_ylabel("cardinality", fontsize=9.5, color=INK)
    axes[2].plot(tt, errs, "-o", color=SIGNAL_COLOR, ms=4)
    axes[2].set_ylabel("error (c)", fontsize=9.5, color=INK)
    axes[2].set_xlabel("time", fontsize=10, color=INK)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(color=GRIDC, lw=0.5)
        a.set_axisbelow(True)
    for a in axes[:2]:
        a.tick_params(labelbottom=False)
    labels = [trajectory[i].scale.signature for i in idx]
    for x, y, lb in zip(tt, cards, labels):
        axes[1].annotate(lb, (x, y), textcoords="offset points", xytext=(0, 7),
                         fontsize=6.5, ha="center", color=MUTED)
    return fig, [lab] + axes


def plot_forward_vs_inverse(
    forward,
    inverse=None,
    *,
    max_cardinality: Optional[int] = None,
    top_n: Optional[int] = None,
    max_error_cents: Optional[float] = None,
    annotate: bool = True,
    ax=None,
    figsize: Tuple[float, float] = (13.5, 8.5),
):
    """Both readings of a signal on one labyrinth: observed vs latent generator.

    :func:`~biotuner.mos.derive.fit_mos` searches for the generator that best
    explains the peaks, wherever it lies -- a *latent* parameter, which need not
    be an interval anything in the signal states.
    :func:`~biotuner.mos.derive.forward_scales` runs the other way: it takes an
    interval the signal does state, declares it the generator, and reads off the
    scale that interval builds.  The two answers are usually different, and the
    difference is the finding, so they belong in one frame.

    The labyrinth is faded to scenery.  On top of it, each forward reading is a
    point at its (generator, cardinality) -- ``s`` grows with how many observed
    intervals proposed that generator, colour darkens as the resulting scale
    explains the whole target set better -- and every reading built on the same
    interval hangs off one radial guide, numbered at the rim and keyed in the
    legend.  The inverse fit is a star on its own dashed ray, in a colour absent
    from the forward ramp.

    Only the bright half is drawn.  A generator and its complement build the
    same scale -- ``mos_series`` gives the pair the same signature at the same
    cardinality, and their degrees are reflections of one another, so the fit
    cannot tell them apart -- and both directions therefore fold into
    ``(0.5, 1)``.  The dark half would be permanently empty, and an empty
    half-circle reads as absence of evidence rather than as a convention.
    Every generator quoted in the title and the legend is the folded one, so a
    fit that happens to name the dark spelling of its own scale is captioned
    with the value under the marker rather than its complement.

    Parameters
    ----------
    forward : ForwardScale or sequence of ForwardScale
        As returned by :func:`~biotuner.mos.derive.forward_scales`.  Re-sorted
        into rank order here, so a caller-sliced or hand-assembled list is fine.
    inverse : MOSFit or sequence of MOSFit, optional
        The competing latent-generator fit; the first is drawn when a ranked
        list is passed.  Omitted, it is computed with
        :func:`~biotuner.mos.derive.best_mos` from the very targets the forward
        readings were scored against (``fit.targets``), over the same ring
        range -- which is the only way the comparison is honest, and cheaper to
        get right here than to remember at every call site.
    max_cardinality : int, optional
        Outermost ring.  Defaults to the largest cardinality drawn, so the
        picture is as small as the data allows.
    top_n : int, optional
        Draw only the best ``n`` forward readings.  Worth setting: a single
        interval can generate a well-formed scale at a dozen cardinalities, and
        :func:`~biotuner.mos.derive.forward_scales` returns all of them.
    max_error_cents : float, optional
        Ceiling of the colour ramp.  Defaults to the worst reading drawn, capped
        at 60 cents -- past half a semitone, "explains the signal" has no
        content and letting one hopeless reading set the scale flattens the
        rest.
    annotate : bool, default True
        Number the generators at the rim and key the numbers in the legend.
    ax : matplotlib polar axes, optional
    figsize : tuple

    Returns
    -------
    (fig, ax)

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> from biotuner.mos.derive import forward_scales
    >>> peaks = [10.07, 15.64, 19.31, 22.91]          # S001, eyes closed (Hz)
    >>> readings = forward_scales(peaks, include_ratios=False,
    ...                           min_cardinality=5, max_cardinality=7)
    >>> fig, ax = plot_forward_vs_inverse(readings, top_n=8)
    >>> float(ax.get_thetamin()), float(ax.get_thetamax())    # the bright half
    (180.0, 360.0)
    >>> sum(line.get_marker() == "*" for line in ax.get_lines())   # the inverse
    1
    >>> plt.close(fig)
    """
    from biotuner.mos.derive import ForwardScale, MOSFit, best_mos

    readings = [forward] if isinstance(forward, ForwardScale) else list(forward)
    if not readings:
        raise ValueError(
            "no forward readings to draw; forward_scales() returns an empty "
            "list when the signal states no interval that generates a scale"
        )
    readings = sorted(readings, key=lambda r: r._rank_key)
    if top_n is not None:
        readings = readings[:top_n]
    period = readings[0].scale.period
    pc = T.PERIOD_CENTS * math.log2(period)

    if isinstance(inverse, (list, tuple)):
        if not inverse:
            raise ValueError("inverse is an empty list; pass a MOSFit or None")
        inverse = inverse[0]
    if inverse is None:
        targets = list(readings[0].fit.targets)
        if not targets:
            raise ValueError(
                "cannot derive the inverse fit: these forward readings carry no "
                "targets, so pass the MOSFit to compare against explicitly"
            )
        inverse = best_mos(
            targets, period=period,
            max_cardinality=max(r.scale.cardinality for r in readings),
        )
    elif not isinstance(inverse, MOSFit):
        raise TypeError(
            f"inverse must be a MOSFit, a list of them, or None; got "
            f"{type(inverse).__name__}"
        )

    cards = [r.scale.cardinality for r in readings] + [inverse.scale.cardinality]
    if max_cardinality is None:
        max_cardinality = max(cards)
    max_cardinality = max(int(max_cardinality), 3)

    # --- scenery ----------------------------------------------------------- #
    owns_figure = ax is None
    fig, ax = _polar_axes(ax, figsize)
    if owns_figure:
        # A half-disc is twice as tall as it is wide, and matplotlib centres it
        # in whatever box it is given -- in a square one, most of the width is
        # dead space between the drawing and the legend.
        fig.subplots_adjust(left=0.01, right=0.40, top=0.86, bottom=0.09)
    plot_labyrinth(max_cardinality, period=period, ax=ax, label="cents",
                   annotate=None)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    _dim_to_scenery(ax)
    # Ring labels one per integer collide with themselves near the pole once the
    # rim is far out, and the inner rings are the ones drawn closest together.
    step = 1 if max_cardinality <= 10 else (2 if max_cardinality <= 18 else 3)
    rticks = list(range(step, max_cardinality + 1, step))
    ax.set_rticks(rticks)
    ax.set_yticklabels([str(t) for t in rticks], fontsize=7, color=MUTED)
    rim = max_cardinality + 0.8

    # --- the forward readings, grouped by the interval they came from ------- #
    # Keyed on the *folded* generator throughout: a group, its guide ray, its
    # marker and its legend row must all speak of the same spelling.
    groups: List[Tuple[float, List]] = []
    for r in readings:
        g = _bright(r.scale.generator)
        for known, members in groups:
            if abs(g - known) < 1e-9:
                members.append(r)
                break
        else:
            groups.append((g, [r]))

    tag_r: Dict[int, float] = {}
    last_theta = last_r = None
    for k in sorted(range(len(groups)), key=lambda i: groups[i][0]):
        theta = _theta(groups[k][0])
        r_tag = rim + 0.85
        if last_theta is not None and math.degrees(theta - last_theta) < 7.0:
            r_tag = last_r + 1.3
        tag_r[k] = r_tag
        last_theta, last_r = theta, r_tag

    for k, (g, _members) in enumerate(groups):
        theta = _theta(g)
        ax.plot([theta, theta], [1.0, tag_r[k] - 0.55], color=MUTED, lw=0.8,
                alpha=0.5, zorder=5)
        if annotate:
            ax.text(theta, tag_r[k], _tag_label(k), fontsize=8.5, color=INK,
                    ha="center", va="center", zorder=13,
                    bbox=dict(boxstyle="circle,pad=0.20", fc="white", ec=MUTED,
                              lw=0.7))

    errors = [r.error_cents for r in readings]
    if max_error_cents is None:
        max_error_cents = float(min(60.0, max(5.0, max(errors))))
    sizes = [_source_size(r.n_sources) for r in readings]
    sc = ax.scatter(
        [_theta(_bright(r.scale.generator)) for r in readings],
        [r.scale.cardinality for r in readings],
        s=sizes, c=errors, cmap=_forward_colormap(), vmin=0.0,
        vmax=max_error_cents, edgecolor="white", linewidth=0.7, zorder=10,
    )

    # --- the inverse fit --------------------------------------------------- #
    # An *open* star: when the latent generator turns out to be one the signal
    # states -- the one result worth the whole figure -- the two markers land on
    # the same spot, and a filled star would hide the very coincidence it is
    # there to reveal.
    g_inv = _bright(inverse.scale.generator)
    t_inv = _theta(g_inv)
    ax.plot([t_inv, t_inv], [1.0, rim], color=INVERSE_COLOR, lw=1.8,
            ls=(0, (5, 2.5)), alpha=0.9, zorder=8)
    ax.plot([t_inv], [inverse.scale.cardinality], marker="*", ms=27, ls="none",
            mfc="none", mec=INVERSE_COLOR, mew=2.2, zorder=12)

    # --- keys -------------------------------------------------------------- #
    handles = []
    for k, (g, members) in enumerate(groups):
        best = members[0]
        num, den = best.interval_pair
        handles.append(Line2D(
            [], [], marker="o", ls="none", ms=8, mec="white", mew=0.7,
            color=sc.to_rgba(best.error_cents),
            label=(f"{_tag_label(k)}   {num:.4g}/{den:.4g} = {best.interval:.3f}"
                   f"  →  {g * pc:.1f} c"
                   f"   ·   best {best.signature}, {best.error_cents:.1f} c"),
        ))
    handles.append(Line2D(
        [], [], marker="*", ls="none", ms=17, mfc="none", mec=INVERSE_COLOR,
        mew=1.8,
        label=(f"inverse fit — latent generator\n     {inverse.signature} @ "
               f"{g_inv * pc:.1f} c, {inverse.error_cents:.1f} c"),
    ))
    spread = sorted({r.n_sources for r in readings})
    if len(spread) > 1:
        for n in (spread[0], spread[-1]):
            handles.append(Line2D(
                [], [], marker="o", ls="none", mfc="none", mec=MUTED, mew=1.0,
                ms=math.sqrt(_source_size(n)),
                label=f"{n} observed interval{'s' if n > 1 else ''} proposed it",
            ))
    leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                    fontsize=8.5, frameon=False, labelspacing=0.9,
                    borderaxespad=0.0,
                    title="forward readings — an interval the signal states, "
                          "stacked\nangle = generator (cents)   ·   ring = notes "
                          "in the scale")
    leg.get_title().set_fontsize(9)
    leg.get_title().set_color(INK)
    if hasattr(leg, "set_alignment"):
        leg.set_alignment("left")          # matplotlib >= 3.6
    else:  # pragma: no cover - the project supports back to 3.5.3
        leg._legend_box.align = "left"

    fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.038, pad=0.07,
                 shrink=0.45, extend="max" if max(errors) > max_error_cents
                 else "neither",
                 label="weighted cents error to the whole signal (dark = better)")

    ax.set_rlim(0, max(rim + 1.6, max(tag_r.values()) + 1.0))
    ax.set_thetamin(180.0)
    ax.set_thetamax(360.0)

    best_fwd = readings[0]
    g_fwd = _bright(best_fwd.scale.generator)
    if abs(g_fwd - g_inv) * pc < 1.0:
        verdict = "the latent generator is one the signal states"
    elif best_fwd.score < inverse.score - 1e-9:
        verdict = "the observed interval explains it better"
    else:
        verdict = "the latent generator explains it better"
    ax.set_title(
        f"Two readings of the same signal — {verdict}\n"
        f"forward {best_fwd.signature} @ {g_fwd * pc:.1f} c "
        f"({best_fwd.error_cents:.1f} c)   vs   inverse {inverse.signature} @ "
        f"{g_inv * pc:.1f} c ({inverse.error_cents:.1f} c)",
        fontsize=12, color=INK, pad=22,
    )
    return fig, ax


def plot_ji_landscape(
    *,
    max_cardinality: int = 24,
    targets: Sequence[float] = JI_5_LIMIT,
    period: float = 2.0,
    resolution: int = 1200,
    generator_range: Tuple[float, float] = (0.5, 1.0),
    max_error_cents: float = 50.0,
    ax=None,
    figsize: Tuple[float, float] = (13.0, 6.5),
):
    """Where in the labyrinth the well-formed scales approximate just intonation.

    Milne et al. §1 point at exactly this use: "a scale labyrinth is used to
    indicate MOS scale tunings that provide good approximations of just
    intonation".  Each cell is the mean cents error from a just interval to the
    nearest degree of the MOS at that (generator, cardinality); blank cells are
    generators with no MOS at that cardinality at all, which is most of them.

    Parameters
    ----------
    max_cardinality : int, default 24
    targets : sequence of float, default 5-limit consonances
    resolution : int, default 1200
        Generator samples across ``generator_range``.
    generator_range : (float, float), default (0.5, 1.0)
        The bright half is enough -- the other half mirrors it.
    max_error_cents : float, default 50.0
        Colour-scale ceiling.  Errors are unbounded as the generator approaches
        the period (where every degree piles up at the root), and letting that
        set the scale flattens everything worth looking at.  Fifty cents is
        already half a semitone; past it "approximation" has no content.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> fig, ax = plot_ji_landscape(max_cardinality=12, resolution=200)
    >>> plt.close(fig)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    lo, hi = generator_range
    gs = np.linspace(lo, hi, resolution, endpoint=False)[1:]
    pc = T.PERIOD_CENTS * math.log2(period)
    tpos = np.mod(np.log(np.asarray(targets, float)) / math.log(period), 1.0)

    heat = np.full((max_cardinality + 1, len(gs)), np.nan)
    for col, g in enumerate(gs):
        for card, _, _ in T.mos_series(
            float(g), max_cardinality=max_cardinality, include_trivial=True
        ):
            degrees = np.asarray(T.degrees_from_generator(float(g), card))
            d = np.abs(tpos[:, None] - degrees[None, :])
            d = np.minimum(d, 1.0 - d)
            heat[card, col] = float(d.min(axis=1).mean() * pc)

    # Dark = accurate. A reversed ramp would make the *worst* fits the most
    # visually salient, which is the opposite of what this plot is for.
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#e9e9ec")  # no MOS at this (generator, cardinality) at all
    im = ax.imshow(
        heat, aspect="auto", origin="lower", cmap=cmap,
        extent=(gs[0] * pc, gs[-1] * pc, -0.5, max_cardinality + 0.5),
        interpolation="nearest", vmin=0.0, vmax=max_error_cents,
    )
    fig.colorbar(im, ax=ax, pad=0.02, extend="max",
                 label="mean cents error to nearest degree")
    ax.set_xlabel("generator (cents)", fontsize=10, color=INK)
    ax.set_ylabel("cardinality", fontsize=10, color=INK)
    ax.set_yticks(range(2, max_cardinality + 1, 1 if max_cardinality <= 16 else 2))
    ax.set_ylim(1.5, max_cardinality + 0.5)
    ax.set_title(
        f"Just-intonation approximation across the labyrinth "
        f"({len(targets)} targets)",
        fontsize=12, color=INK,
    )
    return fig, ax


def plot_fit_field(
    field_or_ratios,
    *,
    weights: Optional[Sequence[float]] = None,
    period: float = 2.0,
    max_cardinality: int = 22,
    resolution: int = 720,
    polar: bool = True,
    max_error_cents: float = 30.0,
    show_peaks: bool = True,
    peaks: Optional[Sequence[float]] = None,
    mark: Union[None, float, MOSScale, Sequence] = None,
    ax=None,
    figsize: Tuple[float, float] = (8.6, 8.6),
):
    """Where in the labyrinth a signal lives -- the whole plane, scored.

    :func:`plot_mos_fit` shows one answer; this shows the landscape that answer
    was chosen from. Each cell is the weighted cents error from the signal's
    ratios to the nearest degree of the scale at that (generator, cardinality).
    Grey means no well-formed scale exists there at all, which is most of the
    plane.

    Reading it matters more than admiring it. A signal usually sits in several
    *disconnected* dark patches rather than one, and a single best-fit answer
    cannot say that. :meth:`~biotuner.mos.derive.FitField.islands` counts them.

    Parameters
    ----------
    field_or_ratios : FitField or sequence of float
        A precomputed field, or the ratios to compute one from. Passing the
        field is the way to draw several views without recomputing.
    weights : sequence of float, optional
        Ignored when a field is passed.
    period, max_cardinality, resolution
        Ignored when a field is passed.
    polar : bool, default True
        Polar keeps the labyrinth's own geometry -- angle is the generator,
        radius the cardinality -- so this figure can be read against
        :func:`plot_labyrinth`. Cartesian is easier to read values off.
    max_error_cents : float, default 30.0
        Colour ceiling. Error is unbounded as the generator approaches the
        period, and letting that set the scale flattens everything else.
    show_peaks : bool, default True
        Mark the signal's own ratio positions.  A :class:`FitField` remembers
        the ratios it was built from, so this works whether you pass ratios or
        a precomputed field.
    peaks : sequence of float, optional
        Ratios to mark instead of the field's own.
    mark : float, MOSScale or sequence, optional
        Generator(s) to trace, e.g. the scale ``fit_mos`` settled on.
    ax : matplotlib axes, optional
        Must be polar when ``polar`` is True.
    figsize : tuple

    Returns
    -------
    (fig, ax)

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> fig, ax = plot_fit_field([1.0, 1.125, 1.5], max_cardinality=10,
    ...                          resolution=120)
    >>> ax.name
    'polar'
    >>> plt.close(fig)
    """
    from biotuner.mos.derive import FitField, fit_field, labyrinth_positions

    if isinstance(field_or_ratios, FitField):
        field = field_or_ratios
    else:
        field = fit_field(list(field_or_ratios), weights, period=period,
                          max_cardinality=max_cardinality, resolution=resolution)
    marks = list(peaks) if peaks is not None else list(field.ratios)

    heat = np.ma.masked_invalid(field.errors)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#e9e9ec")
    n_card = field.errors.shape[0] - 1
    rim = n_card + 1.2

    if polar:
        fig, ax = _polar_axes(ax, figsize)
        theta = np.concatenate([field.generators, [1.0]]) * 2.0 * math.pi
        radial = np.arange(-0.5, n_card + 1.5)
        mesh = ax.pcolormesh(theta, radial, heat, cmap=cmap, vmin=0.0,
                             vmax=max_error_cents, shading="flat")
        ax.set_rlim(0, rim)
        ax.set_yticks(range(2, n_card + 1, 2))
        ax.tick_params(labelsize=7, colors=MUTED)
        ax.set_xticks(np.linspace(0, 2 * math.pi, 12, endpoint=False))
        ax.set_xticklabels(
            [f"{field.period_cents * k / 12:.0f}" for k in range(12)], fontsize=8
        )
        ax.grid(color="white", lw=0.3, alpha=0.3)
    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        mesh = ax.pcolormesh(
            np.concatenate([field.generators, [1.0]]) * field.period_cents,
            np.arange(-0.5, n_card + 1.5), heat, cmap=cmap, vmin=0.0,
            vmax=max_error_cents, shading="flat",
        )
        ax.set_xlabel("generator (cents)", fontsize=10, color=INK)
        ax.set_ylabel("cardinality", fontsize=10, color=INK)
        ax.set_ylim(1.5, n_card + 0.5)
        ax.spines[["top", "right"]].set_visible(False)

    if show_peaks and marks:
        for p in labyrinth_positions(marks, field.period):
            angle = 2.0 * math.pi * p if polar else p * field.period_cents
            ax.plot([angle, angle], [0, rim] if polar else [1.5, n_card + 0.5],
                    color="#2ad4c8", lw=1.0, ls="--", alpha=0.9, zorder=5)

    if mark is not None:
        items = mark if isinstance(mark, (list, tuple)) else [mark]
        for i, item in enumerate(items):
            g = item.generator if isinstance(item, MOSScale) else float(item)
            angle = 2.0 * math.pi * g if polar else g * field.period_cents
            colour = HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)]
            ax.plot([angle, angle], [0, rim] if polar else [1.5, n_card + 0.5],
                    color=colour, lw=1.8, alpha=0.95, zorder=6)

    fig.colorbar(mesh, ax=ax, pad=0.10 if polar else 0.02, shrink=0.72,
                 extend="max", label="weighted cents error to this signal")
    ax.set_title(
        f"Where this signal lives — {field.n_targets} ratios scored against "
        f"every well-formed scale",
        fontsize=11.5, color=INK, pad=18 if polar else 12,
    )
    return fig, ax


def plot_play_state(
    state,
    scale=None,
    *,
    ax=None,
    figsize: Tuple[float, float] = (12.0, 5.5),
):
    """A Fourier Scratching play state and its spectrum (Milne et al. Fig. 9).

    Left: the ``n`` fingers on the circular keyboard, magnitude as radius and
    phase as position, joined into the polygon that is struck in order.  When a
    scale is supplied its keys are drawn underneath, widths proportional to the
    step above each tone.  Right: the DFT the performer actually manipulates.

    Milne et al. render this on a pair of Riemann spheres; the plane is enough
    to read the same information.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> from biotuner.mos.fourier import partial      # doctest: +SKIP
    >>> fig, axes = plot_play_state(partial(7, 1))    # doctest: +SKIP
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        left = fig.add_subplot(1, 2, 1, projection="polar")
        right = fig.add_subplot(1, 2, 2)
        axes = [left, right]
    else:
        axes = list(np.atleast_1d(ax))
        fig = axes[0].figure
        left, right = axes[0], axes[-1]

    left.set_theta_zero_location("N")
    left.set_theta_direction(-1)

    if scale is not None:
        base = _as_scale(scale)
        degrees = list(scale.degrees)
        for i, d in enumerate(degrees):
            nxt = degrees[i + 1] if i + 1 < len(degrees) else 1.0
            left.bar(x=_theta((d + nxt) / 2.0), height=1.0, width=_theta(nxt - d),
                     bottom=0.0,
                     color=LARGE_COLOR if base.word[i] == "L" else SMALL_COLOR,
                     edgecolor="white", linewidth=1.0, alpha=0.25)

    phases = np.asarray(state.phases, float)
    mags = np.asarray(state.magnitudes, float)
    scale_r = mags / (mags.max() or 1.0)
    left.plot(np.append(phases, phases[:1]), np.append(scale_r, scale_r[:1]),
              "-o", color=SIGNAL_COLOR, lw=1.4, ms=8, mec="white", mew=0.8, zorder=5)
    for k, (p, r) in enumerate(zip(phases, scale_r)):
        left.annotate(str(k), (p, r), textcoords="offset points", xytext=(6, 6),
                      fontsize=8, color=INK)
    left.set_rlim(0, 1.25)
    left.set_yticks([])
    left.set_xticks([])
    left.grid(False)
    left.set_title(f"play state, n = {state.n}", fontsize=11, color=INK, pad=14)

    spec = np.asarray(state.spectrum)
    ks = np.arange(len(spec))
    right.bar(ks, np.abs(spec), color=LARGE_COLOR, alpha=0.85, width=0.7)
    right.set_xticks(ks)
    right.set_xlabel("Fourier coefficient k", fontsize=10, color=INK)
    right.set_ylabel("|a_k|", fontsize=10, color=INK)
    right.spines[["top", "right"]].set_visible(False)
    right.set_title("spectrum — what scratching manipulates", fontsize=11, color=INK)
    return fig, axes
