"""Interactive labyrinths: a hover-rich Plotly figure and ipywidgets explorers.

Milne et al. (2011) built the labyrinth as a GUI object -- §1: "the scale
labyrinth allows a musician to choose, simultaneously, a scale structure
(number of small and large steps) and its tuning (the sizes of its period and
generator)".  A static image cannot do that.  Two complementary surfaces here:

:func:`labyrinth_plotly`
    Every arc carries its own hover card -- signature, valid range, landmark
    EDOs, coherence, embedding.  Zoomable, exportable to a standalone HTML
    file, and needs nothing running.

:func:`mos_explorer`
    The real instrument.  Drag the generator and the whole scale universe
    responds: the family recomputes, the wheel redraws, the summary updates,
    and the scale can be played.  Bind it to a
    :class:`~biotuner.biotuner_object.compute_biotuner` and the signal's own
    peaks ride along on the rim.

Both dependencies are optional -- ``pip install biotuner[interactive]``.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from biotuner.mos import theory as T
from biotuner.mos.scale import MOSScale, mos_family

__all__ = [
    "morph_explorer",
    "labyrinth_plotly",
    "mos_explorer",
    "fit_explorer",
    "scratch_explorer",
    "web_explorer",
    "simplex_explorer",
    "trajectory_explorer",
    "dissonance_explorer",
    "matrix_explorer",
]


def _require(module: str, extra: str = "interactive"):
    """Import an optional dependency, or explain how to get it."""
    try:
        return __import__(module, fromlist=["_"])
    except ImportError as exc:
        raise ImportError(
            f"the '{module}' package is required for this functionality. "
            f"Install it with:\n\n    pip install biotuner[{extra}]\n\n"
            f"or directly:\n\n    pip install {module}\n"
        ) from exc


# --------------------------------------------------------------------------- #
# Plotly
# --------------------------------------------------------------------------- #
def _arc_hover(node: T.SBNode, lower: bool, period_cents: float) -> str:
    """Hover card for one MOS sub-arc."""
    b, d = node.left.denominator, node.right.denominator
    n_large, n_small = (b, d) if lower else (d, b)
    lo = node.left if lower else node.node
    hi = node.node if lower else node.right
    lm = T.mos_landmarks(n_large, n_small, bright=float(lo + hi) > 1)
    emb_n, emb_t = T.embedding(n_large, n_small, bright=lm.bright)
    c_lo, c_hi = T.coherence_range(n_large, n_small, bright=lm.bright)
    return (
        f"<b>{n_large}L{n_small}s</b> — {n_large + n_small} notes<br>"
        f"valid {lo} … {hi} "
        f"({float(lo) * period_cents:.1f} … {float(hi) * period_cents:.1f} ¢)<br>"
        f"coherent {c_lo} … {c_hi} "
        f"({float(c_lo) * period_cents:.1f} … {float(c_hi) * period_cents:.1f} ¢)<br>"
        f"equalized at {lm.equalized} = {lm.equalized_edo}-EDO<br>"
        f"s→0 at {lm.small_vanishes} = {lm.small_vanishes_edo}-EDO<br>"
        f"inverse {n_small}L{n_large}s · embedded in {emb_n} notes at {emb_t}"
    )


def labyrinth_plotly(
    max_cardinality: int = 18,
    *,
    period: float = 2.0,
    peaks: Optional[Sequence[float]] = None,
    peak_weights: Optional[Sequence[float]] = None,
    highlight: Union[None, float, MOSScale, Sequence] = None,
    temperaments: bool = False,
    show_spokes: bool = True,
    generator_slider: bool = False,
    slider_steps: int = 121,
    width: int = 850,
    height: int = 850,
):
    """The labyrinth as a Plotly figure, every arc self-describing on hover.

    Parameters
    ----------
    max_cardinality : int, default 18
    period : float, default 2.0
    peaks : sequence of float, optional
        Biosignal peak ratios, drawn on the rim.
    peak_weights : sequence of float, optional
        Marker sizes, e.g. amplitudes.
    highlight : float, MOSScale, or sequence, optional
        Generator fraction(s) or scale(s) to trace with a radial line.
    temperaments : bool, default False
        Overlay named rank-2 temperaments at their optimal generators.
    show_spokes : bool, default True
    generator_slider : bool, default False
        Add a slider that sweeps a highlighted generator around the labyrinth,
        marking its whole MOS family as it goes.  This is what makes the
        exported HTML an *instrument* rather than a picture: unlike
        :func:`mos_explorer` it needs no Python running behind it, because the
        frames are precomputed, so the file works anywhere a browser does.
    slider_steps : int, default 121
        Generator positions on the slider.  Every step is a precomputed frame,
        so this is the main lever on file size.
    width, height : int

    Returns
    -------
    plotly.graph_objects.Figure
        ``fig.write_html('labyrinth.html')`` gives a self-contained page.

    Examples
    --------
    >>> fig = labyrinth_plotly(9)                     # doctest: +SKIP
    >>> fig.write_html('labyrinth.html')              # doctest: +SKIP
    """
    go = _require("plotly.graph_objects")
    if max_cardinality < 2:
        raise ValueError(f"max_cardinality must be at least 2, got {max_cardinality}")

    period_cents = T.PERIOD_CENTS * math.log2(period)
    nodes = T.sb_tree_nodes(max_cardinality)
    rim = max_cardinality + 0.8
    fig = go.Figure()

    def deg(fraction) -> float:
        return 360.0 * float(fraction)

    # Spokes first, so arcs sit on top.
    if show_spokes:
        r, th = [], []
        for node in nodes:
            r += [rim, node.cardinality, None]
            th += [deg(node.node), deg(node.node), None]
        fig.add_trace(
            go.Scatterpolar(
                r=r, theta=th, mode="lines", hoverinfo="skip",
                line=dict(color="rgba(120,124,132,0.40)", width=1),
                name="equal temperaments", showlegend=True,
            )
        )

    # One trace per sub-arc so each carries its own hover card.
    for node in nodes:
        card = node.cardinality
        shade = card / max_cardinality
        colour = f"rgba({int(40 + 120 * shade)},{int(120 - 40 * shade)},{int(160 - 60 * shade)},0.85)"
        for lower in (True, False):
            lo = node.left if lower else node.node
            hi = node.node if lower else node.right
            n = max(8, int(120 * float(hi - lo)))
            th = np.linspace(deg(lo), deg(hi), n)
            c_lo = T.mediant(node.left, node.node)
            c_hi = T.mediant(node.node, node.right)
            coherent = (c_lo, node.node) if lower else (node.node, c_hi)
            text = _arc_hover(node, lower, period_cents)
            fig.add_trace(
                go.Scatterpolar(
                    r=np.full(n, card), theta=th, mode="lines",
                    line=dict(color=colour, width=3),
                    hovertemplate=text + "<extra></extra>",
                    showlegend=False,
                )
            )
            th_c = np.linspace(deg(coherent[0]), deg(coherent[1]), 24)
            fig.add_trace(
                go.Scatterpolar(
                    r=np.full(24, card), theta=th_c, mode="lines",
                    line=dict(color=colour, width=7),
                    hovertemplate=text + "<br><i>coherent here</i><extra></extra>",
                    showlegend=False,
                )
            )

    if temperaments:
        _plotly_temperaments(fig, go, rim, period_cents)

    if highlight is not None:
        items = highlight if isinstance(highlight, (list, tuple)) else [highlight]
        palette = ["#1b3a6b", "#06A77D", "#7B3FA0", "#B8860B", "#2F4F4F"]
        for i, item in enumerate(items):
            g = item.generator if isinstance(item, MOSScale) else float(item)
            per = item.period if isinstance(item, MOSScale) else period
            colour = palette[i % len(palette)]
            fam = mos_family(T.fraction_to_generator(g, per), max_cardinality, per)
            fig.add_trace(
                go.Scatterpolar(
                    r=[0, rim], theta=[deg(g), deg(g)], mode="lines",
                    line=dict(color=colour, width=2.5),
                    name=f"g = {g * period_cents:.1f} ¢", showlegend=True,
                    hovertemplate=f"generator {g * period_cents:.2f} ¢<extra></extra>",
                )
            )
            if fam:
                fig.add_trace(
                    go.Scatterpolar(
                        r=[s.cardinality for s in fam],
                        theta=[deg(g)] * len(fam), mode="markers",
                        marker=dict(color=colour, size=9,
                                    line=dict(color="white", width=1)),
                        text=[s.signature for s in fam],
                        hovertemplate="%{text} — %{r} notes<extra></extra>",
                        showlegend=False,
                    )
                )

    if peaks is not None:
        pos = np.mod(np.log(np.asarray(peaks, float)) / math.log(period), 1.0)
        if peak_weights is None:
            sizes = np.full(len(pos), 11.0)
        else:
            w = np.asarray(peak_weights, float)
            sizes = 6.0 + 16.0 * (w / (w.max() or 1.0))
        fig.add_trace(
            go.Scatterpolar(
                r=np.full(len(pos), rim), theta=[deg(p) for p in pos],
                mode="markers",
                marker=dict(color="#C73E1D", size=sizes,
                            line=dict(color="white", width=1.2)),
                text=[f"{float(r):.5f}" for r in peaks],
                hovertemplate="peak ratio %{text}<br>%{theta:.1f}°<extra></extra>",
                name="signal peaks",
            )
        )

    if generator_slider:
        _add_generator_slider(fig, go, max_cardinality, period, period_cents,
                              rim, slider_steps)

    fig.update_layout(
        width=width, height=height,
        title=f"Scale labyrinth — MOS scales to {max_cardinality} notes",
        polar=dict(
            angularaxis=dict(
                direction="clockwise", rotation=90,
                tickmode="array",
                tickvals=[360 * k / 12 for k in range(12)],
                ticktext=[f"{period_cents * k / 12:.0f}" for k in range(12)],
                gridcolor="rgba(0,0,0,0.06)",
            ),
            radialaxis=dict(
                range=[0, rim + 0.6], tickmode="array",
                tickvals=list(range(1, max_cardinality + 1)),
                angle=0, tickangle=0, gridcolor="rgba(0,0,0,0.06)",
            ),
        ),
        hoverlabel=dict(align="left"),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
    )
    return fig


def _add_generator_slider(fig, go, max_cardinality, period, period_cents, rim,
                          steps):
    """Precompute one frame per generator position and bind a slider to them.

    Only the last trace is swapped per frame; the labyrinth itself is drawn
    once and reused, which keeps the exported file to the size of the frames
    rather than a multiple of the whole figure.
    """
    if steps < 2:
        raise ValueError(f"slider_steps must be at least 2, got {steps}")

    grid = [(k + 1) / (steps + 1) for k in range(steps)]

    def highlight(g):
        """Radial line plus a marker on every ring the generator reaches."""
        fam = mos_family(T.fraction_to_generator(g, period), max_cardinality,
                         period)
        theta = 360.0 * g
        r = [0.0, rim, None] + [float(sc.cardinality) for sc in fam]
        th = [theta, theta, None] + [theta] * len(fam)
        text = [""] * 3 + [
            f"{sc.signature} — {sc.cardinality} notes, "
            f"R = {sc.hardness:.2f}, {'proper' if sc.is_proper else 'improper'}"
            for sc in fam
        ]
        return r, th, text

    r0, th0, text0 = highlight(grid[len(grid) // 2])
    fig.add_trace(
        go.Scatterpolar(
            r=r0, theta=th0, mode="lines+markers",
            line=dict(color="#C73E1D", width=2.5),
            marker=dict(color="#C73E1D", size=9,
                        line=dict(color="white", width=1)),
            text=text0, hovertemplate="%{text}<extra></extra>",
            name="generator", showlegend=True,
        )
    )
    idx = len(fig.data) - 1

    frames, marks = [], []
    for k, g in enumerate(grid):
        r, th, text = highlight(g)
        frames.append(
            go.Frame(
                name=str(k), traces=[idx],
                data=[go.Scatterpolar(r=r, theta=th, mode="lines+markers",
                                      text=text)],
            )
        )
        marks.append(
            dict(
                method="animate", label=f"{g * period_cents:.0f}",
                args=[[str(k)], dict(mode="immediate", frame=dict(duration=0,
                                                                  redraw=True),
                                     transition=dict(duration=0))],
            )
        )

    fig.frames = frames
    fig.update_layout(
        sliders=[dict(
            active=len(grid) // 2,
            currentvalue=dict(prefix="generator  ", suffix=" \u00a2",
                              font=dict(size=15)),
            pad=dict(t=40, b=10), len=0.86, x=0.07, y=0.02,
            steps=marks,
            # Every tenth tick, or the labels collide into a grey smear.
            ticklen=4,
        )],
    )
    for i, st in enumerate(fig.layout.sliders[0].steps):
        if i % 10:
            st.label = ""


def _plotly_temperaments(fig, go, rim: float, period_cents: float) -> None:
    try:
        from biotuner.mos.temperaments import all_temperaments
    except Exception:  # pragma: no cover - optional overlay
        return
    for name, temp in all_temperaments().items():
        if getattr(temp, "periods_per_octave", 1) != 1:
            continue
        g = (temp.pote_generator_cents / T.PERIOD_CENTS) % 1.0
        fig.add_trace(
            go.Scatterpolar(
                r=[0, rim], theta=[360 * g, 360 * g], mode="lines",
                line=dict(color="rgba(162,59,114,0.75)", width=1.2),
                hovertemplate=(
                    f"<b>{name}</b><br>comma {temp.comma}<br>"
                    f"generator {temp.pote_generator_cents:.2f} ¢ (POTE)<br>"
                    f"max prime error {temp.max_error:.2f} ¢<extra></extra>"
                ),
                showlegend=False,
            )
        )


# --------------------------------------------------------------------------- #
# ipywidgets explorers
# --------------------------------------------------------------------------- #
def _default_cardinality(g, options, period):
    """Which member of a generator's family to open on, absent a fitted one.

    The largest is the obvious choice and a poor one: a family runs off into
    increasingly lopsided scales, so its biggest member is usually its least
    representative.  Stacking fifths out to 17 notes gives 12L5s with L/s near
    4, which is nobody's idea of the fifth's scale.  Prefer the largest
    *proper* member -- for the fifth that is the 12-note chromatic -- and fall
    back to the largest only when nothing in the family is coherent.
    """
    if not options:
        return None
    for card in reversed(options):
        try:
            if MOSScale.from_fraction(g, card, period).is_proper:
                return card
        except ValueError:
            continue
    return options[-1]


def mos_explorer(
    bt=None,
    ratios: Optional[Sequence[float]] = None,
    *,
    weights: Optional[Sequence[float]] = None,
    max_cardinality: int = 18,
    source: str = "peaks_ratios",
    n_generators: int = 5,
    generators: Optional[Sequence[float]] = None,
):
    """Drag the generators; watch the whole scale universe respond.

    The interface Milne et al. §1 describe: pick a structure and a tuning at
    once.  A generator slider moves around the labyrinth, the cardinality
    dropdown offers only the note-counts that generator actually admits, and
    everything downstream -- summary, wheel, family, fit against your signal --
    recomputes live.

    Several generators can be active at once, each with its own toggle, and
    where two of them land on a coinciding tone the explorer marks it.  Both
    affordances come from the widget this replaces (``vizs.MOS_interactive``);
    the common-tone lines were the most useful thing about it, since shared
    tones are what let one tuning modulate into another.  The first *active*
    generator is the focused one: the wheel, the summary and the cardinality
    dropdown follow it.

    Parameters
    ----------
    bt : compute_biotuner, optional
        If given, its ratios are overlaid on the rim and used to seed the first
        generator at the best-fitting one.
    ratios : sequence of float, optional
        Ratios to overlay, when you do not have a biotuner object.
    weights : sequence of float, optional
        Per-ratio weights, e.g. peak amplitudes.
    max_cardinality : int, default 18
        Initial ring count.
    source : str, default 'peaks_ratios'
        Which tuning to pull from ``bt``.
    n_generators : int, default 5
        How many generator rows to offer.
    generators : sequence of float, optional
        Starting generator *ratios*, e.g. ``[3/2, 5/4]``.  Without these the
        first row is seeded from the best fit to the data, or the perfect
        fifth, and the remaining rows from the next-best fits.

    Returns
    -------
    ipywidgets.Widget
        Display it in a notebook cell.

    Examples
    --------
    >>> ui = mos_explorer(ratios=[1, 1.125, 1.25, 1.5])   # doctest: +SKIP
    >>> ui                                                # doctest: +SKIP
    """
    widgets = _require("ipywidgets")
    import matplotlib.pyplot as plt

    from biotuner.mos import plotting as P

    if n_generators < 1:
        raise ValueError(f"n_generators must be at least 1, got {n_generators}")

    if bt is not None and ratios is None:
        ratios = list(bt.get_tuning(source))
        amps = getattr(bt, "amps", None)
        if weights is None and amps is not None and len(amps) == len(ratios):
            weights = list(amps)

    # --- seed the generator rows ------------------------------------------ #
    seeds: List[float] = []
    seed_card: Optional[int] = None
    if generators is not None:
        seeds = [T.generator_fraction(float(g)) for g in generators]
    elif ratios:
        try:
            from biotuner.mos.derive import fit_mos

            fits = fit_mos(ratios, weights=weights,
                           max_cardinality=max_cardinality, top_n=n_generators)
            seeds = [f.scale.generator for f in fits]
            # Open on the scale that was actually fitted.  Defaulting to the
            # largest cardinality in the family would show a 17-note scale
            # where the fit found a pentatonic, which reads as a wrong answer.
            if fits:
                seed_card = fits[0].scale.cardinality
        except Exception:
            seeds = []
    if not seeds:
        seeds = [math.log2(1.5)]
    while len(seeds) < n_generators:
        seeds.append(seeds[-1])
    seeds = seeds[:n_generators]

    rows, gens, toggles = [], [], []
    for i, g0 in enumerate(seeds):
        tog = widgets.ToggleButton(
            value=(i == 0), description=f"gen {i + 1}",
            button_style="info" if i == 0 else "",
            layout=widgets.Layout(width="92px"),
        )
        sld = widgets.FloatSlider(
            value=g0 * T.PERIOD_CENTS, min=1.0, max=1199.0, step=0.1,
            description="", readout_format=".1f", continuous_update=False,
            layout=widgets.Layout(width="72%"),
        )
        toggles.append(tog)
        gens.append(sld)
        rows.append(widgets.HBox([tog, sld]))

    style = {"description_width": "112px"}
    wide = widgets.Layout(width="95%")
    period = widgets.FloatSlider(value=2.0, min=1.5, max=3.0, step=0.001,
                                 description="period ratio", readout_format=".3f",
                                 continuous_update=False, layout=wide, style=style)
    rings = widgets.IntSlider(value=max_cardinality, min=5, max=40, step=1,
                              description="max rings", continuous_update=False,
                              layout=wide, style=style)
    card = widgets.Dropdown(options=[], description="cardinality",
                            layout=wide, style=style)
    overlays = widgets.SelectMultiple(
        options=["signal peaks", "temperaments", "coherence", "common tones"],
        value=(["signal peaks"] if ratios else []) + ["coherence", "common tones"],
        description="overlays", rows=4, layout=wide, style=style,
    )
    tol = widgets.FloatSlider(value=5.0, min=0.5, max=25.0, step=0.5,
                              description="common tol ¢", continuous_update=False,
                              layout=wide, style=style)
    play = widgets.Button(description="▶ play scale", button_style="success",
                          layout=widgets.Layout(width="46%"))
    export = widgets.Button(description="copy .scl to output",
                            layout=widgets.Layout(width="46%"))
    out, text = widgets.Output(), widgets.Output()
    state: Dict[str, object] = {"scale": None, "seed_card": seed_card}

    def active() -> List[float]:
        return [
            (s.value / T.PERIOD_CENTS) % 1.0
            for s, t in zip(gens, toggles)
            if t.value and 0.0 < (s.value / T.PERIOD_CENTS) % 1.0 < 1.0
        ]

    common_tones = T.common_tones

    def sync_cardinalities(*_):
        gs = active()
        if not gs:
            card.options = []
            return
        opts = [c for c in T.mos_cardinalities(gs[0], rings.value,
                                               include_trivial=True) if c >= 3]
        fallback = state.pop("seed_card", None) or _default_cardinality(
            gs[0], opts, period.value
        )
        keep = card.value if card.value in opts else fallback
        card.options = opts
        if keep in opts:
            card.value = keep
        elif opts:
            card.value = opts[-1]

    def redraw(*_):
        for t in toggles:
            t.button_style = "info" if t.value else ""
        gs = active()
        if not gs or card.value is None:
            with out:
                out.clear_output(wait=True)
            with text:
                text.clear_output(wait=True)
                print("no active generator -- switch one on")
            return
        try:
            focus = MOSScale.from_fraction(gs[0], int(card.value), period.value)
        except ValueError as exc:
            with text:
                text.clear_output(wait=True)
                print(exc)
            return
        state["scale"] = focus

        with out:
            out.clear_output(wait=True)
            fig = plt.figure(figsize=(13.5, 6.0))
            lab = fig.add_subplot(1, 2, 1, projection="polar")
            P.plot_labyrinth(
                rings.value, ax=lab, period=period.value, highlight=list(gs),
                peaks=ratios if "signal peaks" in overlays.value else None,
                peak_weights=weights if "signal peaks" in overlays.value else None,
                temperaments="temperaments" in overlays.value,
                show_coherence="coherence" in overlays.value,
            )
            legend = lab.get_legend()
            if legend is not None:
                legend.remove()
            if "common tones" in overlays.value:
                for s in common_tones(gs, period.value, rings.value, tol.value):
                    lab.plot([2 * math.pi * s, 2 * math.pi * s],
                             [0, rings.value + 0.8], color="#22252b", lw=0.9,
                             ls=":", alpha=0.75, zorder=8)
            wheel = fig.add_subplot(1, 2, 2, projection="polar")
            P.plot_scale_wheel(focus, ax=wheel)
            plt.show()

        with text:
            text.clear_output(wait=True)
            print(focus.summary())
            if len(gs) > 1:
                print("\n  active generators")
                for k, g in enumerate(gs):
                    fam = [c for c in T.mos_cardinalities(g, rings.value,
                                                          include_trivial=True)
                           if c >= 3]
                    print(f"    {k + 1}. {g * T.PERIOD_CENTS:8.2f} c   MOS at {fam}")
                shared = common_tones(gs, period.value, rings.value, tol.value)
                listing = ", ".join(f"{s * T.PERIOD_CENTS:.1f}" for s in shared[:10])
                print(f"  common tones   {len(shared)} within {tol.value:.1f} c"
                      + (f": {listing}" if shared else ""))
            if ratios:
                from biotuner.mos.derive import _as_positions, _clean_weights, _evaluate

                pos = _as_positions(ratios, period.value)
                w = _clean_weights(weights, len(pos))
                fit = _evaluate(focus, pos, w, 15.0, 0.0)
                print(f"\n  vs signal      weighted error {fit.error_cents:.2f} c, "
                      f"{fit.improvement:.1f}x better than chance, "
                      f"coverage {fit.coverage:.0%}")

    def on_play(_):
        scale = state.get("scale")
        if scale is None:
            return
        with text:
            try:
                from biotuner.biotuner_utils import listen_scale

                listen_scale(list(scale.ratios), fund=250, length=400)
            except Exception as exc:
                print(f"playback unavailable: {exc.__class__.__name__}: {exc}")

    def on_export(_):
        scale = state.get("scale")
        if scale is None:
            return
        with text:
            text.clear_output(wait=True)
            print(scale.to_scala(write=False))

    for w in gens + toggles + [period, rings]:
        w.observe(sync_cardinalities, names="value")
    for w in gens + toggles + [period, rings, card, overlays, tol]:
        w.observe(redraw, names="value")
    play.on_click(on_play)
    export.on_click(on_export)

    sync_cardinalities()
    redraw()
    controls = widgets.VBox(
        rows + [period, rings, card, overlays, tol, widgets.HBox([play, export])]
    )
    return widgets.VBox([controls, out, text])


def fit_explorer(
    ratios: Sequence[float],
    *,
    weights: Optional[Sequence[float]] = None,
    top_n: int = 8,
    **fit_kwargs,
):
    """Step through the ranked MOS fits for a set of ratios.

    Shows each candidate's degrees against the targets, its residuals, and
    where it sits in the labyrinth -- so a close second can be inspected rather
    than taken on trust.

    Examples
    --------
    >>> fit_explorer([1, 1.125, 1.25, 1.5])   # doctest: +SKIP
    """
    widgets = _require("ipywidgets")
    import matplotlib.pyplot as plt

    from biotuner.mos import plotting as P
    from biotuner.mos.derive import explain_fit, fit_mos

    fits = fit_mos(ratios, weights=weights, top_n=top_n, **fit_kwargs)
    if not fits:
        raise ValueError("no MOS scale could be fitted to these ratios")

    pick = widgets.Dropdown(
        options=[
            (f"{i + 1}. {f.signature} @ {f.scale.generator_cents:.1f} ¢ "
             f"(err {f.error_cents:.2f} ¢)", i)
            for i, f in enumerate(fits)
        ],
        value=0, description="fit", layout=widgets.Layout(width="70%"),
    )
    out, text = widgets.Output(), widgets.Output()

    def redraw(*_):
        fit = fits[pick.value]
        with out:
            out.clear_output(wait=True)
            P.plot_mos_fit(fit, ratios, weights=weights)
            plt.show()
            P.plot_labyrinth(24, highlight=fit.scale, peaks=ratios,
                             peak_weights=weights)
            plt.show()
        with text:
            text.clear_output(wait=True)
            print(explain_fit(fit, ratios))

    pick.observe(redraw, names="value")
    redraw()
    return widgets.VBox([pick, out, text])


def scratch_explorer(scale, *, n_fingers: Optional[int] = None):
    """Fourier Scratching, live (Milne et al. §5).

    A slider per Fourier coefficient.  Moving one reshapes the whole play state
    at once -- which is the point of the technique: "the Fourier Scratching
    technique offers the ability to change the play states globally and
    smoothly using only a few parameters".

    Parameters
    ----------
    scale : MOSScale or Mode
        The keyboard the fingers strike.
    n_fingers : int, optional
        Defaults to the scale's cardinality, the case Milne et al. Fig. 8
        illustrates.

    Examples
    --------
    >>> scratch_explorer(MOSScale.from_signature(5, 2, tuning=12))  # doctest: +SKIP
    """
    widgets = _require("ipywidgets")
    import matplotlib.pyplot as plt

    from biotuner.mos import plotting as P
    from biotuner.mos.fourier import partial, to_events

    base = getattr(scale, "scale", scale)
    n = int(n_fingers or base.cardinality)
    if n < 1:
        raise ValueError(f"n_fingers must be at least 1, got {n}")

    start = partial(n, 1)
    mags = [
        widgets.FloatSlider(
            value=float(abs(start.spectrum[k])), min=0.0, max=1.5, step=0.01,
            description=f"|a{k}|", layout=widgets.Layout(width="46%"),
            continuous_update=False,
        )
        for k in range(n)
    ]
    phases = [
        widgets.FloatSlider(
            value=float(np.angle(start.spectrum[k])), min=-math.pi, max=math.pi,
            step=0.02, description=f"∠a{k}", layout=widgets.Layout(width="46%"),
            continuous_update=False,
        )
        for k in range(n)
    ]
    out, text = widgets.Output(), widgets.Output()

    def redraw(*_):
        from biotuner.mos.fourier import PlayState

        spec = np.array(
            [m.value * np.exp(1j * p.value) for m, p in zip(mags, phases)],
            dtype=np.complex128,
        )
        state = PlayState.from_spectrum(spec)
        with out:
            out.clear_output(wait=True)
            P.plot_play_state(state, scale)
            plt.show()
        with text:
            text.clear_output(wait=True)
            events = to_events(state, scale)
            print("finger  degree     cents   loudness")
            for e in events:
                print(
                    f"{e.index:6d}  {e.degree:6d}  {e.cents:8.2f}  {e.loudness:8.3f}"
                )

    for w in mags + phases:
        w.observe(redraw, names="value")
    redraw()
    rows = [widgets.HBox([mags[k], phases[k]]) for k in range(n)]
    return widgets.VBox(rows + [out, text])


def web_explorer(
    generator: float = 1.5,
    *,
    cardinality: Optional[int] = None,
    period: float = 2.0,
    max_cardinality: int = 24,
):
    """Turn a scale into a figure, live.

    The design counterpart to :func:`mos_explorer`. Drag the generator and the
    star polygon reforms; step through the modes and the silhouette rotates;
    switch style and the same scale is redrawn as its generator chain, its step
    ring, its interval web or its whole nested family.

    Each style encodes something rather than decorating: see
    :mod:`biotuner.mos.design`. The ``chain`` style in particular draws the
    star polygon ``{N/k}`` whose density is Carey's ``WF(N, g)``, so the figure
    is the scale's generator structure rather than a picture of it.

    Parameters
    ----------
    generator : float, default 1.5
        Starting generator, as a frequency ratio.
    cardinality : int, optional
        Starting note count; defaults to the largest proper member of the
        generator's family.
    period : float, default 2.0
    max_cardinality : int, default 24

    Returns
    -------
    ipywidgets.Widget

    Examples
    --------
    >>> web_explorer(3 / 2)      # doctest: +SKIP
    """
    widgets = _require("ipywidgets")
    import matplotlib.pyplot as plt

    from biotuner.mos import design as D

    style = {"description_width": "104px"}
    wide = widgets.Layout(width="94%")

    gen = widgets.FloatSlider(
        value=T.generator_fraction(generator, period) * T.PERIOD_CENTS,
        min=1.0, max=1199.0, step=0.1, description="generator ¢",
        readout_format=".1f", continuous_update=False, layout=wide, style=style,
    )
    card = widgets.Dropdown(options=[], description="cardinality",
                            layout=wide, style=style)
    style_pick = widgets.ToggleButtons(
        options=list(D.STYLES), value="chain", description="style",
        style={"description_width": "104px"},
    )
    mode = widgets.IntSlider(value=0, min=0, max=11, step=1, description="mode",
                             continuous_update=False, layout=wide, style=style)
    palette = widgets.Dropdown(options=list(D.PALETTES), value="noir",
                               description="palette", layout=wide, style=style)
    thresh = widgets.FloatSlider(
        value=0.0, min=0.0, max=1.0, step=0.02, description="min harmonicity",
        continuous_update=False, layout=wide, style=style,
    )
    points = widgets.Checkbox(value=True, description="show tones")
    out, text = widgets.Output(), widgets.Output()
    state: Dict[str, object] = {"card": cardinality}

    def options():
        g = (gen.value / T.PERIOD_CENTS) % 1.0
        if not 0.0 < g < 1.0:
            return []
        return [c for c in T.mos_cardinalities(g, max_cardinality,
                                               include_trivial=True) if c >= 3]

    def sync(*_):
        opts = options()
        if not opts:
            card.options = []
            return
        want = state.pop("card", None)
        keep = want if want in opts else (
            card.value if card.value in opts
            else _default_cardinality((gen.value / T.PERIOD_CENTS) % 1.0,
                                      opts, period)
        )
        card.options = opts
        card.value = keep if keep in opts else opts[-1]
        mode.max = max(0, int(card.value) - 1)
        if mode.value > mode.max:
            mode.value = mode.max

    def redraw(*_):
        g = (gen.value / T.PERIOD_CENTS) % 1.0
        if not 0.0 < g < 1.0 or card.value is None:
            return
        try:
            scale = MOSScale.from_fraction(g, int(card.value), period)
        except ValueError as exc:
            with text:
                text.clear_output(wait=True)
                print(exc)
            return
        with out:
            out.clear_output(wait=True)
            fig, ax = D.plot_scale_web(
                scale, style_pick.value, mode=mode.value,
                palette=palette.value, max_cardinality=max_cardinality,
                min_harmonicity=thresh.value, show_points=points.value,
                figsize=(7.2, 7.2),
            )
            plt.show()
        with text:
            text.clear_output(wait=True)
            m = scale.mode(mode.value)
            print(f"{scale.signature}   {m.name}   {m.word}")
            print(f"  generator   {scale.generator_cents:8.2f} c")
            print(f"  star        {{{scale.cardinality}/"
                  f"{D.star_density(scale)}}}   (Carey's WF number)")
            print(f"  steps       L = {scale.step_cents[0]:.1f} c, "
                  f"s = {scale.step_cents[1]:.1f} c,  R = {scale.hardness:.3f}")
            print(f"  proper      {scale.is_proper}")

    for w in (gen,):
        w.observe(sync, names="value")
    for w in (gen, card, style_pick, mode, palette, thresh, points):
        w.observe(redraw, names="value")

    sync()
    redraw()
    controls = widgets.VBox([gen, card, style_pick, mode, palette, thresh, points])
    return widgets.VBox([controls, out, text])


# --------------------------------------------------------------------------- #
# Three step sizes: the simplex explorer
# --------------------------------------------------------------------------- #
#: How far the marked tuning is kept from every edge of the simplex.  An edge is
#: where a step size reaches zero and the scale stops being ternary, so it is
#: not a tuning at all; 1e-3 of a period is ~1.2 cents, below anything audible
#: and far enough from the boundary that the metrics layer's 1e-6 cent
#: clustering tolerance never mistakes two step classes for one.
_SIMPLEX_EPS = 1e-3


def _clamp_simplex(
    u: float, v: float, eps: float = _SIMPLEX_EPS
) -> Tuple[float, float, float, bool]:
    """Force ``(u, v, 1 - u - v)`` into the open simplex, ``eps`` from every edge.

    Two independent sliders cannot express the constraint that ties them:
    ``w = 1 - u - v`` has to stay positive, and
    :class:`~biotuner.mos.ternary.TernaryScale` raises on a non-positive step.
    So the pair is repaired before it is used.  Each coordinate is clipped into
    ``[eps, 1 - 2*eps]``; if the pair still exceeds ``1 - eps``, the *excess
    above* ``eps`` is scaled down proportionally.  Shrinking the excess rather
    than the raw values keeps the direction the user dragged in and lands
    exactly on ``u + v = 1 - eps``, so ``w`` comes out at ``eps`` instead of at
    whatever rounding leaves behind.

    Parameters
    ----------
    u, v : float
        Requested period shares of the large and medium step classes.
    eps : float, default 1e-3
        Minimum share for every class.

    Returns
    -------
    (u, v, w, clamped) : tuple of float, float, float, bool
        ``clamped`` is True when the requested pair was outside the simplex, so
        the caller can report the move rather than make it silently.

    Examples
    --------
    An interior pair is returned untouched:

    >>> u, v, w, clamped = _clamp_simplex(0.5, 0.3)
    >>> round(u, 6), round(v, 6), round(w, 6), clamped
    (0.5, 0.3, 0.2, False)

    A pair beyond the hypotenuse is pulled back onto it, ``w`` landing on the
    floor and the sum staying exactly 1:

    >>> u, v, w, clamped = _clamp_simplex(0.8, 0.7)
    >>> round(u + v + w, 12), round(w, 6), clamped
    (1.0, 0.001, True)

    and the direction of the requested point survives the repair:

    >>> round((u - 0.001) / (v - 0.001), 9) == round(0.799 / 0.699, 9)
    True

    Both extremes are legal afterwards, which is the only property that matters:

    >>> all(x > 0 for x in _clamp_simplex(0.0, 0.0)[:3])
    True
    >>> all(x > 0 for x in _clamp_simplex(1.0, 1.0)[:3])
    True
    """
    if not 0.0 < eps < 1.0 / 3.0:
        raise ValueError(f"eps must lie in (0, 1/3), got {eps!r}")
    u0, v0 = float(u), float(v)
    lo, hi = eps, 1.0 - 2.0 * eps
    u1 = min(max(u0, lo), hi)
    v1 = min(max(v0, lo), hi)
    if u1 + v1 > 1.0 - eps:
        slack = 1.0 - 3.0 * eps
        over = (u1 - eps) + (v1 - eps)
        factor = slack / over
        u1 = eps + (u1 - eps) * factor
        v1 = eps + (v1 - eps) * factor
    clamped = abs(u1 - u0) > 1e-12 or abs(v1 - v0) > 1e-12
    return u1, v1, 1.0 - u1 - v1, clamped


def _simplex_words(word: str, max_words: int = 4000) -> List[str]:
    """The words a simplex explorer offers for ``word``'s signature.

    :func:`~biotuner.mos.ternary.ternary_words` keeps only the max-variety-3
    arrangements, which is the useful shortlist but need not contain the word
    the caller asked for -- ``'LLMsLMs'`` is a perfectly good 3L2M2s word and is
    not MV3.  Dropping it would mean the explorer opened on something other than
    what was requested, so the requested word is prepended when missing.

    Above fourteen notes ``ternary_words`` refuses to enumerate without a cap;
    ``max_words`` supplies one, and the list is then a prefix rather than the
    whole shortlist.

    Parameters
    ----------
    word : str
        Step pattern over ``'L'``, ``'M'``, ``'s'``; its counts fix the list.
    max_words : int, default 4000
        Enumeration cap, applied only above fourteen notes.

    Returns
    -------
    list of str
        ``word`` first if it is not in the shortlist, then the shortlist.

    Examples
    --------
    >>> _simplex_words('LMLsLMs')
    ['LMLsLMs', 'LMLsMLs']
    >>> _simplex_words('LLMsLMs')[0]
    'LLMsLMs'
    """
    from biotuner.mos.ternary import _check_word, ternary_words

    a, b, c = _check_word(word)
    cap = max_words if a + b + c > 14 else None
    try:
        options = ternary_words(a, b, c, max_words=cap)
    except ValueError:  # pragma: no cover - only for absurd counts
        options = []
    return options if word in options else [word] + options


def simplex_explorer(
    word: str = "LLMsLMs",
    *,
    period: float = 2.0,
    field: Optional[str] = "propriety",
):
    """Drag a tuning around a ternary word's triangle.

    The counterpart of :func:`mos_explorer` one step size further out.  An MOS
    has one degree of tuning freedom, so its tunings form an arc of the
    labyrinth; a three-step-size scale has two, so its tunings form a triangle
    (:mod:`biotuner.mos.ternary`).  The barycentric point ``(u, v, w)`` is the
    share of the period taken by *all* the large, all the medium and all the
    small steps, and the sliders move it while the shaded field says what
    happens to propriety, variety or just-intonation error as it moves.

    Parameters
    ----------
    word : str, default 'LLMsLMs'
        Step pattern over ``'L'``, ``'M'``, ``'s'``.  Fixes the signature: the
        dropdown offers the other worthwhile words with the same counts (see
        :func:`_simplex_words`), not other signatures.
    period : float, default 2.0
        Period as a frequency ratio.
    field : {'propriety', 'variety', 'ji_error', None}, default 'propriety'
        Which field to shade initially.

    Returns
    -------
    ipywidgets.Widget

    Notes
    -----
    Keeping the point legal: ``u`` and ``v`` are free but ``w = 1 - u - v`` is
    not, and a scale with a non-positive step does not exist -- the edges of the
    triangle are where a step vanishes and the scale turns binary.  Rather than
    let :class:`~biotuner.mos.ternary.TernaryScale` raise, the pair goes through
    :func:`_clamp_simplex` first, which pulls it back to ``eps`` from the
    nearest edge, writes the repaired values back into the sliders so the
    controls and the marked point never disagree, and prints what it did.

    Examples
    --------
    >>> simplex_explorer('LMLsLMs')                # doctest: +SKIP
    >>> simplex_explorer('LMLsLMs', field=None)    # doctest: +SKIP
    """
    widgets = _require("ipywidgets")
    import matplotlib.pyplot as plt

    from biotuner.mos import ternary as TN

    if field not in ("propriety", "variety", "ji_error", None):
        raise ValueError(
            f"field must be 'propriety', 'variety', 'ji_error' or None, got "
            f"{field!r}"
        )
    options = _simplex_words(word)
    start = TN.TernaryScale.equal_step(word, period)
    u0, v0, _ = start.barycentric

    style = {"description_width": "104px"}
    wide = widgets.Layout(width="94%")

    pick = widgets.Dropdown(options=options, value=word, description="word",
                            layout=wide, style=style)
    u = widgets.FloatSlider(
        value=u0, min=_SIMPLEX_EPS, max=1.0 - 2.0 * _SIMPLEX_EPS, step=0.002,
        description="u  (all L)", readout_format=".3f",
        continuous_update=False, layout=wide, style=style,
    )
    v = widgets.FloatSlider(
        value=v0, min=_SIMPLEX_EPS, max=1.0 - 2.0 * _SIMPLEX_EPS, step=0.002,
        description="v  (all M)", readout_format=".3f",
        continuous_update=False, layout=wide, style=style,
    )
    field_pick = widgets.ToggleButtons(
        options=[("propriety", "propriety"), ("variety", "variety"),
                 ("JI error", "ji_error"), ("none", None)],
        value=field, description="field", style={"description_width": "104px"},
    )
    res = widgets.IntSlider(value=60, min=20, max=160, step=10,
                            description="resolution", continuous_update=False,
                            layout=wide, style=style)
    play = widgets.Button(description="▶ play scale", button_style="success",
                          layout=widgets.Layout(width="46%"))
    out, text = widgets.Output(), widgets.Output()
    state: Dict[str, object] = {"scale": None, "guard": False}

    def redraw(*_):
        if state["guard"]:
            # A write-back below is in progress; the call that started it will
            # do the drawing, once, with the repaired pair.
            return
        want_u, want_v = float(u.value), float(v.value)
        uu, vv, ww, clamped = _clamp_simplex(want_u, want_v)
        if clamped:
            # Put the repaired pair in the sliders, so the controls and the
            # marked point never disagree.  The guard swallows the change
            # events this raises, which is why the figure is drawn once and not
            # three times.
            state["guard"] = True
            try:
                u.value, v.value = uu, vv
            finally:
                state["guard"] = False
        try:
            scale = TN.TernaryScale.from_barycentric(
                pick.value, uu, vv, ww, period
            )
        except ValueError as exc:  # pragma: no cover - the clamp prevents this
            with text:
                text.clear_output(wait=True)
                print(exc)
            return
        state["scale"] = scale

        with out:
            out.clear_output(wait=True)
            TN.plot_ternary_simplex(
                pick.value, field=field_pick.value, resolution=res.value,
                period=period, mark=scale, figsize=(7.8, 6.9),
            )
            plt.show()

        with text:
            text.clear_output(wait=True)
            if clamped:
                print(
                    f"u = {want_u:.4f}, v = {want_v:.4f} left the simplex "
                    f"(w = 1 - u - v must stay > 0); clamped to "
                    f"u = {uu:.4f}, v = {vv:.4f}, w = {ww:.4f}"
                )
            print(scale.summary())

    def on_play(_):
        scale = state.get("scale")
        if scale is None:
            return
        with text:
            try:
                from biotuner.biotuner_utils import listen_scale

                listen_scale(list(scale.ratios), fund=250, length=400)
            except Exception as exc:
                print(f"playback unavailable: {exc.__class__.__name__}: {exc}")

    # Switching word does not move the point: the same (u, v, w) is a tuning of
    # every word with these counts, and jumping back to the equal-step tuning
    # would hide the thing worth seeing -- that propriety depends on the
    # *arrangement* of the steps and not only on their sizes.
    for w in (pick, u, v, field_pick, res):
        w.observe(redraw, names="value")
    play.on_click(on_play)

    redraw()
    controls = widgets.VBox([pick, u, v, field_pick, res, play])
    return widgets.VBox([controls, out, text])


# --------------------------------------------------------------------------- #
# Scrubbing a recording
# --------------------------------------------------------------------------- #
def _fit_targets(fit) -> List[float]:
    """Recover the ratios a :class:`~biotuner.mos.derive.MOSFit` was fitted to.

    A trajectory keeps its fits and throws the windows away, but the fit is a
    lossless record of where the targets were: ``residual = target - offset -
    degree`` folded into half a period, so the target comes back exactly.  This
    is what lets the fit panel show the data rather than only the residuals.

    Parameters
    ----------
    fit : MOSFit

    Returns
    -------
    list of float
        The target ratios, folded into one period and in the order the fit
        recorded them.

    Examples
    --------
    >>> from biotuner.mos.derive import best_mos
    >>> from biotuner.mos.scale import MOSScale
    >>> r = MOSScale.from_signature(4, 3, tuning=19).ratios
    >>> back = _fit_targets(best_mos(r, max_cardinality=12))
    >>> max(abs(a - b) for a, b in zip(sorted(back), sorted(r))) < 1e-9
    True
    """
    scale = fit.scale
    degrees = scale.degrees
    pc = scale.period_cents
    out = []
    for k, res in zip(fit.assignments, fit.residuals):
        pos = (degrees[k] + fit.offset + res / pc) % 1.0
        out.append(scale.period**pos)
    return out


def _draw_path(ax, trajectory: Sequence, upto: int) -> int:
    """Draw windows ``0 .. upto`` of a trajectory on a labyrinth axes.

    Each fitted window is a point at ``(generator, cardinality)``; consecutive
    fitted windows are joined, and a ``None`` window breaks the line rather than
    being interpolated across, exactly as
    :func:`~biotuner.mos.plotting.plot_mos_trajectory` does -- a gap in the fit
    is not a smooth move through the labyrinth.

    The path is drawn in the signal red
    :func:`~biotuner.mos.plotting.plot_mos_trajectory` uses, which is also the
    one colour ``plot_labyrinth`` keeps out of its own palette, so the path
    never reads as an arc or as a highlighted generator.

    Parameters
    ----------
    ax : matplotlib polar axes
        A labyrinth, as drawn by
        :func:`~biotuner.mos.plotting.plot_labyrinth`.
    trajectory : sequence of MOSFit or None
    upto : int
        Index of the last window to draw, inclusive.

    Returns
    -------
    int
        How many windows were drawn.
    """
    runs: List[List[Tuple[float, int]]] = []
    current: List[Tuple[float, int]] = []
    for fit in trajectory[: upto + 1]:
        if fit is None:
            if current:
                runs.append(current)
            current = []
            continue
        current.append((fit.scale.generator, fit.scale.cardinality))
    if current:
        runs.append(current)

    drawn = 0
    for run in runs:
        theta = [2.0 * math.pi * g for g, _ in run]
        r = [c for _, c in run]
        if len(run) > 1:
            ax.plot(theta, r, "-", color="#C73E1D", lw=1.8, alpha=0.85,
                    zorder=11)
        ax.plot(theta, r, "o", color="#C73E1D", ms=7.0, mec="white", mew=1.0,
                zorder=12)
        drawn += len(run)
    return drawn


def trajectory_explorer(
    trajectory: Sequence,
    *,
    times: Optional[Sequence[float]] = None,
    max_cardinality: int = 18,
):
    """Scrub a recording window by window through the labyrinth.

    :func:`~biotuner.mos.derive.mos_trajectory` returns one
    :class:`~biotuner.mos.derive.MOSFit` per window, or ``None`` where nothing
    could be fitted.  :func:`~biotuner.mos.plotting.plot_mos_trajectory` shows
    the whole path at once; this shows one window at a time, with the path so
    far behind it, so a single window's fit can be inspected instead of taken on
    trust from a summary line.

    ``None`` windows are part of the record, not noise to be skipped: the slider
    stops on them, the labyrinth stays up with nothing highlighted, and the text
    panel says the window could not be fitted.  Silently jumping to the next
    fitted window would hide how much of the recording produced no scale at all.

    The fit panel is :func:`~biotuner.mos.plotting.plot_mos_fit`, which owns a
    two-row figure and only accepts a single ``ax``, so it is drawn as a second
    figure below the labyrinth rather than reimplemented here.  Its targets are
    reconstructed from the fit by :func:`_fit_targets`, since a trajectory does
    not keep the windows.

    Parameters
    ----------
    trajectory : sequence of MOSFit or None
        As returned by :func:`~biotuner.mos.derive.mos_trajectory` or
        :func:`~biotuner.mos.derive.trajectory_from_windows`.
    times : sequence of float, optional
        One time per window; the window index is used if omitted.
    max_cardinality : int, default 18
        Outermost labyrinth ring.  A window whose scale is larger than this is
        still reported in the text panel, but its marker falls outside the rim.

    Returns
    -------
    ipywidgets.Widget

    Raises
    ------
    ValueError
        If ``trajectory`` is empty, contains no successful fit, or ``times`` has
        a different length.

    Examples
    --------
    >>> from biotuner.mos.derive import trajectory_from_windows   # doctest: +SKIP
    >>> traj = trajectory_from_windows(windows, max_cardinality=12)  # doctest: +SKIP
    >>> trajectory_explorer(traj)                                 # doctest: +SKIP
    """
    widgets = _require("ipywidgets")
    import matplotlib.pyplot as plt

    from biotuner.mos import plotting as P

    fits = list(trajectory)
    if not fits:
        raise ValueError(
            "trajectory is empty; expected at least one window from "
            "mos_trajectory() or trajectory_from_windows()"
        )
    n_fitted = sum(f is not None for f in fits)
    if n_fitted == 0:
        raise ValueError(
            f"none of the {len(fits)} windows in this trajectory could be "
            "fitted (every entry is None); there is nothing to scrub through"
        )
    if times is not None and len(times) != len(fits):
        raise ValueError(
            f"times has {len(times)} entries but the trajectory has "
            f"{len(fits)} windows"
        )
    if max_cardinality < 2:
        raise ValueError(f"max_cardinality must be at least 2, got {max_cardinality}")

    style = {"description_width": "104px"}
    wide = widgets.Layout(width="94%")
    first = next(i for i, f in enumerate(fits) if f is not None)

    window = widgets.IntSlider(
        value=first, min=0, max=len(fits) - 1, step=1, description="window",
        continuous_update=False, layout=wide, style=style,
    )
    show_path = widgets.Checkbox(value=True, description="path so far")
    show_fit = widgets.Checkbox(value=True, description="fit residual panel")
    out, text = widgets.Output(), widgets.Output()

    def redraw(*_):
        i = int(window.value)
        fit = fits[i]
        t = float(times[i]) if times is not None else float(i)

        with out:
            out.clear_output(wait=True)
            fig = plt.figure(figsize=(7.6, 7.6))
            lab = fig.add_subplot(1, 1, 1, projection="polar")
            P.plot_labyrinth(
                max_cardinality, ax=lab, label="cents",
                highlight=fit.scale if fit is not None else None,
            )
            legend = lab.get_legend()
            if legend is not None:
                legend.remove()
            if show_path.value:
                _draw_path(lab, fits, i)
            lab.set_title(
                f"window {i} of {len(fits) - 1}"
                + (f"   —   {fit.signature}" if fit is not None
                   else "   —   no fit"),
                fontsize=12, pad=18,
            )
            plt.show()
            if show_fit.value and fit is not None:
                # plot_mos_fit's default figsize; shrinking it walks the title
                # into the degree-index labels it writes just above the axes.
                _, fit_axes = P.plot_mos_fit(fit, _fit_targets(fit))
                # Those labels sit at y = 1.02 in axes coordinates, which the
                # default title pad of 6 points is not enough to clear.
                fit_axes[0].set_title(fit_axes[0].get_title(), fontsize=12,
                                      pad=16)
                plt.show()

        with text:
            text.clear_output(wait=True)
            print(f"window {i}   t = {t:g}   "
                  f"({n_fitted} of {len(fits)} windows fitted)")
            if fit is None:
                print("  no MOS could be fitted to this window -- the labyrinth "
                      "is shown with nothing highlighted.")
                return
            improvement = fit.improvement
            imp = "exact" if math.isinf(improvement) else f"{improvement:.2f}x"
            print(f"  signature      {fit.signature}  "
                  f"({fit.scale.cardinality} notes)   {fit.scale.word}")
            print(f"  generator      {fit.scale.generator_cents:.2f} c "
                  f"(fraction {fit.scale.generator:.6f})")
            print(f"  error          {fit.error_cents:.3f} c weighted mean, "
                  f"max {fit.max_error_cents:.3f} c")
            print(f"  improvement    {imp} better than chance "
                  f"({fit.chance_error_cents:.2f} c)")

    for w in (window, show_path, show_fit):
        w.observe(redraw, names="value")

    redraw()
    controls = widgets.VBox([window, show_path, show_fit])
    return widgets.VBox([controls, out, text])


# --------------------------------------------------------------------------- #
# Dynamic Tonality: where matching the timbre pays
# --------------------------------------------------------------------------- #
#: Bounds of :func:`dissonance_explorer`'s partials slider, and therefore of the
#: ``n_partials`` it will accept.  The floor is 2 rather than 1 because a single
#: partial is a sine and there is no timbre to match; the ceiling keeps one
#: sweep inside a second or so, since cost grows as the square of the partial
#: count times the cardinality.  Note that even at the floor the two spectra
#: coincide -- see :func:`_sweep_verdict` on ties.
_PARTIALS_RANGE = (2, 16)


def _harmonic_spectrum(
    fundamental: float, n_partials: int
) -> Tuple[np.ndarray, np.ndarray]:
    """A plain harmonic series with the ``1/h`` roll-off of a matched spectrum.

    Same partial count and same amplitude envelope as
    :func:`~biotuner.mos.timbre.matched_spectrum` leaves by default, so any
    difference between the two is a difference in partial *placement* -- which
    is the only thing Dynamic Tonality changes (Milne et al. §6).

    Examples
    --------
    >>> f, a = _harmonic_spectrum(100.0, 4)
    >>> [float(x) for x in f], [round(float(x), 4) for x in a]
    ([100.0, 200.0, 300.0, 400.0], [1.0, 0.5, 0.3333, 0.25])
    """
    if n_partials < 1:
        raise ValueError(f"n_partials must be >= 1, got {n_partials!r}")
    if fundamental <= 0:
        raise ValueError(
            f"fundamental must be a positive frequency in Hz, got {fundamental!r}"
        )
    h = np.arange(1, int(n_partials) + 1, dtype=float)
    return h * float(fundamental), 1.0 / h


def _dissonance_curve(
    scale,
    *,
    matched: bool = True,
    n_partials: int = 8,
    fundamental: float = 250.0,
    resolution: int = 400,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sensory dissonance of two copies of one timbre, against interval width.

    The Plomp-Levelt curve of Milne et al. §6, computed for the timbre a scale
    would be played with: the same spectrum is sounded twice, the second copy
    transposed by the interval, and
    :func:`~biotuner.mos.timbre.spectral_dissonance` totals the roughness of
    every partial pair.  Minima fall where the two spectra's partials coincide,
    which for a matched timbre is at the scale's own degrees -- that is the
    claim the top panel of :func:`dissonance_explorer` puts on screen.

    Parameters
    ----------
    scale : MOSScale
    matched : bool, default True
        ``True`` uses :func:`~biotuner.mos.timbre.matched_spectrum`, ``False``
        the harmonic series of :func:`_harmonic_spectrum`.
    n_partials : int, default 8
    fundamental : float, default 250.0
        Hz.  Roughness is not scale-invariant, so this changes the numbers.
    resolution : int, default 400
        Interval widths sampled across one period, endpoints included.

    Returns
    -------
    (widths_cents, dissonance) : tuple of ndarray
        ``widths_cents`` runs from 0 to the period.

    Examples
    --------
    Sampled every hundred cents, a harmonic timbre reproduces the textbook
    curve: the peak sits about a semitone out, where the partials beat hardest,

    >>> from biotuner.mos.scale import MOSScale
    >>> m = MOSScale.from_signature(5, 2, tuning=12)
    >>> x, y = _dissonance_curve(m, matched=False, n_partials=6, resolution=13)
    >>> [round(float(v), 1) for v in x[:3]], float(x[-1])
    ([0.0, 100.0, 200.0], 1200.0)
    >>> float(x[int(y.argmax())])
    100.0

    the fifth is a local minimum,

    >>> bool(y[7] < y[6] and y[7] < y[8])
    True

    and the octave is a near-null, since every partial of the upper tone lands
    on one of the lower tone's:

    >>> bool(y[-1] < 0.05 * y.max())
    True
    """
    from biotuner.mos.timbre import matched_spectrum, spectral_dissonance

    if resolution < 2:
        raise ValueError(f"resolution must be at least 2, got {resolution}")
    if matched:
        freqs, amps = matched_spectrum(scale, fundamental, n_partials)
    else:
        freqs, amps = _harmonic_spectrum(fundamental, n_partials)
    widths = np.linspace(0.0, scale.period_cents, int(resolution))
    both_amps = np.concatenate([amps, amps])
    values = np.array(
        [
            spectral_dissonance(
                np.concatenate([freqs, freqs * 2.0 ** (x / T.PERIOD_CENTS)]),
                both_amps,
            )
            for x in widths
        ]
    )
    return widths, values


def _generator_sweep(
    n_large: int,
    n_small: int,
    *,
    n_partials: int = 8,
    resolution: int = 81,
    period: float = 2.0,
    fundamental: float = 250.0,
    bright: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Total scale dissonance across a signature's whole valid generator range.

    Both timbres at every generator: the harmonic series, and the partials
    remapped onto the scale's own lattice.  The endpoints of the range are the
    landmark equal temperaments where a step vanishes, so they are excluded --
    the scale is not itself there.

    Parameters
    ----------
    n_large, n_small : int
    n_partials : int, default 8
    resolution : int, default 81
        Interior generators sampled.
    period : float, default 2.0
    fundamental : float, default 250.0
    bright : bool, default True
        Sweep the generator range above half the period.

    Returns
    -------
    (generator_cents, harmonic, matched) : tuple of ndarray

    Examples
    --------
    >>> g, h, m = _generator_sweep(5, 2, n_partials=6, resolution=9)
    >>> len(g), round(float(g[0]), 1), round(float(g[-1]), 1)
    (9, 689.1, 716.6)
    >>> bool((m < h).mean() > 0.5)      # matched wins most, not all, of it
    True
    """
    from biotuner.mos.timbre import scale_dissonance

    if resolution < 2:
        raise ValueError(f"resolution must be at least 2, got {resolution}")
    lo, hi = T.signature_ranges(n_large, n_small)[1 if bright else 0]
    grid = np.linspace(float(lo), float(hi), int(resolution) + 2)[1:-1]
    harmonic = np.empty(grid.size)
    matched = np.empty(grid.size)
    for i, g in enumerate(grid):
        # validate=False: the generator is inside the signature's own range by
        # construction, and re-deriving the signature per sample is the whole
        # cost of the sweep.
        sc = MOSScale(n_large, n_small, float(g), period, validate=False)
        harmonic[i] = scale_dissonance(
            sc, n_partials=n_partials, matched=False, fundamental=fundamental
        )
        matched[i] = scale_dissonance(
            sc, n_partials=n_partials, matched=True, fundamental=fundamental
        )
    period_cents = T.PERIOD_CENTS * math.log2(period)
    return grid * period_cents, harmonic, matched


def _sweep_verdict(
    generator_cents: np.ndarray, harmonic: np.ndarray, matched: np.ndarray
) -> Dict[str, object]:
    """Where matching the timbre actually wins, measured rather than assumed.

    Dynamic Tonality is usually stated as though a matched timbre is better
    everywhere.  Over a swept generator it is not: it wins over most of the
    range and loses in narrow bands, chiefly where the *harmonic* timbre happens
    to be the matched one because the tuning is already near just.  This reduces
    a sweep to the numbers needed to say that out loud.

    Parameters
    ----------
    generator_cents, harmonic, matched : ndarray
        As returned by :func:`_generator_sweep`.

    A tie is neither a win nor a loss.  That distinction is not pedantry: the
    first partials of a matched spectrum *are* the harmonic series -- the
    lattice cannot move the fundamental, and with an octave period it cannot
    move the second partial either -- so at two partials the two spectra are
    bit-identical at every generator.  Counting those ties as losses would have
    the sweep report "harmonic lower over the whole range, worst loss 0.000",
    which is false in both halves.  They are counted separately and the caller
    is told.

    Parameters
    ----------
    generator_cents, harmonic, matched : ndarray
        As returned by :func:`_generator_sweep`.

    Returns
    -------
    dict
        ``n``, ``matched_wins``, ``ties`` and ``win_fraction``;
        ``median_reduction_pct`` over the whole sweep; ``worst_loss`` (the most
        negative reduction, 0.0 if the matched timbre never loses) with
        ``worst_loss_cents`` giving the generator it happens at; and
        ``loss_ranges``, the contiguous generator spans where the harmonic
        timbre is *strictly* the smoother of the two.

    Examples
    --------
    >>> g, h, m = _generator_sweep(5, 2, n_partials=6, resolution=41)
    >>> v = _sweep_verdict(g, h, m)
    >>> v['n'], 0.5 < v['win_fraction'] < 1.0
    (41, True)
    >>> bool(v['worst_loss'] < 0.0) and len(v['loss_ranges']) > 0
    True

    A sweep the matched timbre never loses reports no ranges at all:

    >>> v = _sweep_verdict(np.array([1.0, 2.0]), np.array([5.0, 5.0]),
    ...                    np.array([4.0, 4.0]))
    >>> v['win_fraction'], v['worst_loss'], v['loss_ranges']
    (1.0, 0.0, [])

    Two identical timbres tie; neither is reported as the smoother:

    >>> v = _sweep_verdict(np.array([1.0, 2.0]), np.array([5.0, 5.0]),
    ...                    np.array([5.0, 5.0]))
    >>> v['matched_wins'], v['ties'], v['loss_ranges'], v['worst_loss']
    (0, 2, [], 0.0)
    """
    g = np.asarray(generator_cents, dtype=float)
    h = np.asarray(harmonic, dtype=float)
    m = np.asarray(matched, dtype=float)
    if not (g.shape == h.shape == m.shape):
        raise ValueError(
            f"generator_cents, harmonic and matched must have the same shape, "
            f"got {g.shape}, {h.shape} and {m.shape}"
        )
    if g.size == 0:
        raise ValueError("cannot judge an empty sweep")
    reduction = h - m
    wins = reduction > 0.0
    lost = reduction < 0.0
    losses = np.flatnonzero(lost)

    ranges: List[Tuple[float, float]] = []
    if losses.size:
        start = prev = int(losses[0])
        for i in losses[1:]:
            i = int(i)
            if i != prev + 1:
                ranges.append((float(g[start]), float(g[prev])))
                start = i
            prev = i
        ranges.append((float(g[start]), float(g[prev])))

    return {
        "n": int(g.size),
        "matched_wins": int(wins.sum()),
        "ties": int((~wins & ~lost).sum()),
        "win_fraction": float(wins.mean()),
        "median_reduction_pct": float(np.median(100.0 * reduction / h)),
        "worst_loss": float(reduction.min()) if losses.size else 0.0,
        "worst_loss_cents": float(g[int(np.argmin(reduction))]) if losses.size
        else float("nan"),
        "loss_ranges": ranges,
    }


def dissonance_explorer(
    n_large: int = 5,
    n_small: int = 2,
    *,
    n_partials: int = 8,
    period: float = 2.0,
):
    """Retune the timbre with the scale, and watch the roughness follow.

    Milne et al. §6: a rank-2 tuning has its own lattice, and a timbre whose
    partials are mapped onto that lattice beats against the scale far less than
    a harmonic one does.  Two panels put both halves of that claim on screen.

    **Top** -- the Plomp-Levelt dissonance curve against interval width across
    one period, for the harmonic timbre and for the matched one, with the
    scale's own degrees marked.  A matched timbre's minima migrate onto the
    degrees; a harmonic one's stay at the just ratios wherever the scale happens
    to have put them.

    **Bottom** -- total scale dissonance as the generator sweeps its whole valid
    range, one curve per timbre.  This is the panel that shows where matching
    wins and where it does not, and it does not always win: see
    :func:`_sweep_verdict`, whose measurement is printed under the figure rather
    than summarised charitably.

    Parameters
    ----------
    n_large, n_small : int, default 5, 2
        The signature.  Fixed: the generator slider stays inside its valid
        range, so the scale on screen is always this one.
    n_partials : int, default 8
        Starting partial count.  Must lie in ``_PARTIALS_RANGE``, the partials
        slider's own bounds; a value outside it is refused rather than silently
        pulled to the nearest end, which would leave the explorer showing a
        timbre the caller did not ask for.
    period : float, default 2.0

    Returns
    -------
    ipywidgets.Widget

    Raises
    ------
    ValueError
        If the signature is not a co-prime pair of positive counts, or
        ``n_partials`` lies outside the slider's range.

    Notes
    -----
    What is cached, and why: the sweep is the expensive half -- ``resolution``
    scales times two timbres, each sounding every degree with every partial --
    while the generator slider changes only which vertical line is drawn on it.
    Sweeps are therefore kept in a dict on a closure variable, keyed by
    ``(n_large, n_small, n_partials, resolution)``: everything the curve depends
    on and nothing the slider touches.  ``period`` and the fundamental are fixed
    for the lifetime of one explorer, so they are not part of the key.

    Examples
    --------
    >>> dissonance_explorer()             # doctest: +SKIP
    >>> dissonance_explorer(4, 3)         # doctest: +SKIP
    """
    widgets = _require("ipywidgets")
    import matplotlib.pyplot as plt

    from biotuner.mos import plotting as P
    from biotuner.mos.timbre import dissonance_advantage

    if n_large < 1 or n_small < 1:
        raise ValueError(
            f"both step counts must be >= 1, got {n_large}L {n_small}s"
        )
    if math.gcd(n_large, n_small) != 1:
        raise ValueError(
            f"an MOS signature must be co-prime (Milne et al. §2); "
            f"{n_large}L{n_small}s has gcd = {math.gcd(n_large, n_small)}"
        )
    p_lo, p_hi = _PARTIALS_RANGE
    if not p_lo <= int(n_partials) <= p_hi:
        # ipywidgets would clip this into range without a word, leaving the
        # explorer showing a timbre nobody asked for.
        raise ValueError(
            f"n_partials must lie in {p_lo}..{p_hi}, the partials slider's "
            f"range, got {n_partials!r}"
        )
    cardinality = n_large + n_small
    period_cents = T.PERIOD_CENTS * math.log2(period)
    lo, hi = T.signature_ranges(n_large, n_small)[1]
    c_lo, c_hi = T.coherence_range(n_large, n_small, bright=True)
    lo_c, hi_c = float(lo) * period_cents, float(hi) * period_cents
    # Stay off the endpoints: a step vanishes there and the scale is an EDO.
    inset = 0.01 * (hi_c - lo_c)
    fundamental = 250.0

    style = {"description_width": "116px"}
    wide = widgets.Layout(width="94%")

    gen = widgets.FloatSlider(
        value=0.5 * (float(c_lo) + float(c_hi)) * period_cents,
        min=lo_c + inset, max=hi_c - inset, step=(hi_c - lo_c) / 400.0,
        description="generator ¢", readout_format=".2f",
        continuous_update=False, layout=wide, style=style,
    )
    partials = widgets.IntSlider(
        value=int(n_partials), min=p_lo, max=p_hi, step=1,
        description="partials", continuous_update=False, layout=wide,
        style=style,
    )
    show_matched = widgets.Checkbox(value=True, description="matched timbre")
    res = widgets.IntSlider(
        value=81, min=21, max=241, step=20, description="sweep resolution",
        continuous_update=False, layout=wide, style=style,
    )
    out, text = widgets.Output(), widgets.Output()
    Sweep = Tuple[np.ndarray, np.ndarray, np.ndarray]
    cache: Dict[Tuple[int, int, int, int], Sweep] = {}

    def sweep():
        key = (n_large, n_small, int(partials.value), int(res.value))
        if key not in cache:
            cache[key] = _generator_sweep(
                n_large, n_small, n_partials=int(partials.value),
                resolution=int(res.value), period=period,
                fundamental=fundamental,
            )
        return cache[key]

    def redraw(*_):
        g = gen.value / period_cents
        scale = MOSScale(n_large, n_small, g, period, validate=False)
        n_p = int(partials.value)
        gens, harmonic, matched = sweep()
        verdict = _sweep_verdict(gens, harmonic, matched)

        with out:
            out.clear_output(wait=True)
            fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.4))

            top = axes[0]
            x, y_h = _dissonance_curve(
                scale, matched=False, n_partials=n_p, fundamental=fundamental
            )
            top.plot(x, y_h, color=P.SMALL_COLOR, lw=1.6, label="harmonic timbre")
            if show_matched.value:
                _, y_m = _dissonance_curve(
                    scale, matched=True, n_partials=n_p, fundamental=fundamental
                )
                top.plot(x, y_m, color=P.LARGE_COLOR, lw=1.6,
                         label="matched timbre")
            for c in scale.cents[1:]:
                top.axvline(c, color="#6a6f78", lw=0.8, ls=":", alpha=0.8)
            top.axvline(scale.cents[0], color="#6a6f78", lw=0.8, ls=":",
                        alpha=0.8, label="scale degrees")
            top.set_xlim(0, period_cents)
            top.set_xlabel("interval width (cents)", fontsize=10)
            top.set_ylabel("sensory dissonance", fontsize=10)
            top.set_title(
                f"{scale.signature} at {scale.generator_cents:.2f} ¢   —   "
                f"roughness of two {n_p}-partial tones",
                fontsize=12,
            )
            top.legend(fontsize=8.5, frameon=False)
            top.spines[["top", "right"]].set_visible(False)

            bot = axes[1]
            bot.axvspan(float(c_lo) * period_cents, float(c_hi) * period_cents,
                        color="#dfe1e4", alpha=0.55, lw=0,
                        label="coherent (R < 2)")
            bot.plot(gens, harmonic, color=P.SMALL_COLOR, lw=1.6,
                     label="harmonic timbre")
            if show_matched.value:
                bot.plot(gens, matched, color=P.LARGE_COLOR, lw=1.6,
                         label="matched timbre")
            bot.axvline(scale.generator_cents, color="#C73E1D", lw=1.6,
                        label="this generator")
            bot.set_xlim(lo_c, hi_c)
            bot.set_xlabel("generator (cents)", fontsize=10)
            bot.set_ylabel("total scale dissonance", fontsize=10)
            bot.set_title(
                f"whole {cardinality}-note scale sounded at once, "
                f"across the {n_large}L{n_small}s range",
                fontsize=11,
            )
            bot.legend(fontsize=8.5, frameon=False)
            bot.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            plt.show()

        with text:
            text.clear_output(wait=True)
            adv = dissonance_advantage(
                scale, n_partials=n_p, fundamental=fundamental
            )
            print(f"{scale.signature} at {scale.generator_cents:.2f} c   "
                  f"R = {scale.hardness:.3f}, "
                  f"{'proper' if scale.is_proper else 'IMPROPER'}")
            print(f"  timbre         {n_p} partials, 1/h roll-off, "
                  f"fundamental {fundamental:g} Hz")
            if not show_matched.value:
                # Everything below compares the two timbres, and the matched
                # one is not on screen. Printing its numbers anyway would have
                # the text and the picture disagree about what is being shown.
                print(f"  this tuning    harmonic {adv['harmonic']:.3f}")
                print("  sweep          matched timbre hidden — tick it to "
                      "compare")
                return
            print(f"  this tuning    harmonic {adv['harmonic']:.3f}, "
                  f"matched {adv['matched']:.3f}, "
                  f"reduction {adv['reduction']:+.3f} "
                  f"({adv['reduction_pct']:+.2f} %)")
            print(f"  sweep          {verdict['n']} generators over "
                  f"{lo_c:.1f} .. {hi_c:.1f} c")
            print(f"                 matched lower at "
                  f"{verdict['matched_wins']} of {verdict['n']} "
                  f"({verdict['win_fraction']:.0%}), median reduction "
                  f"{verdict['median_reduction_pct']:+.2f} %")
            if verdict["ties"] == verdict["n"]:
                # Reachable at the low end of the partials slider: the lattice
                # cannot move partial 1, nor partial 2 when the period is the
                # octave, so a two-partial matched spectrum *is* the harmonic
                # series and the two curves coincide exactly.
                print("                 the two timbres are identical at every "
                      f"generator sampled — {n_p} partials is too few for the "
                      "lattice map to move any of them, so neither timbre is "
                      "the smoother")
            elif verdict["loss_ranges"]:
                shown = ", ".join(
                    f"{a:.1f}-{b:.1f}" for a, b in verdict["loss_ranges"][:6]
                )
                more = len(verdict["loss_ranges"]) - 6
                print(f"                 harmonic lower over {shown} c"
                      + (f" and {more} more band(s)" if more > 0 else ""))
                print(f"                 worst loss {verdict['worst_loss']:.3f} "
                      f"at {verdict['worst_loss_cents']:.1f} c")
            elif verdict["ties"]:
                print(f"                 harmonic never lower; "
                      f"{verdict['ties']} of {verdict['n']} generators tie")
            else:
                print("                 matched lower everywhere sampled")

    for w in (gen, partials, show_matched, res):
        w.observe(redraw, names="value")

    redraw()
    controls = widgets.VBox([gen, partials, show_matched, res])
    return widgets.VBox([controls, out, text])


# --------------------------------------------------------------------------- #
# Watching propriety break
# --------------------------------------------------------------------------- #

def morph_explorer(
    start: Tuple[int, int] = (5, 2),
    end: Tuple[int, int] = (4, 3),
    *,
    tuning: int = 19,
    max_cardinality: int = 18,
):
    """Drive a journey between two scales and watch the labyrinth from inside.

    The three strategies in :mod:`~biotuner.mos.morph` answer the same question
    -- how do you get from this scale to that one -- and disagree completely
    about the answer.  Reading three static figures side by side shows *that*
    they disagree; changing the destination with the strategy held fixed shows
    *why*, because you can watch one route lengthen while another stays put.

    The audio button is the point of the widget rather than an extra.  A tuning
    morph and a voice morph between the same pair can differ by an order of
    magnitude in total voice motion (8687 c against 668 c for ``5L2s`` to
    ``4L3s``), and that difference is far more obvious in the ear than in the
    plot: one glides through a wrap-around, the other barely moves.

    Not every pair admits every strategy.  A tuning morph needs both scales to
    have the same note count, and a tree route can exceed ``max_cardinality``.
    Rather than hiding those buttons, the widget lets you press them and prints
    what went wrong, since "you cannot slide a 7-note scale into a 5-note one"
    is the lesson, not an error to be papered over.

    Parameters
    ----------
    start, end : (int, int)
        Signatures ``(n_large, n_small)``.
    tuning : int, default 19
        EDO both endpoints are tuned in.  Endpoints in the same EDO make the
        tree routes exact.  19 rather than 12 because 12-EDO has no ``4L3s`` at
        all -- its only candidate is the landmark where the small step vanishes
        -- and an explorer whose opening view is an error message is a poor
        explorer.  Signatures with no generator in the chosen EDO are reported
        in the text panel, not raised.
    max_cardinality : int, default 18
        Outermost labyrinth ring, and the ceiling on a tree route.

    Returns
    -------
    ipywidgets.Widget

    Examples
    --------
    >>> morph_explorer()                      # doctest: +SKIP
    >>> morph_explorer((2, 3), (5, 7))        # doctest: +SKIP
    """
    widgets = _require("ipywidgets")
    import matplotlib.pyplot as plt

    from biotuner.mos import morph as MO
    from biotuner.mos.scale import MOSScale

    for name, sig in (("start", start), ("end", end)):
        if len(sig) != 2 or min(sig) < 1:
            raise ValueError(
                f"{name} must be a signature (n_large, n_small) with both "
                f"counts at least 1, got {sig!r}"
            )
    if max_cardinality < 3:
        raise ValueError(
            f"max_cardinality must be at least 3, got {max_cardinality}"
        )

    style = {"description_width": "92px"}
    narrow = widgets.Layout(width="230px")

    def _sig_pair(label, sig):
        return (
            widgets.IntSlider(value=sig[0], min=1, max=12, description=f"{label} L",
                              continuous_update=False, layout=narrow, style=style),
            widgets.IntSlider(value=sig[1], min=1, max=12, description=f"{label} s",
                              continuous_update=False, layout=narrow, style=style),
        )

    a_l, a_s = _sig_pair("from", start)
    b_l, b_s = _sig_pair("to", end)
    edo = widgets.IntSlider(value=tuning, min=5, max=53, description="EDO",
                            continuous_update=False, layout=narrow, style=style)
    strategy = widgets.ToggleButtons(
        options=list(MO.STRATEGIES), value="tuning", description="strategy",
        style={"description_width": "92px"},
    )
    frames = widgets.IntSlider(value=64, min=8, max=192, step=8, description="frames",
                               continuous_update=False, layout=narrow, style=style)
    palette = widgets.Dropdown(options=["light", "noir"], value="light",
                               description="palette", layout=narrow, style=style)
    column = widgets.Checkbox(value=True, description="generator column")
    play = widgets.Button(description="listen", icon="play",
                          layout=widgets.Layout(width="130px"))
    seconds = widgets.FloatSlider(value=8.0, min=2.0, max=20.0, step=1.0,
                                  description="seconds", continuous_update=False,
                                  layout=narrow, style=style)

    out, text, sound = widgets.Output(), widgets.Output(), widgets.Output()

    def _build():
        """The current morph, or the reason there isn't one.

        Both failure modes report rather than raise. A signature may have no
        generator at all in the chosen EDO -- most do not -- and a strategy may
        refuse the pair even when both scales exist. Either way the widget
        stays up and says which.
        """
        kwargs = {}
        if strategy.value == "tree":
            kwargs["max_cardinality"] = int(max_cardinality)
        else:
            kwargs["steps"] = int(frames.value)
        try:
            a = MOSScale.from_signature(
                int(a_l.value), int(a_s.value), tuning=int(edo.value))
            b = MOSScale.from_signature(
                int(b_l.value), int(b_s.value), tuning=int(edo.value))
            return MO.morph(a, b, strategy=strategy.value, **kwargs), None
        except ValueError as exc:
            return None, str(exc)

    def redraw(*_):
        m, problem = _build()
        sound.clear_output()
        with text:
            text.clear_output(wait=True)
            print(m.summary() if m is not None
                  else f"no {strategy.value} morph here:\n  {problem}")
        with out:
            out.clear_output(wait=True)
            if m is None:
                return
            fig = plt.figure(figsize=(12.6, 6.4))
            lab = fig.add_subplot(1, 2, 1, projection="polar")
            traj = fig.add_subplot(1, 2, 2)
            try:
                MO.plot_morph_path(m, ax=lab, palette=palette.value,
                                   column=column.value, colorbar=False,
                                   max_cardinality=max_cardinality)
            except ValueError:
                # Every frame was off-scale; the trajectory still tells the story.
                lab.axis("off")
                lab.text(0.5, 0.5, "never lands on a\nwell-formed scale",
                         transform=lab.transAxes, ha="center", va="center",
                         fontsize=11)
            MO.plot_morph_trajectory(m, ax=traj, palette=palette.value)
            fig.tight_layout()
            plt.show()

    def listen(_):
        m, problem = _build()
        with sound:
            sound.clear_output(wait=True)
            if m is None:
                print(f"nothing to play: {problem}")
                return
            try:
                from IPython.display import Audio, display
            except ImportError:  # pragma: no cover - IPython ships with ipywidgets
                print("IPython is needed to play audio")
                return
            wave = MO.morph_audio(m, seconds=float(seconds.value))
            display(Audio(wave, rate=44100, autoplay=False))

    for w in (a_l, a_s, b_l, b_s, edo, strategy, frames, palette, column):
        w.observe(redraw, names="value")
    play.on_click(listen)

    redraw()
    controls = widgets.VBox([
        strategy,
        widgets.HBox([widgets.VBox([a_l, a_s]), widgets.VBox([b_l, b_s]),
                      widgets.VBox([edo, frames]),
                      widgets.VBox([palette, seconds])]),
        widgets.HBox([column, play]),
    ])
    return widgets.VBox([controls, out, sound, text])

def _class_overlaps(scale, tol: float = 1e-6) -> List[Tuple[int, float]]:
    """Adjacent generic classes whose specific sizes overlap, and by how much.

    Milne et al. §2: a coherent scale keeps generic and specific interval sizes
    in step -- every ``(k+1)``-step interval larger than every ``k``-step one.
    Where that fails the two classes overlap in cents, and the scale is
    improper.  This is the same predicate
    :func:`~biotuner.mos.metrics.is_proper` decides, reported per class pair and
    with the size of the breach, which is what the right-hand panel of
    :func:`matrix_explorer` draws.

    Parameters
    ----------
    scale : MOSScale, Mode, or (cents_list, period_cents)
        Anything :func:`~biotuner.mos.metrics.generic_interval_sizes` accepts.
    tol : float, default 1e-6
        Cents of slack, so a tie is not an overlap.

    Returns
    -------
    list of (int, float)
        ``(k, overlap_cents)`` for every ``k`` whose class overlaps class
        ``k + 1``, ascending in ``k``.  Empty exactly when the scale is proper.

    Examples
    --------
    12-EDO's diatonic is proper -- its tritone is a tie, not an overlap:

    >>> from biotuner.mos.scale import MOSScale
    >>> _class_overlaps(MOSScale.from_signature(5, 2, tuning=12))
    []

    Pythagorean tuning breaks it, and only at one pair: the major third
    (408 c) overrunning the diminished fourth (384 c).

    >>> [(k, round(o, 3)) for k, o in
    ...  _class_overlaps(MOSScale.from_generator(3 / 2, 7))]
    [(3, 23.46)]

    The breach is exactly ``L - 2s``, which is why every constrained pair of a
    well-formed scale opens at the same instant, at ``R = 2``:

    >>> big, small = MOSScale.from_generator(3 / 2, 7).step_cents
    >>> round(big - 2 * small, 3)
    23.46
    """
    from biotuner.mos.metrics import generic_interval_sizes

    sizes = generic_interval_sizes(scale)
    out: List[Tuple[int, float]] = []
    for k in sorted(sizes)[:-1]:
        overlap = max(sizes[k]) - min(sizes[k + 1])
        if overlap > tol:
            out.append((k, float(overlap)))
    return out


def _overlap_onsets(
    n_large: int,
    n_small: int,
    *,
    period: float = 2.0,
    bright: bool = True,
    steps: int = 60,
) -> Dict[int, float]:
    """Generator, in cents, at which each class pair first overlaps.

    Found by bisection between the coherent boundary and the far end of the
    valid range, rather than asserted from theory -- the point of the panel is
    to watch propriety break, so where it breaks should be measured.

    The answer is a tie, and that is the interesting part: every constrained
    pair opens at the same generator, because the breach is ``L - 2s`` for all
    of them (see :func:`_class_overlaps`).  A signature with ``n_small == 1``
    has no constrained pair at all and this returns ``{}`` -- which is exactly
    why :func:`~biotuner.mos.metrics.is_proper` and
    :attr:`~biotuner.mos.scale.MOSScale.is_proper` part company there.

    Parameters
    ----------
    n_large, n_small : int
    period : float, default 2.0
    bright : bool, default True
    steps : int, default 60
        Bisection steps per class pair.

    Returns
    -------
    dict
        ``{k: generator_cents}`` for each class pair that ever overlaps.

    Examples
    --------
    The diatonic breaks at one pair, at 12-EDO's fifth -- the tuning where its
    embedding chromatic is equally tuned:

    >>> {k: round(g, 4) for k, g in _overlap_onsets(5, 2).items()}
    {3: 700.0}

    4L3s has two constrained pairs and they open together:

    >>> {k: round(g, 4) for k, g in _overlap_onsets(4, 3).items()}
    {2: 872.7273, 4: 872.7273}

    2L1s has none, so it is proper at every tuning in its range:

    >>> _overlap_onsets(2, 1)
    {}
    """
    from biotuner.mos.metrics import generic_interval_sizes

    lo, hi = T.signature_ranges(n_large, n_small)[1 if bright else 0]
    c_lo, c_hi = T.coherence_range(n_large, n_small, bright=bright)
    # The coherent sub-range shares the equalized endpoint with the valid range;
    # the improper part is whatever is left, and it can lie on either side.
    if float(c_lo) > float(lo):
        boundary, far = float(c_lo), float(lo)
    else:
        boundary, far = float(c_hi), float(hi)
    # Never evaluate on the endpoint itself: a step has vanished there.
    far = far + (boundary - far) * 1e-9

    def gaps(g: float) -> Dict[int, float]:
        sc = MOSScale(n_large, n_small, g, period, validate=False)
        sizes = generic_interval_sizes(sc)
        return {
            k: max(sizes[k]) - min(sizes[k + 1]) for k in sorted(sizes)[:-1]
        }

    at_far = gaps(far)
    period_cents = T.PERIOD_CENTS * math.log2(period)
    onsets: Dict[int, float] = {}
    for k, value in at_far.items():
        if value <= 0.0:
            continue
        a, b = boundary, far
        for _ in range(int(steps)):
            mid = 0.5 * (a + b)
            if gaps(mid)[k] > 0.0:
                b = mid
            else:
                a = mid
        onsets[k] = 0.5 * (a + b) * period_cents
    return onsets


def matrix_explorer(n_large: int = 5, n_small: int = 2, *, period: float = 2.0):
    """Watch propriety break, as the definition rather than as a boolean.

    Milne et al. §2: a well-formed scale is coherent while Blackwood's
    ``R = L / s`` stays below 2, and coherence means the generic and specific
    orderings agree -- every third wider than every second, and so on.  The
    generator slider deliberately runs the *whole* valid range, coherent part
    and improper part alike, so the boundary can be crossed rather than
    described.

    **Left** -- :func:`~biotuner.mos.metrics.interval_matrix` as a heatmap: one
    row per starting degree, one column per generic class, in cents.  Every
    column holds two sizes for a non-degenerate MOS (Myhill's property), and
    the column's spread is what has to stay inside its lane.

    **Right** -- one horizontal strip per generic class, spanning the specific
    sizes that class takes.  While the scale is proper the strips are disjoint
    and stack like a staircase; past ``R = 2`` neighbouring strips slide into
    each other, and the shaded band is the overlap.  That band *is* impropriety,
    not an illustration of it.

    Parameters
    ----------
    n_large, n_small : int, default 5, 2
        The signature.  Must be co-prime, as any MOS signature is.
    period : float, default 2.0

    Returns
    -------
    ipywidgets.Widget

    Notes
    -----
    The text panel prints propriety from both
    :attr:`~biotuner.mos.scale.MOSScale.is_proper` (the ``R <= 2`` shortcut) and
    :func:`~biotuner.mos.metrics.is_proper` (measured off the interval matrix),
    and flags any disagreement.  They agree everywhere except at
    ``n_small == 1``, where there is no constrained class pair at all and the
    measured verdict is "proper at every tuning" while the shortcut still calls
    ``R > 2`` improper.  That divergence is real, documented in
    :func:`~biotuner.mos.metrics.is_proper`, and reproduced by
    ``matrix_explorer(2, 1)``.

    Examples
    --------
    >>> matrix_explorer()               # doctest: +SKIP
    >>> matrix_explorer(2, 1)           # the propriety shortcut's blind spot
    ... # doctest: +SKIP
    """
    widgets = _require("ipywidgets")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from biotuner.mos import metrics as MT

    if n_large < 1 or n_small < 1:
        raise ValueError(
            f"both step counts must be >= 1, got {n_large}L {n_small}s"
        )
    if math.gcd(n_large, n_small) != 1:
        raise ValueError(
            f"an MOS signature must be co-prime (Milne et al. §2); "
            f"{n_large}L{n_small}s has gcd = {math.gcd(n_large, n_small)}"
        )
    cardinality = n_large + n_small
    if cardinality < 3:
        raise ValueError(
            f"{n_large}L{n_small}s has {cardinality} notes and only one generic "
            "interval class, so there is no adjacent pair to overlap"
        )

    period_cents = T.PERIOD_CENTS * math.log2(period)
    lo, hi = T.signature_ranges(n_large, n_small)[1]
    c_lo, c_hi = T.coherence_range(n_large, n_small, bright=True)
    lo_c, hi_c = float(lo) * period_cents, float(hi) * period_cents
    inset = 0.005 * (hi_c - lo_c)
    onsets = _overlap_onsets(n_large, n_small, period=period)

    style = {"description_width": "116px"}
    wide = widgets.Layout(width="94%")
    gen = widgets.FloatSlider(
        value=0.5 * (float(c_lo) + float(c_hi)) * period_cents,
        min=lo_c + inset, max=hi_c - inset, step=(hi_c - lo_c) / 500.0,
        description="generator ¢", readout_format=".2f",
        continuous_update=False, layout=wide, style=style,
    )
    mark = widgets.Checkbox(value=True, description="mark overlaps")
    out, text = widgets.Output(), widgets.Output()

    def redraw(*_):
        scale = MOSScale(n_large, n_small, gen.value / period_cents, period,
                         validate=False)
        matrix = MT.interval_matrix(scale)
        sizes = MT.generic_interval_sizes(scale)
        overlaps = dict(_class_overlaps(scale))

        with out:
            out.clear_output(wait=True)
            fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.0))

            left = axes[0]
            im = left.imshow(matrix, aspect="auto", cmap="viridis",
                             origin="upper", vmin=0.0, vmax=period_cents)
            left.set_xticks(range(cardinality - 1),
                            [str(k) for k in range(1, cardinality)])
            left.set_yticks(range(cardinality),
                            [str(i) for i in range(cardinality)])
            left.set_xlabel("generic interval class (steps)", fontsize=10)
            left.set_ylabel("starting degree", fontsize=10)
            left.set_title("interval matrix (cents)", fontsize=11)
            fig.colorbar(im, ax=left, shrink=0.8, pad=0.02, label="cents")
            if cardinality <= 12:
                # 0.65 of the period is where viridis turns from teal to green
                # and stops carrying white text; above it, ink.
                for i in range(cardinality):
                    for j in range(cardinality - 1):
                        value = matrix[i, j]
                        left.text(j, i, f"{value:.0f}", ha="center",
                                  va="center", fontsize=7,
                                  color="white" if value < 0.65 * period_cents
                                  else "#22252b")

            right = axes[1]
            for k in sorted(sizes):
                span = sizes[k]
                right.plot([min(span), max(span)], [k, k], "-",
                           color="#6a6f78", lw=2.0, solid_capstyle="butt",
                           zorder=3)
                right.plot(span, [k] * len(span), "|", color="#22252b",
                           ms=13, mew=1.6, zorder=4)
            if mark.value:
                for k, amount in overlaps.items():
                    x0 = min(sizes[k + 1])
                    right.add_patch(
                        Rectangle((x0, k - 0.42), amount, 1.84,
                                  facecolor="#C73E1D", alpha=0.30, lw=0,
                                  zorder=2)
                    )
                    right.text(x0 + amount / 2.0, k + 0.5, f"{amount:.1f} ¢",
                               ha="center", va="center", fontsize=8,
                               color="#C73E1D", zorder=5)
            right.set_ylim(0.4, cardinality - 0.4)
            right.set_xlim(-0.02 * period_cents, 1.02 * period_cents)
            right.set_yticks(range(1, cardinality))
            right.set_xlabel("specific size (cents)", fontsize=10)
            right.set_ylabel("generic interval class (steps)", fontsize=10)
            right.set_title(
                "specific sizes per class"
                + (f"   —   {len(overlaps)} overlapping pair(s)"
                   if overlaps else "   —   disjoint, so proper"),
                fontsize=11,
            )
            right.grid(color="#dfe1e4", lw=0.5, axis="x")
            right.set_axisbelow(True)
            right.spines[["top", "right"]].set_visible(False)
            fig.suptitle(
                f"{scale.signature} at {scale.generator_cents:.2f} ¢   —   "
                f"R = {scale.hardness:.3f}",
                fontsize=12,
            )
            fig.tight_layout()
            plt.show()

        with text:
            text.clear_output(wait=True)
            shortcut = scale.is_proper
            measured = bool(MT.is_proper(scale))
            print(f"{scale.signature} at {scale.generator_cents:.2f} c   "
                  f"L = {scale.step_cents[0]:.2f} c, "
                  f"s = {scale.step_cents[1]:.2f} c")
            print(f"  R              {scale.hardness:.4f}  "
                  f"(={MT.blackwood_r(scale):.4f} off the interval matrix)")
            print(f"  proper         {shortcut} from R <= 2, "
                  f"{measured} from the interval matrix")
            if shortcut != measured:
                print("  DISAGREEMENT   the R <= 2 shortcut and the measured "
                      "verdict differ; with n_small == 1 there is no "
                      "constrained class pair, so no tuning can be improper "
                      "(see biotuner.mos.metrics.is_proper).")
            if not onsets:
                print("  overlaps       no class pair can ever overlap in this "
                      "signature")
            else:
                first = min(onsets.values())
                tied = sorted(k for k, g in onsets.items()
                              if abs(g - first) < 1e-6)
                # Which way "out of the coherent range" points depends on the
                # signature: the coherent sub-range shares the equalized
                # endpoint with the valid range, and that endpoint can be
                # either the low one or the high one.
                mid = 0.5 * (float(c_lo) + float(c_hi)) * period_cents
                way = "below" if first < mid else "above"
                print(f"  first overlap  class {tied[0]} vs {tied[0] + 1} at "
                      f"{first:.3f} c, {way} the coherent range"
                      + (f"; tied with {tied[1:]} -- every constrained pair "
                         f"opens together at R = 2" if len(tied) > 1 else ""))
            if overlaps:
                listing = ", ".join(
                    f"{k}/{k + 1} by {amount:.2f} c"
                    for k, amount in sorted(overlaps.items())
                )
                print(f"  overlapping    {listing}")
            else:
                print("  overlapping    none at this tuning")

    for w in (gen, mark):
        w.observe(redraw, names="value")

    redraw()
    controls = widgets.VBox([gen, mark])
    return widgets.VBox([controls, out, text])
