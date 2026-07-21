"""Visualisations of how a biosignal maps to matter.

Each function takes a biosignal's peak frequencies (Hz) and renders one facet of
the mapping — to the periodic table, to materials, to natural domains, to the four
compositional levels — or the matching mechanism itself. Every function returns
``(fig, ax)`` so figures can be composed, saved, or restyled.

Design notes
------------
- Resonance *magnitude* uses a perceptually-uniform sequential ramp (magma).
- *Elements* are coloured by their own flame colour (their physical identity).
- *Domains* / *levels* are categorical: bars are always axis-labelled, so the
  semantic colour (water-blue, life-green, fire-red …) is reinforcement, never the
  sole identity channel — a fixed 7-domain taxonomy cannot be made CVD-distinct by
  colour alone (validated), so identity lives in the label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from biotuner.bioelements import periodic as P
from biotuner.bioelements import units
from biotuner.bioelements.affinity import material_affinity
from biotuner.bioelements.matching import match_elements, match_lines
from biotuner.bioelements.spectrum import element_spectrum
from biotuner.bioelements.bridges import element_flame_color

# --- palette (validated categorical for levels; domains = labelled reinforcement) --- #
INK, MUTED, GRIDC, SURFACE = "#22252b", "#6a6f78", "#e7e8ea", "#fcfcfb"
DOMAIN_COLORS = {
    "geosphere": "#a86a28", "hydrosphere": "#2e6fb0", "atmosphere": "#35a0be",
    "biosphere": "#4a9d5b", "cosmosphere": "#8a5fb8", "technosphere": "#52606e",
    "energetic-process": "#d1552e",
}
LEVEL_COLORS = {
    "element": "#c07a2e", "compound": "#2f83b8", "mixture": "#4e9a44", "structure": "#9a5ab0",
}
LEVEL_ORDER = ("element", "compound", "mixture", "structure")
RES_CMAP = "magma"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def peaks_from_signal(data, sf, *, fmin=1.5, fmax=45.0, n=6):
    """Convenience: strongest spectral peaks (Hz) of a raw 1-D signal."""
    from scipy.signal import welch, find_peaks
    f, pxx = welch(np.asarray(data, float), sf, nperseg=int(sf * 8))
    m = (f >= fmin) & (f <= fmax)
    fb, db = f[m], 10 * np.log10(pxx[m] + 1e-30)
    idx, _ = find_peaks(db, distance=2)
    if len(idx) == 0:
        return np.array([fb[np.argmax(db)]])
    return fb[np.sort(idx[np.argsort(db[idx])[::-1][:n]])]


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRIDC)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.set_axisbelow(True)


def _material_frame(peaks_hz, *, tol_cents=55.0, top=40):
    """All non-element materials scored, with kind + domain, sorted by affinity."""
    from biotuner.bioelements.materials import MATERIALS
    peaks_hz = np.atleast_1d(np.asarray(peaks_hz, float))
    rows = []
    for name, m in MATERIALS.items():
        rows.append({
            "material": name, "kind": m.kind, "domain": m.domain,
            "archetype": m.archetype or "",
            "affinity": material_affinity(peaks_hz, m, tol_cents=tol_cents, top=top),
        })
    return pd.DataFrame(rows).sort_values("affinity", ascending=False).reset_index(drop=True)


def _barlabels(ax, bars, vals, horizontal=True):
    for b, v in zip(bars, vals):
        if horizontal:
            ax.text(b.get_width() + 0.008, b.get_y() + b.get_height() / 2,
                    f"{v:.2f}", va="center", ha="left", fontsize=8, color=MUTED)
        else:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{v:.2f}", va="bottom", ha="center", fontsize=8, color=MUTED)


# --------------------------------------------------------------------------- #
# 1. biosignal -> the periodic table
# --------------------------------------------------------------------------- #
def plot_periodic_resonance(peaks_hz, *, table="air", tol_cents=55.0, ax=None):
    """The periodic table, each element lit by how strongly the signal resonates
    with its emission lines (sequential magma ramp)."""
    me = match_elements(peaks_hz, table=table, tol_cents=tol_cents).set_index("element")
    vmax = float(me["score"].max()) or 1.0
    cmap = plt.get_cmap(RES_CMAP)
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 5.6))
    else:
        fig = ax.figure
    for name in P.NAMES:
        pos = P.element_position(name)
        if pos is None:
            continue
        r, c = pos
        v = float(me["score"].get(name, 0.0)) / vmax
        ax.add_patch(FancyBboxPatch((c + 0.05, -r + 0.05), 0.9, 0.9,
                     boxstyle="round,pad=0,rounding_size=0.12",
                     linewidth=0, facecolor=cmap(v)))
        ax.text(c + 0.5, -r + 0.5, P.symbol(name), ha="center", va="center",
                fontsize=6.5, color="white" if v < 0.55 else "#1a1200", weight="bold")
    ax.set_xlim(-0.4, 18.4)
    ax.set_ylim(-P.N_ROWS - 0.2, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Which elements does the signal resonate with?",
                 fontsize=13, color=INK, weight="bold", loc="left", pad=10)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("resonance score", color=MUTED, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_visible(False)
    top3 = me["score"].sort_values(ascending=False).head(3)
    ax.text(0, -P.N_ROWS - 0.05, "top: " + " · ".join(f"{e}" for e in top3.index),
            fontsize=9, color=MUTED)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


# --------------------------------------------------------------------------- #
# 2. biosignal -> ranked elements (coloured by their flame colour)
# --------------------------------------------------------------------------- #
def plot_element_resonance(peaks_hz, *, top=18, table="air", tol_cents=55.0, ax=None):
    """Top elements by resonance, each bar in the element's own flame colour."""
    me = match_elements(peaks_hz, table=table, tol_cents=tol_cents).head(top).iloc[::-1]
    colors = [element_flame_color(e) for e in me["element"]]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(4, 0.34 * top)))
    else:
        fig = ax.figure
    y = np.arange(len(me))
    bars = ax.barh(y, me["score"], color=colors, height=0.72,
                   edgecolor=SURFACE, linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{P.symbol(e)}  {e}" for e in me["element"]], fontsize=9, color=INK)
    _barlabels(ax, bars, me["score"].values)
    ax.set_xlim(0, min(1.0, me["score"].max() * 1.18))
    ax.set_xlabel("resonance score", color=MUTED, fontsize=9)
    ax.grid(axis="x", color=GRIDC, linewidth=0.8)
    _clean(ax)
    ax.set_title("Top elements the signal resonates with",
                 fontsize=13, color=INK, weight="bold", loc="left", pad=8)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------- #
# 3. biosignal -> materials (bars labelled; colour = domain reinforcement)
# --------------------------------------------------------------------------- #
def plot_material_affinity(peaks_hz, *, top=14, tol_cents=55.0, ax=None, frame=None, legend=True):
    """Top materials by affinity; bar length = affinity, colour reinforces the
    material's natural domain (the material name is the identity, on the axis).
    Bare elements are excluded — this is the *materials* view (compounds, mixtures,
    structures); use :func:`plot_element_resonance` for elements."""
    df = _material_frame(peaks_hz, tol_cents=tol_cents) if frame is None else frame
    df = df[df["kind"] != "element"].head(top).iloc[::-1]
    colors = [DOMAIN_COLORS.get(d, MUTED) for d in df["domain"]]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.4, max(4, 0.36 * top)))
    else:
        fig = ax.figure
    y = np.arange(len(df))
    bars = ax.barh(y, df["affinity"], color=colors, height=0.72,
                   edgecolor=SURFACE, linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(df["material"], fontsize=9, color=INK)
    _barlabels(ax, bars, df["affinity"].values)
    ax.set_xlim(0, min(1.0, df["affinity"].max() * 1.18))
    ax.set_xlabel("affinity", color=MUTED, fontsize=9)
    ax.grid(axis="x", color=GRIDC, linewidth=0.8)
    _clean(ax)
    ax.set_title("Top materials by affinity",
                 fontsize=13, color=INK, weight="bold", loc="left", pad=8)
    doms = list(dict.fromkeys(df["domain"]))
    if legend and len(doms) > 1:
        handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=8,
                   markerfacecolor=DOMAIN_COLORS.get(d, MUTED), markeredgecolor="none",
                   label=d) for d in doms]
        ax.legend(handles=handles, title="domain", fontsize=8, title_fontsize=8,
                  frameon=False, loc="lower right", labelcolor=INK,
                  bbox_to_anchor=(1.0, 0.0))
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------- #
# 4. biosignal -> natural domains  (aggregate)
# --------------------------------------------------------------------------- #
def plot_resonance_by_domain(peaks_hz, *, tol_cents=55.0, agg="mean", ax=None, frame=None):
    """How the signal's resonance distributes across the natural domains of matter."""
    df = _material_frame(peaks_hz, tol_cents=tol_cents) if frame is None else frame
    df = df[df["kind"] != "element"]        # domains of compounds/mixtures/structures
    g = df.groupby("domain")["affinity"].agg(agg).sort_values()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.4))
    else:
        fig = ax.figure
    y = np.arange(len(g))
    bars = ax.barh(y, g.values, color=[DOMAIN_COLORS.get(d, MUTED) for d in g.index],
                   height=0.7, edgecolor=SURFACE, linewidth=1.2)
    ax.set_yticks(y); ax.set_yticklabels(g.index, fontsize=10, color=INK)
    _barlabels(ax, bars, g.values)
    ax.set_xlim(0, g.max() * 1.18)
    ax.set_xlabel(f"{agg} affinity", color=MUTED, fontsize=9)
    ax.grid(axis="x", color=GRIDC, linewidth=0.8)
    _clean(ax)
    ax.set_title("Resonance by domain",
                 fontsize=13, color=INK, weight="bold", loc="left", pad=8)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------- #
# 5. biosignal -> compositional levels  (aggregate)
# --------------------------------------------------------------------------- #
def plot_resonance_by_level(peaks_hz, *, tol_cents=55.0, agg="mean", ax=None, frame=None):
    """Resonance across the four compositional levels: element -> compound ->
    mixture -> structure."""
    df = _material_frame(peaks_hz, tol_cents=tol_cents) if frame is None else frame
    g = df.groupby("kind")["affinity"].agg(agg)
    order = [k for k in LEVEL_ORDER if k in g.index]
    vals = [g[k] for k in order]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
    else:
        fig = ax.figure
    x = np.arange(len(order))
    bars = ax.bar(x, vals, color=[LEVEL_COLORS[k] for k in order], width=0.66,
                  edgecolor=SURFACE, linewidth=1.5)
    ax.set_xticks(x); ax.set_xticklabels(order, fontsize=10, color=INK)
    _barlabels(ax, bars, vals, horizontal=False)
    ax.set_ylim(0, max(vals) * 1.2)
    ax.set_ylabel(f"{agg} affinity", color=MUTED, fontsize=9)
    ax.grid(axis="y", color=GRIDC, linewidth=0.8)
    _clean(ax)
    ax.set_title("Resonance by level",
                 fontsize=13, color=INK, weight="bold", loc="left", pad=8)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------- #
# 6. the mechanism: signal peaks vs an element's emission lines
# --------------------------------------------------------------------------- #
def plot_line_match(peaks_hz, name, *, table="air", tol_cents=55.0, ax=None):
    """Show the matching itself: an element's emission lines (a spectral barcode),
    the signal's folded peaks laid over them, and the coincidences that score."""
    sp = element_spectrum(name, table=table)
    line_nm = np.array([units.fold_to_optical(w, is_hz=False) for w in sp.wavelength]) / 10.0
    inten = np.asarray(sp.intensity, float); inten = inten / (inten.max() or 1)
    peaks_hz = np.atleast_1d(np.asarray(peaks_hz, float))
    fold_nm = np.array([units.fold_to_optical(pk, is_hz=True) for pk in peaks_hz]) / 10.0
    hits = match_lines(peaks_hz, sp, tol_cents=tol_cents)
    hit_nm = set(round(float(w) / 10.0, 2) for w in hits["line_wl"]) if len(hits) else set()

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 3.4))
    else:
        fig = ax.figure
    flame = element_flame_color(name)
    for nm, it in zip(line_nm, inten):
        matched = round(float(nm), 2) in hit_nm
        ax.vlines(nm, 0, 0.55 + 0.45 * it, color=flame if matched else "#c9ccd2",
                  linewidth=2.2 if matched else 0.9, alpha=1.0 if matched else 0.8, zorder=2 if matched else 1)
    for nm in fold_nm:
        ax.vlines(nm, -0.72, 0, color=INK, linewidth=1.6, zorder=3)
        ax.plot(nm, 0, marker="v", color=INK, markersize=7, zorder=4)
    for nm in fold_nm:
        if any(abs(nm - h) < 6 for h in hit_nm):
            ax.plot(nm, 0, marker="o", markersize=15, markerfacecolor="none",
                    markeredgecolor=flame, markeredgewidth=2.2, zorder=5)
    ax.axhline(0, color=GRIDC, linewidth=1)
    ax.set_xlim(378, 722)
    ax.set_ylim(-0.9, 1.05)
    ax.set_yticks([])
    ax.set_xlabel("wavelength (nm), octave-folded to the visible band", color=MUTED, fontsize=9)
    _clean(ax)
    ax.spines["left"].set_visible(False)
    score = float(hits["intensity"].sum() / (sp.intensity.sum() + 1e-9)) if len(hits) else 0.0
    ax.text(380, 0.92, f"{P.symbol(name)}  {name}", fontsize=12, color=flame, weight="bold")
    ax.text(380, -0.85, "your rhythms", fontsize=9, color=INK)
    ax.set_title(f"Matching the signal to {name}'s emission spectrum  ·  score {score:.2f}",
                 fontsize=13, color=INK, weight="bold", loc="left", pad=8)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------- #
# 7. the whole map, for one signal
# --------------------------------------------------------------------------- #
def plot_signal_map(peaks_hz, *, tol_cents=55.0):
    """A one-figure dashboard: the periodic table lit up, the resonance by domain
    and by compositional level, and the top materials — the whole mapping at once."""
    frame = _material_frame(peaks_hz, tol_cents=tol_cents)
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], hspace=0.32, wspace=0.28)
    ax_pt = fig.add_subplot(gs[0, :])
    plot_periodic_resonance(peaks_hz, tol_cents=tol_cents, ax=ax_pt)
    plot_resonance_by_domain(peaks_hz, ax=fig.add_subplot(gs[1, 0]), frame=frame)
    plot_resonance_by_level(peaks_hz, ax=fig.add_subplot(gs[1, 1]), frame=frame)
    plot_material_affinity(peaks_hz, top=10, ax=fig.add_subplot(gs[1, 2]), frame=frame, legend=False)
    fig.suptitle("A biosignal, mapped to matter", fontsize=16, color=INK, weight="bold", x=0.02, ha="left")
    fig.patch.set_facecolor(SURFACE)
    return fig, fig.axes
