"""
Smooth transitions for animation pipelines.

Three lightweight helpers for morphing between chords, extension levels,
and rendering paradigms — designed so a video pipeline can produce
flicker-free transitions on top of the existing
:mod:`.interference_patterns` paradigms.

Helpers
-------
- :func:`interpolate_chords` — chord-A ⇄ chord-B morph at parameter
  ``t ∈ [0, 1]`` (sorted-by-ratio pairing; unpaired components fade
  in/out at the endpoints).
- :func:`fade_in_components` — smoothly grow a base chord into its
  extension by fading in only the *new* components.
- :func:`blend_fields` — pixel-space crossfade between two
  ``field_2d`` GeometryData. Works between any pair of paradigms.

These three cover the three morphing axes documented in the score card:
phase animation works on the existing API directly (set
``input.phases = 2π·f·t`` per frame); chord morphs use
:func:`interpolate_chords`; extension morphs use
:func:`fade_in_components`; algorithm morphs use :func:`blend_fields`.

Composability
-------------
All three return either ``HarmonicInput`` or ``GeometryData``, so they
slot in cleanly to any rendering pipeline:

    inp_t = interpolate_chords(major, minor, t)
    fld_t = quasicrystal_field_2d(inp_t, n_fold=7)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from biotuner.harmonic_geometry.geometry_data import GeometryData
from biotuner.harmonic_geometry.inputs import HarmonicInput


# ============================================================ interpolate_chords


def _component_arrays(
    inp: HarmonicInput,
) -> Tuple[List[float], List[float], List[float]]:
    """Extract (ratios, amps, phases) as plain Python lists."""
    ratios = [float(r) for r in inp.to_ratios()]
    amps = inp.normalized_amplitudes().tolist()
    phases = (
        list(inp.phases)
        if inp.phases is not None
        else [0.0] * len(ratios)
    )
    return ratios, amps, phases


def _sort_components(
    ratios: List[float],
    amps: List[float],
    phases: List[float],
) -> Tuple[List[float], List[float], List[float]]:
    """Sort components by ratio (ascending). Stable for ties."""
    if not ratios:
        return [], [], []
    order = sorted(range(len(ratios)), key=lambda i: ratios[i])
    return (
        [ratios[i] for i in order],
        [amps[i] for i in order],
        [phases[i] for i in order],
    )


def interpolate_chords(
    a: HarmonicInput,
    b: HarmonicInput,
    t: float,
    *,
    base_freq: Optional[float] = None,
    equave: Optional[float] = None,
) -> HarmonicInput:
    """Smoothly morph chord ``a`` into chord ``b`` at parameter ``t ∈ [0, 1]``.

    Pairing strategy
    ----------------
    Both chords are sorted by ratio (ascending). The first
    ``min(N_a, N_b)`` components are paired index-by-index and
    linearly interpolated in ratio, amplitude, and phase. Any extra
    components in the longer chord are kept at their full ratio + phase,
    but their amplitudes ramp from 0 (at the opposite endpoint) to
    their full value (at their own endpoint) — so when ``t=0`` the
    output reproduces ``a`` exactly, and when ``t=1`` it reproduces ``b``.

    Parameters
    ----------
    a, b : HarmonicInput
        Source and target chords.
    t : float
        Interpolation parameter. Clamped to ``[0, 1]``.
    base_freq, equave : float, optional
        Override the corresponding fields on the returned input. By
        default they are linearly interpolated between ``a`` and ``b``.

    Returns
    -------
    HarmonicInput
        ``metadata['transition'] = {'kind': 'interpolate_chords', 't': t}``.

    Examples
    --------
    >>> major = HarmonicInput(ratios=[1, 5/4, 3/2])
    >>> minor = HarmonicInput(ratios=[1, 6/5, 3/2])
    >>> midpoint = interpolate_chords(major, minor, 0.5)
    >>> [float(r) for r in midpoint.to_ratios()]
    [1.0, 1.225, 1.5]
    """
    t = float(np.clip(t, 0.0, 1.0))

    ra, aa, pa = _sort_components(*_component_arrays(a))
    rb, ab, pb = _sort_components(*_component_arrays(b))

    n_pairs = min(len(ra), len(rb))

    new_ratios: List[float] = []
    new_amps: List[float] = []
    new_phases: List[float] = []

    # Paired components — full linear interpolation
    for i in range(n_pairs):
        new_ratios.append((1.0 - t) * ra[i] + t * rb[i])
        new_amps.append((1.0 - t) * aa[i] + t * ab[i])
        new_phases.append((1.0 - t) * pa[i] + t * pb[i])

    # Extras in `a` only — keep ratio, fade amplitude out as t→1
    if len(ra) > n_pairs:
        for i in range(n_pairs, len(ra)):
            new_ratios.append(ra[i])
            new_amps.append((1.0 - t) * aa[i])
            new_phases.append(pa[i])

    # Extras in `b` only — keep ratio, fade amplitude in as t→1
    if len(rb) > n_pairs:
        for i in range(n_pairs, len(rb)):
            new_ratios.append(rb[i])
            new_amps.append(t * ab[i])
            new_phases.append(pb[i])

    out_base_freq = (
        float(base_freq)
        if base_freq is not None
        else (1.0 - t) * a.base_freq + t * b.base_freq
    )
    out_equave = (
        float(equave)
        if equave is not None
        else (1.0 - t) * a.equave + t * b.equave
    )

    return HarmonicInput(
        ratios=new_ratios,
        amplitudes=new_amps,
        phases=new_phases,
        base_freq=out_base_freq,
        equave=out_equave,
        metadata={
            "transition": {
                "kind": "interpolate_chords",
                "t": float(t),
                "n_pairs": int(n_pairs),
                "n_extras_a": int(max(0, len(ra) - n_pairs)),
                "n_extras_b": int(max(0, len(rb) - n_pairs)),
            }
        },
    )


# ============================================================ fade_in_components


def fade_in_components(
    base: HarmonicInput,
    extended: HarmonicInput,
    t: float,
    *,
    match_tol: float = 1e-6,
) -> HarmonicInput:
    """Smoothly grow ``base`` into ``extended`` as ``t`` goes 0 → 1.

    For every component present in ``extended``:

    * If its ratio matches a base component (within ``match_tol``), the
      amplitude is linearly interpolated from the base value
      (at ``t=0``) to the extended value (at ``t=1``).
    * Otherwise (the component only exists in ``extended``), the
      amplitude ramps from 0 (at ``t=0``) to its full extended value
      (at ``t=1``).

    At ``t=0`` the output reproduces ``base`` exactly (every new
    component has amplitude 0); at ``t=1`` it reproduces ``extended``
    exactly (shared components have switched to extended amps).

    Designed for animating "chord at extension level 0 → 4 → 8" without
    components abruptly popping in.

    Parameters
    ----------
    base : HarmonicInput
        The starting (smaller) chord. All its components must also
        appear (within ``match_tol``) in ``extended``.
    extended : HarmonicInput
        The target chord — typically ``extend_harmonics(base, …)``.
    t : float
        In ``[0, 1]``. Clamped.
    match_tol : float, default=1e-6
        Tolerance for matching ratios when deciding which extended
        components are "new" vs "shared".

    Returns
    -------
    HarmonicInput
    """
    t = float(np.clip(t, 0.0, 1.0))

    rb, ab, pb = _component_arrays(base)
    re, ae, pe = _component_arrays(extended)

    # Index extended by ratio for shared/new classification.
    # We mark each extended component as 'shared' if there's a base
    # component within match_tol of its ratio.
    new_ratios: List[float] = []
    new_amps: List[float] = []
    new_phases: List[float] = []

    # Build a quick lookup of base ratios
    rb_arr = np.asarray(rb, dtype=np.float64)

    for r, amp, phi in zip(re, ae, pe):
        # Find the base index whose ratio matches within tol (if any).
        if len(rb_arr) > 0:
            diffs = np.abs(rb_arr - r)
            j = int(np.argmin(diffs))
            shared = bool(diffs[j] <= match_tol)
        else:
            shared = False
            j = -1

        if shared:
            # Interpolate shared amplitude from the base value (at t=0)
            # to the extended value (at t=1). Phase blends similarly.
            new_ratios.append(r)
            new_amps.append((1.0 - t) * ab[j] + t * amp)
            new_phases.append((1.0 - t) * pb[j] + t * phi)
        else:
            # New component — fade its amplitude in from 0 to the
            # extended value as t goes 0 → 1.
            new_ratios.append(r)
            new_amps.append(amp * t)
            new_phases.append(phi)

    return HarmonicInput(
        ratios=new_ratios,
        amplitudes=new_amps,
        phases=new_phases,
        base_freq=base.base_freq,
        equave=base.equave,
        metadata={
            "transition": {
                "kind": "fade_in_components",
                "t": float(t),
                "n_base": len(rb),
                "n_extended": len(re),
            }
        },
    )


# ================================================================ blend_fields


def blend_fields(
    geom_a: GeometryData,
    geom_b: GeometryData,
    t: float,
    *,
    require_same_grid: bool = True,
) -> GeometryData:
    """Pixel-space crossfade between two ``field_2d`` geometries.

    Used for **algorithm morphing** (e.g., harmonic_interference →
    quasicrystal): render two paradigms on the same grid, then blend
    them. No physical interpretation — purely a visual transition.

    Parameters
    ----------
    geom_a, geom_b : GeometryData
        Both must have ``geom_type='field_2d'`` and matching coordinate
        shapes.
    t : float
        In ``[0, 1]``. Clamped.
    require_same_grid : bool, default=True
        If True, additionally require that the ``field_grid`` arrays
        match within float tolerance, so the blended field has a
        well-defined coordinate system. Set False to allow blending
        fields whose grids differ (the resulting ``field_grid`` is
        taken from ``geom_a``).

    Returns
    -------
    GeometryData
        ``geom_type='field_2d'``, ``metadata.kind='blended'``, with the
        constituent paradigm names recorded in ``parameters``.
    """
    t = float(np.clip(t, 0.0, 1.0))

    if geom_a.geom_type != "field_2d" or geom_b.geom_type != "field_2d":
        raise ValueError(
            "blend_fields requires both inputs to have geom_type='field_2d'."
        )
    fa = np.asarray(geom_a.coordinates, dtype=np.float64)
    fb = np.asarray(geom_b.coordinates, dtype=np.float64)
    if fa.shape != fb.shape:
        raise ValueError(
            f"shape mismatch: {fa.shape} vs {fb.shape}."
        )

    if require_same_grid:
        if (geom_a.field_grid is None) != (geom_b.field_grid is None):
            raise ValueError(
                "field_grid presence mismatch: one geometry has it, the other doesn't."
            )
        if geom_a.field_grid is not None:
            for g_a, g_b in zip(geom_a.field_grid, geom_b.field_grid):
                if not np.allclose(g_a, g_b, atol=1e-9):
                    raise ValueError(
                        "field_grid mismatch between geom_a and geom_b "
                        "(set require_same_grid=False to bypass this check)."
                    )

    blended = (1.0 - t) * fa + t * fb

    kind_a = (geom_a.metadata or {}).get("kind", "unknown")
    kind_b = (geom_b.metadata or {}).get("kind", "unknown")

    return GeometryData(
        geom_type="field_2d",
        coordinates=blended,
        field_grid=geom_a.field_grid,
        parameters={
            "t": float(t),
            "kind_a": str(kind_a),
            "kind_b": str(kind_b),
        },
        metadata={
            "kind": "blended",
            "domain": "open_2d",
            "transition": {
                "kind": "blend_fields",
                "t": float(t),
                "kind_a": str(kind_a),
                "kind_b": str(kind_b),
            },
        },
    )
