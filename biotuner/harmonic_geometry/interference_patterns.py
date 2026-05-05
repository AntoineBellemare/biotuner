"""
Open-medium wave interference patterns from harmonic inputs.

This module is the *unbounded-wave* sibling of :mod:`.chladni` and
:mod:`.spherical_harmonics`. Where those produce eigenmodes confined by
a boundary (plate, sphere), the functions here produce
**travelling-wave coherent superpositions in free space**, each
generating its own family of emergent 2-D interference complexity.

What's here
-----------
Five rendering paradigms, each producing a qualitatively different
emergent symmetry from the same chord:

- :func:`harmonic_interference_field_2d` — **radial / soft rotational**.
  Chord components projected onto many plane-wave directions and
  averaged. Concentric ring + radial spoke patterns.
- :func:`quasicrystal_field_2d` — **discrete N-fold quasi-periodic**.
  Each chord component is a sum of N plane waves at evenly spaced
  angles (no averaging). For non-crystallographic N (5, 7, 11, 13) the
  result is a true quasi-crystal: exact N-fold symmetry, never repeats.
- :func:`standing_wave_lattice_2d` — **Cartesian 2-D lattice**. The
  chord's outer-product ``(r_i, r_j)`` paints an `N²`-mode 2-D Fourier
  lattice. Square-lattice symmetric, intermodulation cross-peaks.
- :func:`vortex_field_2d` — **spiral / topological**. Each chord
  component is a vortex mode with topological charge derived from the
  ratio. Sum produces interleaved spiral arms with phase singularities.
- :func:`interference_field_2d` — **N-source Young's-style fringes**.
  Idealised free-space superposition from emitters at fixed positions.

Composability with peaks_extension
----------------------------------
Every function here takes a :class:`HarmonicInput`. To enrich a chord
*before* rendering, pass it through one of the helpers in
:mod:`.extensions` first::

    chord = HarmonicInput(ratios=[1, 5/4, 3/2])
    rich  = extend_harmonics(chord, n_harmonics=8)
    field = harmonic_interference_field_2d(rich)

Each renderer scales its visual richness with component count.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.special import eval_genlaguerre, jv as bessel_jv

from biotuner.harmonic_geometry.geometry_data import GeometryData
from biotuner.harmonic_geometry.inputs import HarmonicInput


# ============================================================ shared helpers


def _resolve_amps_phases(
    n: int,
    amps: Optional[Sequence[float]],
    phases: Optional[Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Resolve per-component amplitudes / phases with the module's defaults."""
    if amps is None:
        a = np.full(n, 1.0 / max(n, 1), dtype=np.float64)
    else:
        a = np.asarray(amps, dtype=np.float64)
        if a.shape[0] != n:
            raise ValueError(
                f"amps has length {a.shape[0]} but {n} components were given."
            )
    if phases is None:
        p = np.zeros(n, dtype=np.float64)
    else:
        p = np.asarray(phases, dtype=np.float64)
        if p.shape[0] != n:
            raise ValueError(
                f"phases has length {p.shape[0]} but {n} components were given."
            )
    return a, p


def _input_components(
    inp: HarmonicInput,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (ratios, amplitudes, phases) from a HarmonicInput."""
    ratios = np.asarray(
        [float(r) for r in inp.to_ratios()], dtype=np.float64
    )
    amps = inp.normalized_amplitudes()
    phases = (
        np.asarray(inp.phases, dtype=np.float64)
        if inp.phases is not None
        else np.zeros(len(ratios), dtype=np.float64)
    )
    return ratios, amps, phases


_VALID_OUTPUTS = {"intensity", "amplitude", "amplitude_pow", "real"}


def _output_transform(
    field_complex: np.ndarray, output: str, power: float = 0.5
) -> np.ndarray:
    """Map a complex field to the requested real output."""
    if output == "intensity":
        return np.abs(field_complex) ** 2
    if output == "amplitude":
        return np.abs(field_complex)
    if output == "amplitude_pow":
        return np.abs(field_complex) ** float(power)
    if output == "real":
        return field_complex.real
    raise ValueError(
        f"output must be one of {sorted(_VALID_OUTPUTS)}; got {output!r}."
    )


def _xy_grid(extent: float, resolution: int) -> Tuple[np.ndarray, np.ndarray]:
    if extent <= 0:
        raise ValueError(f"extent must be > 0, got {extent!r}.")
    if resolution < 4:
        raise ValueError(f"resolution must be >= 4, got {resolution!r}.")
    x = np.linspace(-extent, extent, resolution)
    y = np.linspace(-extent, extent, resolution)
    X, Y = np.meshgrid(x, y, indexing="xy")
    return X, Y


# ====================================== 1. flagship: harmonic interference 2D


def harmonic_interference_field_2d(
    input: HarmonicInput,
    *,
    n_directions: int = 24,
    base_period: float = 1.0,
    extent: float = 1.5,
    resolution: int = 384,
    output: str = "amplitude_pow",
    power: float = 0.5,
) -> GeometryData:
    """Rich 2-D interference field with soft rotational symmetry.

    For each chord component ``i``, project ``n_directions`` coherent
    plane waves at evenly-spaced angles around the unit circle, then
    sum the complex field everywhere. Amplitudes are divided by
    ``n_directions`` to *average* the projections — the result has
    approximate continuous rotational symmetry (concentric rings),
    overlaid with a discrete N-fold lattice from the finite direction
    count.

    Mathematical form
    -----------------

    ::

        Ψ(x, y) = Σ_i  (a_i / N_dir) · Σ_θ  exp(i · k_i · (x·cos θ + y·sin θ)  +  i · φ_i)

    Where ``k_i = 2π · r_i / L`` and ``θ`` ranges over ``N_dir`` angles
    in ``[0, 2π)``.

    Parameters
    ----------
    input : HarmonicInput
    n_directions : int, default=24
        Plane-wave directions per component. Larger values approach
        full rotational symmetry.
    base_period : float, default=1.0
    extent : float, default=1.5
    resolution : int, default=384
    output : {'amplitude_pow', 'amplitude', 'intensity', 'real'}
    power : float, default=0.5

    Returns
    -------
    GeometryData
        ``geom_type='field_2d'``, ``metadata.kind='harmonic_interference_field_2d'``.
    """
    if base_period <= 0:
        raise ValueError(f"base_period must be > 0, got {base_period!r}.")
    if n_directions < 2:
        raise ValueError(f"n_directions must be >= 2, got {n_directions!r}.")

    ratios, amps, phases = _input_components(input)
    n = len(ratios)
    if n == 0:
        raise ValueError("input must have at least one component.")

    X, Y = _xy_grid(extent, resolution)

    angles = np.linspace(0.0, 2.0 * np.pi, n_directions, endpoint=False)
    cos_t = np.cos(angles)
    sin_t = np.sin(angles)

    field = np.zeros_like(X, dtype=np.complex128)
    inv_L = 1.0 / base_period
    for r_i, a_i, phi_i in zip(ratios, amps, phases):
        k_i = 2.0 * np.pi * r_i * inv_L
        for c, s in zip(cos_t, sin_t):
            field = field + (a_i / n_directions) * np.exp(
                1j * (k_i * (X * c + Y * s) + phi_i)
            )

    out = _output_transform(field, output, power=power)

    return GeometryData(
        geom_type="field_2d",
        coordinates=out,
        field_grid=(X, Y),
        parameters={
            "n_directions": int(n_directions),
            "base_period": float(base_period),
            "extent": float(extent),
            "resolution": int(resolution),
            "output": str(output),
            "power": float(power),
            "ratios": ratios.tolist(),
            "amps": amps.tolist(),
            "phases": phases.tolist(),
        },
        metadata={
            "kind": "harmonic_interference_field_2d",
            "domain": "open_2d",
            "symmetry": "soft_rotational",
            "n_components": int(n),
        },
    )


# =================================================== 2. quasicrystal field 2D


def quasicrystal_field_2d(
    input: HarmonicInput,
    *,
    n_fold: int = 5,
    base_period: float = 1.0,
    extent: float = 1.5,
    resolution: int = 384,
    output: str = "amplitude_pow",
    power: float = 0.5,
    direction_phase_step: float = 0.0,
) -> GeometryData:
    """Quasi-crystal field with exact discrete N-fold symmetry.

    Same idea as :func:`harmonic_interference_field_2d` but the
    plane-wave projections are kept *separate* (no averaging) and ``N``
    is small. For non-crystallographic ``n_fold`` (5, 7, 11, 13) the
    pattern is a *quasi-crystal*: exact ``n_fold``-rotational symmetry,
    no translational period — a 2-D analogue of Penrose tilings.

    Mathematical form
    -----------------

    ::

        Ψ(x, y) = Σ_i Σ_{k=0..n_fold-1}  a_i · exp(i · k_i · (x·cos α_k + y·sin α_k)  +  i · ψ_{i,k})

    where ``α_k = 2π·k/n_fold`` and ``ψ_{i,k} = φ_i + k · ε``
    (per-direction phase step ``ε`` defaults to 0).

    Parameters
    ----------
    input : HarmonicInput
    n_fold : int, default=5
        Number of plane-wave directions per chord component. Choose
        non-crystallographic values (5, 7, 11, 13) for a true
        quasi-crystal; crystallographic values (3, 4, 6) yield ordinary
        periodic lattices.
    base_period : float, default=1.0
    extent : float, default=1.5
    resolution : int, default=384
    output : {'amplitude_pow', 'amplitude', 'intensity', 'real'}
    power : float, default=0.5
    direction_phase_step : float, default=0.0
        Optional phase advance ``ε`` applied per direction. Setting
        ``ε = 2π/n_fold`` makes each direction's contribution
        chirally rotated, producing pinwheel-type quasicrystals.

    Returns
    -------
    GeometryData
        ``metadata.kind='quasicrystal_field_2d'``, ``symmetry='discrete_nfold'``.
    """
    if base_period <= 0:
        raise ValueError(f"base_period must be > 0, got {base_period!r}.")
    if n_fold < 2:
        raise ValueError(f"n_fold must be >= 2, got {n_fold!r}.")

    ratios, amps, phases = _input_components(input)
    n = len(ratios)
    if n == 0:
        raise ValueError("input must have at least one component.")

    X, Y = _xy_grid(extent, resolution)

    angles = np.arange(n_fold, dtype=np.float64) * (2.0 * np.pi / n_fold)
    cos_t = np.cos(angles)
    sin_t = np.sin(angles)

    field = np.zeros_like(X, dtype=np.complex128)
    inv_L = 1.0 / base_period
    for r_i, a_i, phi_i in zip(ratios, amps, phases):
        k_i = 2.0 * np.pi * r_i * inv_L
        for k_idx, (c, s) in enumerate(zip(cos_t, sin_t)):
            psi = phi_i + k_idx * direction_phase_step
            field = field + a_i * np.exp(
                1j * (k_i * (X * c + Y * s) + psi)
            )

    out = _output_transform(field, output, power=power)

    return GeometryData(
        geom_type="field_2d",
        coordinates=out,
        field_grid=(X, Y),
        parameters={
            "n_fold": int(n_fold),
            "base_period": float(base_period),
            "extent": float(extent),
            "resolution": int(resolution),
            "output": str(output),
            "power": float(power),
            "direction_phase_step": float(direction_phase_step),
            "ratios": ratios.tolist(),
            "amps": amps.tolist(),
            "phases": phases.tolist(),
        },
        metadata={
            "kind": "quasicrystal_field_2d",
            "domain": "open_2d",
            "symmetry": "discrete_nfold",
            "n_components": int(n),
        },
    )


# ============================================ 3. standing-wave 2D Fourier lattice


def standing_wave_lattice_2d(
    input: HarmonicInput,
    *,
    base_period: float = 1.0,
    extent: float = 2.0,
    resolution: int = 384,
    output: str = "amplitude_pow",
    power: float = 0.5,
    cross_phase: bool = False,
) -> GeometryData:
    """Cartesian 2-D Fourier lattice from the chord's outer product.

    Builds an ``N²``-mode 2-D field where the chord's ratios appear as
    *both* x- and y-direction frequencies, with intermodulation
    cross-peaks at every ``(r_i, r_j)`` combination.

    Mathematical form
    -----------------

    ::

        Ψ(x, y) = Σ_i Σ_j  a_i · a_j · exp(i · 2π · (r_i · x + r_j · y) / L  +  i · φ_{i,j})

    where ``φ_{i,j} = φ_i + φ_j`` (or ``φ_i · φ_j`` when ``cross_phase=True``,
    introducing additional asymmetry).

    The result is **square-lattice symmetric** (90° rotation invariant
    if all input phases are zero). With ``N`` chord components, the
    lattice has ``N²`` peaks. Extension multiplies this dramatically —
    a 10-component chord paints a 100-peak lattice with rich IMD-style
    sub-peaks between them.

    Parameters
    ----------
    input : HarmonicInput
    base_period : float, default=1.0
    extent : float, default=2.0
    resolution : int, default=384
    output : {'amplitude_pow', 'amplitude', 'intensity', 'real'}
    power : float, default=0.5
    cross_phase : bool, default=False
        If True, use the *antisymmetric* combination ``φ_i − φ_j`` for
        the cross-term phase instead of the symmetric ``φ_i + φ_j``.
        This breaks the ``(i, j) ↔ (j, i)`` exchange symmetry — and so
        the field's ``x ↔ y`` swap symmetry — introducing chirality
        in the lattice.

    Returns
    -------
    GeometryData
        ``metadata.kind='standing_wave_lattice_2d'``, ``symmetry='cartesian'``.
    """
    if base_period <= 0:
        raise ValueError(f"base_period must be > 0, got {base_period!r}.")

    ratios, amps, phases = _input_components(input)
    n = len(ratios)
    if n == 0:
        raise ValueError("input must have at least one component.")

    X, Y = _xy_grid(extent, resolution)
    inv_L = 1.0 / base_period

    field = np.zeros_like(X, dtype=np.complex128)
    for i, (r_i, a_i, phi_i) in enumerate(zip(ratios, amps, phases)):
        kx = 2.0 * np.pi * r_i * inv_L
        for j, (r_j, a_j, phi_j) in enumerate(zip(ratios, amps, phases)):
            ky = 2.0 * np.pi * r_j * inv_L
            # `+` is commutative ⇒ symmetric under (i, j) ↔ (j, i);
            # `-` is antisymmetric ⇒ breaks that symmetry, hence chiral.
            phi_ij = (phi_i - phi_j) if cross_phase else (phi_i + phi_j)
            field = field + (a_i * a_j) * np.exp(
                1j * (kx * X + ky * Y + phi_ij)
            )

    out = _output_transform(field, output, power=power)

    return GeometryData(
        geom_type="field_2d",
        coordinates=out,
        field_grid=(X, Y),
        parameters={
            "base_period": float(base_period),
            "extent": float(extent),
            "resolution": int(resolution),
            "output": str(output),
            "power": float(power),
            "cross_phase": bool(cross_phase),
            "ratios": ratios.tolist(),
            "amps": amps.tolist(),
            "phases": phases.tolist(),
        },
        metadata={
            "kind": "standing_wave_lattice_2d",
            "domain": "open_2d",
            "symmetry": "cartesian",
            "n_components": int(n),
            "n_modes": int(n * n),
        },
    )


# ===================================================== 4. vortex / spiral field


_RADIAL_KINDS = {"bessel", "laguerre_gauss", "propagating", "gaussian"}
_P_INDEX_RULES = {"denominator", "index", "zero"}


def _vortex_radial_factor(
    R: np.ndarray,
    k_i: float,
    l_i: int,
    p_i: int,
    beam_waist: float,
    radial_kind: str,
) -> np.ndarray:
    """Per-component radial factor for :func:`vortex_field_2d`.

    Returns a real-valued ndarray with the same shape as ``R``.
    The angular ``exp(i·l·θ)`` factor is applied by the caller.

    radial_kind options
    -------------------
    * ``'bessel'`` (default): ``J_{|l|}(k·r)``. The natural cylindrical
      eigenmode of an open 2-D circular geometry. Oscillates indefinitely
      with slow ``1/√(k·r)`` decay → many ring crossings, rich rendering.
    * ``'laguerre_gauss'``: full Laguerre-Gauss mode
      ``(r/w)^|l| · L_p^|l|(2r²/w²) · exp(-r²/w²)``.
      Has ``p`` radial zeros, giving a controlled number of rings.
    * ``'propagating'``: ``(r/w)^|l| · exp(-r²/2w²) · cos(k·r)``.
      Original Gaussian-envelope LG-zero modulated by a propagating
      cosine — cheaper than Bessel, gives oscillatory rings inside a
      bounded envelope.
    * ``'gaussian'``: original behaviour, ``(r/w)^|l| · exp(-r²/2w²)``,
      monotonic radial profile (kept for backward compatibility and
      for the smooth-blob aesthetic).
    """
    inv_w = 1.0 / beam_waist
    if radial_kind == "bessel":
        return bessel_jv(abs(l_i), k_i * R)
    if radial_kind == "propagating":
        rho_w = R * inv_w
        return (
            (rho_w ** abs(l_i))
            * np.exp(-(rho_w ** 2) / 2.0)
            * np.cos(k_i * R)
        )
    if radial_kind == "laguerre_gauss":
        rho_w = R * inv_w
        rho_w_sq = rho_w ** 2
        # Full LG_pl: (r/w)^|l| · L_p^|l|(2r²/w²) · exp(-r²/w²)
        return (
            (rho_w ** abs(l_i))
            * eval_genlaguerre(p_i, abs(l_i), 2.0 * rho_w_sq)
            * np.exp(-rho_w_sq)
        )
    if radial_kind == "gaussian":
        rho_w = R * inv_w
        return (rho_w ** abs(l_i)) * np.exp(-(rho_w ** 2) / 2.0)
    raise ValueError(
        f"radial_kind must be one of {sorted(_RADIAL_KINDS)}, got {radial_kind!r}."
    )


def vortex_field_2d(
    input: HarmonicInput,
    *,
    radial_kind: str = "bessel",
    beam_waist: float = 1.0,
    extent: float = 2.0,
    resolution: int = 384,
    output: str = "amplitude_pow",
    power: float = 0.5,
    charge_scale: float = 1.0,
    use_numerator_charges: bool = True,
    p_index_rule: str = "denominator",
    radial_indices: Optional[Sequence[int]] = None,
) -> GeometryData:
    """Coherent superposition of chord-driven optical vortex modes.

    Each chord component contributes one vortex mode with topological
    charge ``l_i`` derived from the ratio's rationalised numerator (so
    Major's ratio 5/4 → l = 5, ratio 3/2 → l = 3, etc.). The composite
    field has chord-driven **spiral arms and phase singularities** —
    the modulus shows interleaving rotational lobes braided around
    each other.

    Mathematical form
    -----------------

    The field is

    ::

        Ψ(r, θ) = Σ_i  a_i · radial_i(r) · exp(i · l_i · θ + i · φ_i)

    where the radial factor depends on ``radial_kind``. See
    :func:`_vortex_radial_factor` for the four supported families.

    The default ``radial_kind='bessel'`` uses the natural cylindrical
    eigenmode ``J_{|l|}(k·r)`` — its oscillatory radial profile gives a
    rich ring pattern that scales properly with extension (more chord
    components ⇒ more interlocking rings + spirals). The simpler
    Gaussian-envelope variants are visually softer and deliberately
    don't grow with extension.

    Parameters
    ----------
    input : HarmonicInput
    radial_kind : {'bessel', 'laguerre_gauss', 'propagating', 'gaussian'}, default='bessel'
        How to build the radial factor of each vortex mode. See
        :func:`_vortex_radial_factor` for details.
    beam_waist : float, default=1.0
        Reference radial scale ``w``. Wavenumbers are
        ``k_i = 2π · r_i / w``.
    extent : float, default=2.0
    resolution : int, default=384
    output : {'amplitude_pow', 'amplitude', 'intensity', 'real'}
    power : float, default=0.5
    charge_scale : float, default=1.0
        Multiplies the chord-derived azimuthal charges. ``charge_scale=2``
        doubles every component's spiral-arm count.
    use_numerator_charges : bool, default=True
        If True, ``l_i`` is the numerator of
        ``Fraction(r_i).limit_denominator(20)`` (musical chord ratios
        produce small spread-out integer charges this way). If False,
        ``l_i = round(charge_scale · r_i)``.
    p_index_rule : {'denominator', 'index', 'zero'}, default='denominator'
        Only used when ``radial_kind='laguerre_gauss'``. Picks the
        radial index ``p_i`` per component:

        * ``'denominator'`` — denominator of the rationalised ratio
          (Major's 5/4, 3/2 → p = 4, 2). Chord-distinct ring counts.
        * ``'index'`` — sequential ``p = 0, 1, 2, …`` per component.
          Useful when the chord ratios all rationalise to the same
          denominator.
        * ``'zero'`` — ``p = 0``, no radial nodes (collapses LG to its
          zeroth-order Gaussian-modulated form).
    radial_indices : sequence of int, optional
        If given, overrides ``p_index_rule`` and uses these ``p_i``
        values directly. Length must match the number of components.

    Returns
    -------
    GeometryData
        ``metadata.kind='vortex_field_2d'``, ``symmetry='spiral'``,
        ``parameters['radial_kind']`` records which radial flavour was used.
    """
    if radial_kind not in _RADIAL_KINDS:
        raise ValueError(
            f"radial_kind must be one of {sorted(_RADIAL_KINDS)}, "
            f"got {radial_kind!r}."
        )
    if p_index_rule not in _P_INDEX_RULES:
        raise ValueError(
            f"p_index_rule must be one of {sorted(_P_INDEX_RULES)}, "
            f"got {p_index_rule!r}."
        )
    if beam_waist <= 0:
        raise ValueError(f"beam_waist must be > 0, got {beam_waist!r}.")

    ratios, amps, phases = _input_components(input)
    n = len(ratios)
    if n == 0:
        raise ValueError("input must have at least one component.")

    X, Y = _xy_grid(extent, resolution)
    R = np.hypot(X, Y)
    THETA = np.arctan2(Y, X)

    # ── Topological charges per component ────────────────────────────────
    from fractions import Fraction

    charges: list[int] = []
    fraction_objs: list[Fraction] = [
        Fraction(float(r)).limit_denominator(20) for r in ratios
    ]
    if use_numerator_charges:
        for f in fraction_objs:
            charges.append(int(round(charge_scale * f.numerator)))
    else:
        for r_i in ratios:
            charges.append(int(round(charge_scale * float(r_i))))

    # ── Radial p-indices per component (LG only) ─────────────────────────
    if radial_indices is not None:
        p_indices = [int(p) for p in radial_indices]
        if len(p_indices) != n:
            raise ValueError(
                f"radial_indices has length {len(p_indices)} but {n} "
                "components were given."
            )
    else:
        if p_index_rule == "denominator":
            # Subtract 1 so a denominator of 1 produces p=0 (no extra rings).
            p_indices = [max(0, f.denominator - 1) for f in fraction_objs]
        elif p_index_rule == "index":
            p_indices = list(range(n))
        else:  # 'zero'
            p_indices = [0] * n

    # ── Compute the field ────────────────────────────────────────────────
    field = np.zeros_like(X, dtype=np.complex128)
    inv_w = 1.0 / beam_waist
    for r_i, a_i, phi_i, l_i, p_i in zip(
        ratios, amps, phases, charges, p_indices
    ):
        k_i = 2.0 * np.pi * float(r_i) / beam_waist
        radial = _vortex_radial_factor(
            R, k_i, l_i, p_i, beam_waist, radial_kind
        )
        # Normalise to keep amplitudes in a comparable range across
        # components — different l (and the radial flavour) give wildly
        # different peak magnitudes otherwise.
        peak = float(np.max(np.abs(radial)))
        if peak > 0:
            radial = radial / peak
        field = field + a_i * radial * np.exp(1j * (l_i * THETA + phi_i))

    out = _output_transform(field, output, power=power)

    return GeometryData(
        geom_type="field_2d",
        coordinates=out,
        field_grid=(X, Y),
        parameters={
            "radial_kind": str(radial_kind),
            "beam_waist": float(beam_waist),
            "extent": float(extent),
            "resolution": int(resolution),
            "output": str(output),
            "power": float(power),
            "charge_scale": float(charge_scale),
            "use_numerator_charges": bool(use_numerator_charges),
            "p_index_rule": str(p_index_rule),
            "charges": charges,
            "p_indices": p_indices,
            "ratios": ratios.tolist(),
            "amps": amps.tolist(),
            "phases": phases.tolist(),
        },
        metadata={
            "kind": "vortex_field_2d",
            "domain": "open_2d",
            "symmetry": "spiral",
            "n_components": int(n),
            "radial_kind": str(radial_kind),
        },
    )


# ============================================== 5. multi-source interference


_LAYOUTS = {"line", "circle", "pairwise"}


def _emitter_positions(
    n_sources: int,
    layout: str,
    spacing: float,
) -> np.ndarray:
    """Return ``(n_sources, 2)`` array of source positions in the xy plane."""
    if layout not in _LAYOUTS:
        raise ValueError(
            f"layout must be one of {sorted(_LAYOUTS)}, got {layout!r}."
        )
    if layout == "line":
        xs = (np.arange(n_sources) - (n_sources - 1) / 2.0) * spacing
        ys = np.zeros(n_sources)
    elif layout == "circle":
        theta = np.linspace(0, 2 * np.pi, n_sources, endpoint=False)
        xs = spacing * np.cos(theta)
        ys = spacing * np.sin(theta)
    else:  # 'pairwise'
        half = n_sources // 2
        rest = n_sources - half
        xs = np.concatenate([
            np.full(half, -spacing / 2.0),
            np.full(rest, +spacing / 2.0),
        ])
        ys = np.concatenate([
            (np.arange(half) - (half - 1) / 2.0) * (spacing / max(half, 1)),
            (np.arange(rest) - (rest - 1) / 2.0) * (spacing / max(rest, 1)),
        ])
    return np.stack([xs, ys], axis=1)


def interference_field_2d(
    input: HarmonicInput,
    *,
    layout: str = "line",
    spacing: float = 1.0,
    extent: float = 4.0,
    resolution: int = 384,
    base_wavelength: float = 0.6,
    output: str = "amplitude_pow",
    power: float = 0.5,
) -> GeometryData:
    """N-source interference field from a chord, in free space.

    Each chord component is associated with a wavelength
    ``λ_i = base_wavelength / r_i`` and emits an idealised (no
    ``1/r`` falloff) 2-D wave from the ``i``-th source position.
    The composite field is

    ::

        u(x, y) = Σ_i  a_i · exp(i · k_i · |r - r_i|  +  i · φ_i)

    The classical Young's two-slit case is recovered with a
    2-component equal-ratio input and ``layout='line'``.

    Returns
    -------
    GeometryData
        ``metadata.kind='interference_field_2d'``.
    """
    if base_wavelength <= 0:
        raise ValueError(
            f"base_wavelength must be > 0, got {base_wavelength!r}."
        )

    ratios, amps, phases = _input_components(input)
    n_sources = len(ratios)
    if n_sources == 0:
        raise ValueError("input must have at least one component.")

    sources = _emitter_positions(n_sources, layout, spacing)

    X, Y = _xy_grid(extent, resolution)

    field = np.zeros_like(X, dtype=np.complex128)
    for (sx, sy), r_i, a_i, phi_i in zip(sources, ratios, amps, phases):
        d = np.hypot(X - sx, Y - sy)
        lam_i = base_wavelength / r_i
        k_i = 2.0 * np.pi / lam_i
        field = field + a_i * np.exp(1j * (k_i * d + phi_i))

    out = _output_transform(field, output, power=power)

    return GeometryData(
        geom_type="field_2d",
        coordinates=out,
        field_grid=(X, Y),
        parameters={
            "layout": str(layout),
            "spacing": float(spacing),
            "extent": float(extent),
            "resolution": int(resolution),
            "base_wavelength": float(base_wavelength),
            "output": str(output),
            "power": float(power),
            "ratios": ratios.tolist(),
            "amps": amps.tolist(),
            "phases": phases.tolist(),
        },
        metadata={
            "kind": "interference_field_2d",
            "domain": "open_2d",
            "symmetry": "source_array",
            "n_sources": int(n_sources),
        },
    )
