"""
Faraday surface waves — parametric-instability cymatics.

Belongs to the ``parametric`` family of :mod:`biotuner.harmonic_geometry.media`:
when a fluid container is shaken vertically at frequency :math:`f`, the
free surface develops standing-wave patterns at subharmonic frequency
:math:`f / 2`. The selected wavenumber follows the capillary-gravity
dispersion relation

.. math:: \\omega^2 = g k + \\frac{\\sigma}{\\rho} k^3 ,

where :math:`g` is gravity, :math:`\\sigma` surface tension, and
:math:`\\rho` fluid density. For a chord, each component drives one
wavenumber; the surface response is a superposition of plane waves at
those wavenumbers, with their orientations selected by the pattern
symmetry (stripe / square / hexagonal / 12-fold quasipattern). This is
the operator behind "true" cymatics imagery (water on a vibrating plate)
— a different regime from the rigid-plate Chladni eigenmode it visually
resembles.

Dispersion regimes
------------------
- ``"gravity"`` — :math:`\\omega^2 = g k`, large wavelengths, smooth
  ripples. Dominant at low frequency.
- ``"capillary"`` — :math:`\\omega^2 = (\\sigma / \\rho) k^3`, small
  wavelengths, sharp wavelets. Dominant at high frequency.
- ``"mixed"`` (default) — full capillary-gravity dispersion using
  ``gravity``, ``surface_tension``, and ``density`` arguments.

Pattern symmetry
----------------
- ``"stripe"`` — one direction per chord component (1-fold).
- ``"square"`` — two orthogonal directions (4-fold).
- ``"hexagonal"`` (default) — three directions at 60° (resonant triad,
  the most commonly observed Faraday pattern).
- ``"twelve_fold"`` — six directions at 30° (quasipattern).
- ``"random"`` — N random directions per component (seedable).

References
----------
.. [1] Faraday, M. (1831). On a peculiar class of acoustical figures.
       Philosophical Transactions of the Royal Society.
.. [2] Cross, M. C. & Hohenberg, P. C. (1993). Pattern formation outside
       of equilibrium. Reviews of Modern Physics, 65, 851.
.. [3] Edwards, W. S. & Fauve, S. (1994). Patterns and quasi-patterns in
       the Faraday experiment. J. Fluid Mech., 278, 123.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from biotuner.harmonic_geometry.geometry_data import GeometryData
from biotuner.harmonic_geometry.inputs import HarmonicInput
from biotuner.harmonic_geometry.media.base import Medium

_VALID_PATTERNS = ("stripe", "square", "hexagonal", "twelve_fold", "random")
_VALID_DISPERSIONS = ("gravity", "capillary", "mixed")
_VALID_OUTPUTS = ("amplitude", "intensity", "real")


# ============================================================ dispersion


def _dispersion_wavenumbers(
    ratios: np.ndarray,
    *,
    dispersion: str,
    base_wavenumber: float,
    gravity: float,
    surface_tension: float,
    density: float,
) -> np.ndarray:
    """Map chord ratios to per-component wavenumbers via the dispersion relation.

    The lowest-frequency component is assigned ``base_wavenumber``; the rest
    follow from the chord ratio via the dispersion regime. This keeps the
    overall pattern size controllable while preserving the chord's
    relative-frequency structure.

    Parameters
    ----------
    ratios : (N,) ndarray
        Chord ratios, sorted or unsorted.
    dispersion : {"gravity", "capillary", "mixed"}
    base_wavenumber : float
        Wavenumber assigned to ratio = 1.0 (the chord root).
    gravity, surface_tension, density : float
        Used only in the "mixed" regime (full capillary-gravity).

    Returns
    -------
    (N,) ndarray of wavenumbers, all > 0.
    """
    if dispersion == "gravity":
        # ω² ∝ k  ⇒  k ∝ ω²  ⇒  k_i / k_0 = (ratio_i)²
        return base_wavenumber * ratios ** 2
    if dispersion == "capillary":
        # ω² ∝ k³  ⇒  k ∝ ω^(2/3)
        return base_wavenumber * ratios ** (2.0 / 3.0)
    if dispersion == "mixed":
        # Full ω² = g k + (σ/ρ) k³. Anchor k_0 at base_wavenumber, then for
        # each ratio compute ω_i² = ratio_i² · ω_0² and solve the depressed
        # cubic σ/ρ · k³ + g · k − ω² = 0 for the real positive root.
        sigma_over_rho = surface_tension / max(density, 1e-12)
        omega0_sq = gravity * base_wavenumber + sigma_over_rho * base_wavenumber ** 3
        ks = np.empty_like(ratios, dtype=np.float64)
        for i, r in enumerate(ratios):
            omega_sq = float(r) ** 2 * omega0_sq
            # σ/ρ · k³ + g · k − ω² = 0; coefficients of cubic ax³ + bx + c = 0
            a = sigma_over_rho
            b = gravity
            c = -omega_sq
            if a <= 0:
                # Pure-gravity fallback.
                ks[i] = omega_sq / max(gravity, 1e-12)
                continue
            # Solve depressed cubic via numpy (returns 3 roots).
            roots = np.roots([a, 0.0, b, c])
            real = [float(r.real) for r in roots
                    if abs(r.imag) < 1e-9 * max(1.0, abs(r.real)) and r.real > 0]
            if not real:
                # Shouldn't happen for this monotone cubic, but fall back.
                ks[i] = omega_sq / max(gravity, 1e-12)
            else:
                ks[i] = min(real)
        return ks
    raise ValueError(
        f"dispersion must be one of {_VALID_DISPERSIONS}; got {dispersion!r}."
    )


# ================================================================ angles


def _pattern_angles(
    pattern: str,
    *,
    rng: Optional[np.random.Generator] = None,
    n_random: int = 6,
) -> np.ndarray:
    """Return the directional angles (radians) for a given symmetry pattern.

    Each chord component is replicated at every angle in this array; the
    overall pattern's discrete rotational symmetry equals ``len(angles)``.
    """
    if pattern == "stripe":
        return np.array([0.0])
    if pattern == "square":
        return np.array([0.0, np.pi / 2])
    if pattern == "hexagonal":
        return np.array([0.0, np.pi / 3, 2 * np.pi / 3])
    if pattern == "twelve_fold":
        return np.linspace(0.0, np.pi, 6, endpoint=False)
    if pattern == "random":
        rng = rng or np.random.default_rng()
        return rng.uniform(0.0, np.pi, size=int(n_random))
    raise ValueError(
        f"pattern must be one of {_VALID_PATTERNS}; got {pattern!r}."
    )


# ============================================================== output map


def _output_transform(complex_field: np.ndarray, output: str) -> np.ndarray:
    if output == "amplitude":
        return np.abs(complex_field)
    if output == "intensity":
        return np.abs(complex_field) ** 2
    if output == "real":
        return complex_field.real
    raise ValueError(
        f"output must be one of {_VALID_OUTPUTS}; got {output!r}."
    )


# ============================================================== Faraday


class Faraday(Medium):
    """Parametric-family medium: Faraday capillary-gravity surface waves.

    For each chord component, computes the wavenumber selected by the
    chosen dispersion regime, then sums plane waves at that wavenumber
    rotated by every angle in the pattern's symmetry set. Drive amplitude
    modulates the per-mode amplitude as a soft proxy for the Mathieu
    growth rate; viscosity provides damping that suppresses higher
    wavenumbers (capillary modes are damped faster than gravity modes in
    real Faraday systems).

    Parameters
    ----------
    pattern : {"stripe", "square", "hexagonal", "twelve_fold", "random"},
        default "hexagonal"
        Discrete rotational symmetry of the response. Hexagonal is the
        most commonly observed Faraday pattern in the resonant-triad
        regime.
    dispersion : {"gravity", "capillary", "mixed"}, default "mixed"
        Dispersion relation that maps each chord component to a wavenumber.
    base_wavenumber : float, default 8 * pi
        Wavenumber assigned to the chord root (ratio = 1.0). Sets the
        overall pattern length scale on the unit-square domain. Defaults
        to ``8π`` so the chord root has ~4 wavelengths across the domain.
    drive_amplitude : float, default 1.0
        Per-mode amplitude scale.
    viscosity : float, default 0.01
        Viscous damping coefficient. Acts as a low-pass on the wavenumber
        spectrum via ``exp(-viscosity · k²)``; higher viscosity suppresses
        short-wavelength (capillary-dominated) modes.
    gravity, surface_tension, density : float
        Physical constants for the ``"mixed"`` dispersion regime. Defaults
        match water (g=9.81 m/s², σ=0.072 N/m, ρ=1000 kg/m³).
    extent : float, default 1.0
        Half-width of the square render domain ``[-extent, extent]²``.
    resolution : int, default 256
        Grid resolution per axis.
    output : {"amplitude", "intensity", "real"}, default "amplitude"
        How the complex field is reduced to a real scalar.
    n_random : int, default 6
        Number of random angles when ``pattern="random"``.
    seed : int, optional
        RNG seed for ``pattern="random"``.

    Notes
    -----
    This implementation captures Faraday *pattern selection* (the
    qualitative regimes) rather than the full Mathieu stability problem.
    Suitable for cymatics-style imagery; a more rigorous Floquet-based
    variant could be added later as ``mode="mathieu"``.
    """

    family = "parametric"

    def __init__(
        self,
        *,
        pattern: str = "hexagonal",
        dispersion: str = "mixed",
        base_wavenumber: float = 8.0 * math.pi,
        drive_amplitude: float = 1.0,
        viscosity: float = 0.01,
        gravity: float = 9.81,
        surface_tension: float = 0.072,
        density: float = 1000.0,
        extent: float = 1.0,
        resolution: int = 256,
        output: str = "amplitude",
        n_random: int = 6,
        seed: Optional[int] = None,
    ) -> None:
        if pattern not in _VALID_PATTERNS:
            raise ValueError(
                f"pattern must be one of {_VALID_PATTERNS}; got {pattern!r}."
            )
        if dispersion not in _VALID_DISPERSIONS:
            raise ValueError(
                f"dispersion must be one of {_VALID_DISPERSIONS}; got "
                f"{dispersion!r}."
            )
        if output not in _VALID_OUTPUTS:
            raise ValueError(
                f"output must be one of {_VALID_OUTPUTS}; got {output!r}."
            )
        if base_wavenumber <= 0:
            raise ValueError("base_wavenumber must be > 0.")
        if viscosity < 0:
            raise ValueError("viscosity must be >= 0.")
        if extent <= 0:
            raise ValueError("extent must be > 0.")
        if resolution < 4:
            raise ValueError("resolution must be >= 4.")
        if n_random < 1:
            raise ValueError("n_random must be >= 1.")

        self.pattern = pattern
        self.dispersion = dispersion
        self.base_wavenumber = float(base_wavenumber)
        self.drive_amplitude = float(drive_amplitude)
        self.viscosity = float(viscosity)
        self.gravity = float(gravity)
        self.surface_tension = float(surface_tension)
        self.density = float(density)
        self.extent = float(extent)
        self.resolution = int(resolution)
        self.output = output
        self.n_random = int(n_random)
        self.seed = seed

    # ----------------------------------------------------------- contract

    def default_source(self) -> None:
        return None

    def respond(
        self,
        forcing: HarmonicInput,
        **overrides: Any,
    ) -> GeometryData:
        if not isinstance(forcing, HarmonicInput):
            raise TypeError(
                "Faraday.respond requires a HarmonicInput; got "
                f"{type(forcing).__name__}."
            )

        pattern = overrides.pop("pattern", self.pattern)
        dispersion = overrides.pop("dispersion", self.dispersion)
        base_k = float(overrides.pop("base_wavenumber", self.base_wavenumber))
        drive = float(overrides.pop("drive_amplitude", self.drive_amplitude))
        visc = float(overrides.pop("viscosity", self.viscosity))
        g = float(overrides.pop("gravity", self.gravity))
        sigma = float(overrides.pop("surface_tension", self.surface_tension))
        rho = float(overrides.pop("density", self.density))
        extent = float(overrides.pop("extent", self.extent))
        resolution = int(overrides.pop("resolution", self.resolution))
        output = overrides.pop("output", self.output)
        n_random = int(overrides.pop("n_random", self.n_random))
        seed = overrides.pop("seed", self.seed)

        if overrides:
            raise TypeError(
                f"Unexpected override keys: {sorted(overrides)}."
            )
        if pattern not in _VALID_PATTERNS:
            raise ValueError(f"pattern must be one of {_VALID_PATTERNS}.")
        if dispersion not in _VALID_DISPERSIONS:
            raise ValueError(f"dispersion must be one of {_VALID_DISPERSIONS}.")
        if output not in _VALID_OUTPUTS:
            raise ValueError(f"output must be one of {_VALID_OUTPUTS}.")

        ratios = np.asarray(
            [float(r) for r in forcing.to_ratios()], dtype=np.float64
        )
        amps = forcing.normalized_amplitudes()
        phases = (
            np.asarray(forcing.phases, dtype=np.float64)
            if forcing.phases is not None
            else np.zeros_like(ratios)
        )

        ks = _dispersion_wavenumbers(
            ratios,
            dispersion=dispersion,
            base_wavenumber=base_k,
            gravity=g,
            surface_tension=sigma,
            density=rho,
        )

        rng = np.random.default_rng(seed) if seed is not None else None
        angles = _pattern_angles(pattern, rng=rng, n_random=n_random)

        x = np.linspace(-extent, extent, resolution)
        y = np.linspace(-extent, extent, resolution)
        X, Y = np.meshgrid(x, y, indexing="xy")

        # Superposition: each chord component i with wavenumber k_i is
        # replicated at every angle θ_j with phase φ_i. The amplitude is
        # damped by exp(-ν · k_i²) (viscous low-pass) and scaled by the
        # drive amplitude (acts as a Mathieu-growth proxy).
        field = np.zeros_like(X, dtype=np.complex128)
        n_angles = len(angles)
        damping = np.exp(-visc * ks ** 2)
        for k_i, a_i, phi_i, damp_i in zip(ks, amps, phases, damping):
            mode = np.zeros_like(X, dtype=np.complex128)
            for theta in angles:
                kx = k_i * math.cos(theta)
                ky = k_i * math.sin(theta)
                mode += np.exp(1j * (kx * X + ky * Y + phi_i))
            # Normalize so each component contributes consistent energy
            # regardless of pattern (more angles → divide further).
            mode /= max(1, n_angles)
            field += drive * a_i * damp_i * mode

        result = _output_transform(field, output)

        return GeometryData(
            geom_type="field_2d",
            coordinates=result,
            field_grid=(X, Y),
            parameters={
                "pattern": pattern,
                "dispersion": dispersion,
                "base_wavenumber": float(base_k),
                "drive_amplitude": float(drive),
                "viscosity": float(visc),
                "gravity": float(g),
                "surface_tension": float(sigma),
                "density": float(rho),
                "extent": float(extent),
                "resolution": int(resolution),
                "output": output,
                "wavenumbers": ks.tolist(),
                "angles": angles.tolist(),
            },
            metadata={
                "kind": "faraday_field_2d",
                "family": "parametric",
                "domain": "open_2d_surface",
                "symmetry": pattern,
            },
        )

    def __repr__(self) -> str:
        return (
            f"Faraday(pattern={self.pattern!r}, "
            f"dispersion={self.dispersion!r}, "
            f"viscosity={self.viscosity})"
        )
