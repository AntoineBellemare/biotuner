"""
Granular-layer redistribution on a wave field.

Belongs to the ``transport`` family of :mod:`biotuner.harmonic_geometry.media`:
takes a pre-computed scalar wave field (typically from
:class:`RigidPlate`) and produces the steady-state particle distribution
of grains driven by that field's effective potential — i.e. the actual
visible Chladni-figure that the bare eigenmode does not provide.

Physical model
--------------
For small-Péclet steady state, particle density follows a Boltzmann
distribution over an effective potential ``V(x, y)`` derived from the
wave field ``u(x, y)``::

    ρ(x, y) = (1 / Z) · exp(−V(x, y) / T)

The sign of ``V`` selects the physical regime:

- ``affinity > 0`` — ``V = +α · u²`` (or ``α · |∇u|²``): particles
  collect at **nodes** (where the field is small). This is the classical
  Chladni sand regime: dense grains driven away from antinodes by direct
  momentum coupling.
- ``affinity < 0`` — ``V = −α · u²``: particles collect at **antinodes**.
  This is the lycopodium / smoke regime where acoustic streaming pulls
  fine powder toward maxima of displacement.
- ``affinity = 0`` — uniform distribution.

``temperature`` (Boltzmann ``T``) controls how sharp the peaks are.
Low ``T`` gives crisp nodal lines / antinodal blobs; high ``T`` smears
them toward uniform.

``field_kind`` selects whether the potential is built from the
displacement (``"displacement"``: ``V ∝ u²``) or its gradient
(``"energy_gradient"``: ``V ∝ |∇u|²``) — the latter is closer to the
acoustic streaming picture and tends to produce sharper rings.

Outputs
-------
``output_mode='density'`` returns a 2-D scalar density field
(``geom_type='field_2d'``); ``output_mode='particles'`` samples ``n_particles``
positions from that density and returns a point cloud
(``geom_type='point_cloud_2d'``).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from biotuner.harmonic_geometry.geometry_data import GeometryData
from biotuner.harmonic_geometry.inputs import HarmonicInput
from biotuner.harmonic_geometry.media.base import Medium

_VALID_FIELD_KINDS = ("displacement", "energy_gradient")
_VALID_OUTPUT_MODES = ("density", "particles")


# =============================================================== potential


def _potential_from_field(
    field: np.ndarray,
    field_grid: Optional[Tuple[np.ndarray, ...]],
    field_kind: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build an effective potential ``V ∝ ±·…·`` from a scalar wave field.

    Returns ``(V_unsigned, valid_mask)`` where ``V_unsigned`` is the
    non-negative magnitude (the sign is applied later via ``affinity``)
    normalized to ``[0, 1]`` on the valid (finite) region, and
    ``valid_mask`` is a bool array marking finite cells.
    """
    valid = np.isfinite(field)
    if not np.any(valid):
        raise ValueError("Input field has no finite cells.")

    if field_kind == "displacement":
        # Time-averaged ⟨u²⟩ ∝ u² for cos(ωt) standing waves.
        V = np.zeros_like(field, dtype=np.float64)
        V[valid] = field[valid] ** 2
    elif field_kind == "energy_gradient":
        # Kinetic-energy / streaming-relevant: |∇u|² (per-cell magnitude).
        # Replace NaNs with 0 before gradient to avoid NaN propagation
        # across the entire domain; then re-mask.
        f_safe = np.where(valid, field, 0.0)
        if field_grid is not None and len(field_grid) >= 2:
            X, Y = field_grid[0], field_grid[1]
            # Spacing along each axis (assume regular meshgrid).
            dx = float(X[0, 1] - X[0, 0]) if X.shape[1] > 1 else 1.0
            dy = float(Y[1, 0] - Y[0, 0]) if Y.shape[0] > 1 else 1.0
            gy, gx = np.gradient(f_safe, dy, dx)
        else:
            gy, gx = np.gradient(f_safe)
        V = np.zeros_like(field, dtype=np.float64)
        V[valid] = gx[valid] ** 2 + gy[valid] ** 2
    else:
        raise ValueError(
            f"field_kind must be one of {_VALID_FIELD_KINDS}; got "
            f"{field_kind!r}."
        )

    vmax = float(V[valid].max())
    if vmax > 0:
        V = V / vmax
    return V, valid


# ================================================================ sampling


def _sample_particles(
    density: np.ndarray,
    field_grid: Tuple[np.ndarray, np.ndarray],
    n_particles: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Inverse-CDF sample ``n_particles`` (x, y) positions from a 2-D density.

    Cells outside the valid mask (NaN density) are treated as zero
    probability. Sub-pixel jitter is added uniformly within each chosen
    cell so samples are not stuck to the grid lattice.
    """
    valid = np.isfinite(density)
    p = np.where(valid, density, 0.0)
    total = p.sum()
    if total <= 0:
        # Degenerate: uniform over the valid region.
        p = np.where(valid, 1.0, 0.0)
        total = p.sum()
        if total <= 0:
            raise ValueError("Density has no valid cells; cannot sample.")
    p = p / total
    flat = p.ravel()

    H, W = density.shape
    X, Y = field_grid
    dx = float(X[0, 1] - X[0, 0]) if W > 1 else 1.0
    dy = float(Y[1, 0] - Y[0, 0]) if H > 1 else 1.0

    chosen = rng.choice(flat.size, size=int(n_particles), p=flat)
    rows = chosen // W
    cols = chosen % W
    jitter_x = rng.uniform(-0.5, 0.5, size=int(n_particles)) * dx
    jitter_y = rng.uniform(-0.5, 0.5, size=int(n_particles)) * dy
    xs = X[rows, cols] + jitter_x
    ys = Y[rows, cols] + jitter_y
    return np.stack([xs, ys], axis=1)


# ================================================================== Medium


class Granular(Medium):
    """Transport-family medium: Boltzmann-equilibrium grain redistribution.

    Parameters
    ----------
    affinity : float, default=1.0
        Sign and strength of the potential.
        ``> 0`` collects at nodes (classical sand);
        ``< 0`` at antinodes (powder / smoke);
        ``0`` gives a uniform distribution.
    temperature : float, default=0.1
        Boltzmann temperature in units of the normalized potential
        (which is rescaled to ``[0, 1]``). Lower = sharper.
    field_kind : {'displacement', 'energy_gradient'}, default='displacement'
        ``'displacement'`` builds ``V ∝ u²`` (canonical Chladni sand);
        ``'energy_gradient'`` builds ``V ∝ |∇u|²``.
    output_mode : {'density', 'particles'}, default='density'
        ``'density'`` returns ``field_2d``; ``'particles'`` returns
        ``point_cloud_2d`` sampled from the density.
    n_particles : int, default=4000
        Number of sampled particles when ``output_mode='particles'``.
    seed : int, optional
        RNG seed for particle sampling. Ignored in density mode.
    nodal_emphasis : bool, default=False
        If ``True``, build the sand density via the Gaussian-of-zero-
        crossing transform ``exp(-w² / σ²)`` applied directly to the
        input field amplitude. This is the cymatics / Chladni sand
        model: density concentrates on the nodal lines (zero-crossings).
        Bypasses the standard Boltzmann formulation
        (``affinity``/``temperature``/``field_kind``) — those are
        ignored when ``nodal_emphasis=True``.
    sigma : float, optional
        Stripe half-width for the nodal-emphasis transform. When
        ``None`` (the default) σ is auto-derived from the upstream
        field's ``parameters['int_modes']`` so that stripe width
        scales inversely with the chord's peak wavenumber — high-WN
        chords (e.g. Dim7 = [35, 42, 49, 60]) get small σ; low-WN
        chords get larger σ. Pass an explicit float to override.
        Ignored when ``nodal_emphasis=False``.

    Notes
    -----
    If a :class:`HarmonicInput` is passed to :meth:`respond`, it is
    auto-wrapped with :class:`RigidPlate` (default rectangular domain).
    To customize the source, pre-compute the field explicitly::

        plate = RigidPlate(domain=Rectangular(1, 1), resolution=512).respond(chord)
        sand  = Granular(affinity=1.0, temperature=0.05).respond(plate)
    """

    family = "transport"

    def __init__(
        self,
        *,
        affinity: float = 1.0,
        temperature: float = 0.1,
        field_kind: str = "displacement",
        output_mode: str = "density",
        n_particles: int = 4000,
        seed: Optional[int] = None,
        nodal_emphasis: bool = False,
        sigma: Optional[float] = None,
    ) -> None:
        if field_kind not in _VALID_FIELD_KINDS:
            raise ValueError(
                f"field_kind must be one of {_VALID_FIELD_KINDS}; got "
                f"{field_kind!r}."
            )
        if output_mode not in _VALID_OUTPUT_MODES:
            raise ValueError(
                f"output_mode must be one of {_VALID_OUTPUT_MODES}; got "
                f"{output_mode!r}."
            )
        if temperature <= 0:
            raise ValueError(
                f"temperature must be > 0; got {temperature!r}."
            )
        if n_particles < 1:
            raise ValueError(
                f"n_particles must be >= 1; got {n_particles!r}."
            )
        if sigma is not None and sigma <= 0:
            raise ValueError(f"sigma must be > 0; got {sigma!r}.")
        self.affinity = float(affinity)
        self.temperature = float(temperature)
        self.field_kind = field_kind
        self.output_mode = output_mode
        self.n_particles = int(n_particles)
        self.seed = seed
        self.nodal_emphasis = bool(nodal_emphasis)
        self.sigma = sigma  # None == auto-scale from upstream metadata

    # ----------------------------------------------------------- contract

    def default_source(self) -> Medium:
        from biotuner.harmonic_geometry.media.eigenmode.rigid_plate import (
            RigidPlate,
        )

        return RigidPlate()

    def respond(
        self,
        forcing,
        **overrides: Any,
    ) -> GeometryData:
        """Compute the steady-state grain distribution from a wave field.

        Parameters
        ----------
        forcing : HarmonicInput or GeometryData
            If a :class:`HarmonicInput`, auto-wrap with the default source
            (a :class:`RigidPlate`). If a :class:`GeometryData`, must have
            ``geom_type='field_2d'``.
        **overrides
            Per-call overrides: any of the constructor parameters above.

        Returns
        -------
        GeometryData
            ``field_2d`` (density) or ``point_cloud_2d`` (sampled particles).
        """
        affinity = float(overrides.pop("affinity", self.affinity))
        temperature = float(overrides.pop("temperature", self.temperature))
        field_kind = overrides.pop("field_kind", self.field_kind)
        output_mode = overrides.pop("output_mode", self.output_mode)
        n_particles = int(overrides.pop("n_particles", self.n_particles))
        seed = overrides.pop("seed", self.seed)
        nodal_emphasis = bool(
            overrides.pop("nodal_emphasis", self.nodal_emphasis)
        )
        sigma = overrides.pop("sigma", self.sigma)  # may be None == auto

        if overrides:
            raise TypeError(
                f"Unexpected override keys: {sorted(overrides)}. "
                "Granular.respond accepts overrides for affinity, "
                "temperature, field_kind, output_mode, n_particles, "
                "seed, nodal_emphasis, sigma."
            )
        if field_kind not in _VALID_FIELD_KINDS:
            raise ValueError(f"field_kind must be one of {_VALID_FIELD_KINDS}.")
        if output_mode not in _VALID_OUTPUT_MODES:
            raise ValueError(f"output_mode must be one of {_VALID_OUTPUT_MODES}.")
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")
        if sigma is not None and sigma <= 0:
            raise ValueError("sigma must be > 0.")

        field_data = self._resolve_field(forcing)
        if field_data.geom_type != "field_2d":
            raise TypeError(
                "Granular expects a 2-D scalar wave field "
                "(geom_type='field_2d'); got geom_type="
                f"{field_data.geom_type!r}."
            )

        field = np.asarray(field_data.coordinates, dtype=np.float64)
        field_grid = field_data.field_grid

        if nodal_emphasis:
            # Cymatics sand model: density concentrates on zero-crossings.
            # Bypasses the Boltzmann formulation — `affinity`, `temperature`,
            # and `field_kind` are ignored on this branch.
            if sigma is None:
                # Auto-σ: scale inversely with the upstream chord's peak
                # wavenumber so stripe width tracks the local wavelength.
                # Falls back to 0.05 if no int_modes metadata is present.
                modes_meta = (field_data.parameters or {}).get("int_modes")
                if modes_meta:
                    from biotuner.harmonic_geometry.media.eigenmode.rigid_plate import (
                        _auto_sigma_for_modes,
                    )
                    sigma = _auto_sigma_for_modes(modes_meta)
                else:
                    sigma = 0.05
            sigma = float(sigma)
            valid = np.isfinite(field)
            density = np.zeros_like(field, dtype=np.float64)
            density[valid] = np.exp(
                -(field[valid] * field[valid]) / (sigma * sigma)
            )
        else:
            V_unsigned, valid = _potential_from_field(
                field, field_grid, field_kind
            )
            # Apply signed affinity: V = affinity * V_unsigned ∈ [-|aff|, +|aff|].
            V = affinity * V_unsigned

            # Boltzmann distribution. Subtract V.min() for numerical stability;
            # this is just a normalization constant.
            V_safe = np.where(valid, V, 0.0)
            V_shift = V_safe - V_safe[valid].min()
            density = np.zeros_like(field, dtype=np.float64)
            density[valid] = np.exp(-V_shift[valid] / temperature)

        # Normalize density so the valid region sums to 1 (interpretable
        # as a probability density on the grid).
        total = density[valid].sum()
        if total > 0:
            density[valid] = density[valid] / total
        density[~valid] = np.nan

        parameters = {
            "affinity": affinity,
            "temperature": temperature,
            "field_kind": field_kind,
            "output_mode": output_mode,
            "nodal_emphasis": nodal_emphasis,
            "sigma": sigma,
        }
        metadata = {
            "kind": "granular_density"
            if output_mode == "density"
            else "granular_particles",
            "family": "transport",
            "source_kind": field_data.metadata.get("kind"),
        }

        if output_mode == "density":
            return GeometryData(
                geom_type="field_2d",
                coordinates=density,
                field_grid=field_grid,
                parameters=parameters,
                metadata=metadata,
            )

        # output_mode == "particles"
        if field_grid is None:
            raise ValueError(
                "Cannot sample particles without a field_grid on the "
                "source GeometryData."
            )
        rng = np.random.default_rng(seed)
        # field_grid for field_2d is (X, Y).
        X, Y = field_grid[0], field_grid[1]
        positions = _sample_particles(
            density, (X, Y), n_particles=n_particles, rng=rng
        )
        parameters["n_particles"] = int(n_particles)
        parameters["seed"] = seed
        return GeometryData(
            geom_type="point_cloud_2d",
            coordinates=positions,
            parameters=parameters,
            metadata=metadata,
        )

    def __repr__(self) -> str:
        return (
            f"Granular(affinity={self.affinity}, "
            f"temperature={self.temperature}, "
            f"field_kind={self.field_kind!r}, "
            f"output_mode={self.output_mode!r})"
        )
