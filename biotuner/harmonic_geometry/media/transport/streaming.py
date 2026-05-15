"""
Acoustic streaming — slow secondary flow driven by a wave field.

Belongs to the ``transport`` family of
:mod:`biotuner.harmonic_geometry.media`. Acoustic streaming is the
time-averaged DC flow that appears whenever a viscous fluid carries a
finite-amplitude wave: nonlinear self-interaction of the wave produces
a steady force ``F = −ρ_0 ⟨(u · ∇)u⟩`` that drives a slow secondary
flow. Two canonical regimes:

- **Rayleigh streaming** (this implementation, simplified): in a thin
  boundary layer near a vibrating surface, vortex pairs (Rayleigh
  rolls) form between adjacent antinodes. The streaming velocity
  pattern is built from the Reynolds-stress divergence of the
  primary wave field.
- **Eckart streaming**: a unidirectional bulk flow in the direction of
  wave propagation, driven by viscous attenuation. Less visually
  interesting in a 2-D standing-wave context; not currently
  implemented separately (the same operator with ``viscosity``
  parameter raised approximates the bulk limit).

Physical model
--------------
For a real-valued primary wave field ``p(x, y)`` (interpreted as
acoustic pressure or normal-mode amplitude), the slow streaming
velocity at steady state in 2-D obeys, schematically:

    u_stream = (1 / μ_eff) · (curl of the wave-induced body force)

where the body force comes from ``⟨p · ∇p⟩`` after time-averaging.
We implement this as:

    F_x ∝ −p · ∂_x p
    F_y ∝ −p · ∂_y p

and then take the **2-D curl** of ``F`` (which is a scalar in 2-D),
finally solving a Poisson equation to recover the streamfunction
``ψ``. The streaming velocity is then ``(u, v) = (-∂_y ψ, ∂_x ψ)``,
which automatically gives the closed-loop pattern characteristic of
Rayleigh rolls.

For computational simplicity the Poisson solve uses an FFT — exact
for periodic boundaries, approximate (but visually fine) for general
shapes; cells outside the valid region are masked back to NaN.

``viscosity`` scales the overall magnitude (units arbitrary). Output
is identical structurally to :class:`Tracer`: ``vector_field_2d``,
``field_2d`` (speed), or steady tracer density.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from biotuner.harmonic_geometry.geometry_data import GeometryData
from biotuner.harmonic_geometry.media.base import Medium


_VALID_OUTPUT_MODES = ("flow", "speed", "tracer_density")


def _streamfunction_from_force(
    fx: np.ndarray,
    fy: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Recover the streamfunction ψ from a 2-D body force via FFT Poisson.

    Computes the 2-D curl ``ω = ∂_x F_y − ∂_y F_x``, then solves
    ``∇² ψ = ω`` with periodic boundaries via FFT. Returns ψ.
    """
    # ω = ∂x fy − ∂y fx.
    dfy_dy, dfy_dx = np.gradient(fy, dy, dx)
    dfx_dy, dfx_dx = np.gradient(fx, dy, dx)
    omega = dfy_dx - dfx_dy

    ny, nx = omega.shape
    kx = np.fft.fftfreq(nx, d=dx) * 2.0 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2.0 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX ** 2 + KY ** 2
    # Avoid division by zero at the zero-mode (mean of ψ is arbitrary).
    K2[0, 0] = 1.0
    omega_hat = np.fft.fft2(omega)
    psi_hat = -omega_hat / K2
    psi_hat[0, 0] = 0.0
    psi = np.real(np.fft.ifft2(psi_hat))
    return psi


def _streaming_velocity(
    field: np.ndarray,
    field_grid: Optional[Tuple[np.ndarray, ...]],
    viscosity: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the streaming velocity (u, v) from a wave field."""
    valid = np.isfinite(field)
    f_safe = np.where(valid, field, 0.0).astype(np.float64)

    if field_grid is not None and len(field_grid) >= 2:
        X, Y = field_grid[0], field_grid[1]
        dx = float(X[0, 1] - X[0, 0]) if X.shape[1] > 1 else 1.0
        dy = float(Y[1, 0] - Y[0, 0]) if Y.shape[0] > 1 else 1.0
    else:
        dx = dy = 1.0

    # Wave gradient.
    gy, gx = np.gradient(f_safe, dy, dx)
    # Reynolds-stress body force F = −p · ∇p (sign sets vortex chirality).
    fx = -f_safe * gx
    fy = -f_safe * gy

    psi = _streamfunction_from_force(fx, fy, dx, dy)
    # u = -∂y ψ, v = +∂x ψ.
    dpsi_dy, dpsi_dx = np.gradient(psi, dy, dx)
    u = -dpsi_dy
    v = dpsi_dx

    # Apply viscosity scaling (smaller viscosity = stronger streaming).
    scale = 1.0 / max(viscosity, 1e-12)
    u = u * scale
    v = v * scale

    u = np.where(valid, u, np.nan)
    v = np.where(valid, v, np.nan)
    return u, v, valid


class Streaming(Medium):
    """Transport-family medium: acoustic-streaming (Rayleigh rolls).

    Parameters
    ----------
    viscosity : float, default 1.0
        Inverse magnitude scale of the streaming velocity. Lower
        viscosity gives stronger streaming for the same input wave.
    normalize : bool, default True
        Rescale the velocity field so its 99th-percentile magnitude
        is 1 (the underlying physics is amplitude-quadratic, so
        absolute magnitudes are otherwise arbitrary).
    output_mode : {"flow", "speed", "tracer_density"}, default "flow"
        See :class:`Tracer` for semantics; identical here.
    epsilon : float, default 1e-3
        Regularizer for ``tracer_density`` output.
    """

    family = "transport"

    def __init__(
        self,
        *,
        viscosity: float = 1.0,
        normalize: bool = True,
        output_mode: str = "flow",
        epsilon: float = 1e-3,
    ) -> None:
        if output_mode not in _VALID_OUTPUT_MODES:
            raise ValueError(
                f"output_mode must be one of {_VALID_OUTPUT_MODES}; got "
                f"{output_mode!r}."
            )
        if viscosity <= 0:
            raise ValueError("viscosity must be > 0.")
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0.")

        self.viscosity = float(viscosity)
        self.normalize = bool(normalize)
        self.output_mode = output_mode
        self.epsilon = float(epsilon)

    # ----------------------------------------------------------- contract

    def default_source(self) -> Medium:
        # Acoustic streaming is canonical for closed-domain
        # standing waves — Rayleigh rolls between adjacent antinodes.
        from biotuner.harmonic_geometry.media.eigenmode.rigid_plate import (
            RigidPlate,
        )

        return RigidPlate()

    def respond(self, forcing, **overrides: Any) -> GeometryData:
        viscosity = float(overrides.pop("viscosity", self.viscosity))
        normalize = bool(overrides.pop("normalize", self.normalize))
        output_mode = overrides.pop("output_mode", self.output_mode)
        epsilon = float(overrides.pop("epsilon", self.epsilon))

        if overrides:
            raise TypeError(
                f"Unexpected override keys: {sorted(overrides)}."
            )
        if output_mode not in _VALID_OUTPUT_MODES:
            raise ValueError(
                f"output_mode must be one of {_VALID_OUTPUT_MODES}."
            )
        if viscosity <= 0:
            raise ValueError("viscosity must be > 0.")

        field_data = self._resolve_field(forcing)
        if field_data.geom_type != "field_2d":
            raise TypeError(
                "Streaming expects a 2-D scalar wave field "
                "(geom_type='field_2d'); got geom_type="
                f"{field_data.geom_type!r}."
            )
        field = np.asarray(field_data.coordinates, dtype=np.float64)
        field_grid = field_data.field_grid

        u, v, valid = _streaming_velocity(field, field_grid, viscosity)

        if normalize:
            speed_arr = np.hypot(np.where(valid, u, 0.0),
                                  np.where(valid, v, 0.0))
            scale = (float(np.nanpercentile(speed_arr[valid], 99))
                     if valid.any() else 0.0)
            if scale > 0:
                u = u / scale
                v = v / scale

        parameters = {
            "viscosity": viscosity,
            "normalize": normalize,
            "output_mode": output_mode,
            "epsilon": epsilon,
        }
        metadata = {
            "kind": f"streaming_{output_mode}",
            "family": "transport",
            "source_geom_type": field_data.geom_type,
            "source_kind": (field_data.metadata or {}).get("kind", ""),
        }

        if output_mode == "flow":
            coords = np.stack([np.where(valid, u, 0.0),
                                np.where(valid, v, 0.0)], axis=-1)
            return GeometryData(
                geom_type="vector_field_2d",
                coordinates=coords,
                field_grid=field_grid,
                parameters=parameters,
                metadata=metadata,
            )

        speed_full = np.hypot(np.where(valid, u, 0.0), np.where(valid, v, 0.0))
        speed_full = np.where(valid, speed_full, np.nan)
        if output_mode == "speed":
            return GeometryData(
                geom_type="field_2d",
                coordinates=speed_full,
                field_grid=field_grid,
                parameters=parameters,
                metadata=metadata,
            )

        density = np.full_like(field, np.nan, dtype=np.float64)
        s_in = speed_full[valid]
        density_in = 1.0 / (s_in + epsilon)
        d_max = float(np.nanmax(density_in)) if density_in.size else 1.0
        if d_max > 0:
            density_in = density_in / d_max
        density[valid] = density_in
        return GeometryData(
            geom_type="field_2d",
            coordinates=density,
            field_grid=field_grid,
            parameters=parameters,
            metadata=metadata,
        )

    def __repr__(self) -> str:
        return (
            f"Streaming(viscosity={self.viscosity}, "
            f"output_mode={self.output_mode!r})"
        )
