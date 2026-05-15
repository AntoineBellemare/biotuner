"""
Passive-tracer advection on a wave field.

Belongs to the ``transport`` family of
:mod:`biotuner.harmonic_geometry.media`. Where :class:`Granular` outputs
*where particles end up at equilibrium*, :class:`Tracer` outputs *how
they move*: it returns the velocity field that a passive scalar would
experience when floating on the wave field, or — equivalently — the
streamlines along which it drifts.

Three flow regimes are exposed via ``flow_kind``:

- ``"gradient"`` — ``u = ∇φ``. Particles flow toward maxima (descent
  toward minima of ``-φ``). Sources and sinks at antinodes/nodes.
- ``"curl"`` (default) — ``u = (-∂_y φ, ∂_x φ)``. Particles flow
  *along* level curves of the field. Produces closed loops around
  every maximum and saddle-driven separatrices between them — the
  most visually rich regime, akin to streamlines around a topographic
  map.
- ``"mixed"`` — weighted blend ``u = γ ∇φ + (1−γ) curl(φ)``. The
  ``mixing`` parameter (0…1) selects the blend; useful for animating
  between the two.

Output modes
------------
``"flow"`` (default) returns the velocity field as ``vector_field_2d``;
``"speed"`` returns the scalar speed ``|u|`` as ``field_2d`` (slow
regions = where a tracer would accumulate at steady state); ``"tracer_density"``
returns the steady-state passive-scalar density ``1 / (|u| + ε)``
normalized over the domain, also as ``field_2d``.

Defaults to wrapping a HarmonicInput through :class:`Interference` (an
open-medium wave_field source). For closed-domain modes (Chladni),
pre-compute with :class:`RigidPlate` and pass that GeometryData in.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from biotuner.harmonic_geometry.geometry_data import GeometryData
from biotuner.harmonic_geometry.media.base import Medium


_VALID_FLOW_KINDS = ("gradient", "curl", "mixed")
_VALID_OUTPUT_MODES = ("flow", "speed", "tracer_density")


def _flow_from_field(
    field: np.ndarray,
    field_grid: Optional[Tuple[np.ndarray, ...]],
    flow_kind: str,
    mixing: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a velocity field from a scalar field.

    Returns
    -------
    u, v : ndarrays of same shape as ``field``
        Velocity components.
    valid : bool ndarray
        Cells where the gradient is finite (NaN cells propagated to
        the edge of the valid region — see below).
    """
    valid = np.isfinite(field)
    f_safe = np.where(valid, field, 0.0).astype(np.float64)
    if field_grid is not None and len(field_grid) >= 2:
        X, Y = field_grid[0], field_grid[1]
        dx = float(X[0, 1] - X[0, 0]) if X.shape[1] > 1 else 1.0
        dy = float(Y[1, 0] - Y[0, 0]) if Y.shape[0] > 1 else 1.0
    else:
        dx = dy = 1.0
    gy, gx = np.gradient(f_safe, dy, dx)

    if flow_kind == "gradient":
        u, v = gx, gy
    elif flow_kind == "curl":
        # 2-D curl of a scalar potential = (−∂y, +∂x). Trajectories
        # follow level curves of ``field``.
        u, v = -gy, gx
    elif flow_kind == "mixed":
        m = float(np.clip(mixing, 0.0, 1.0))
        u = m * gx + (1.0 - m) * (-gy)
        v = m * gy + (1.0 - m) * gx
    else:
        raise ValueError(
            f"flow_kind must be one of {_VALID_FLOW_KINDS}; got {flow_kind!r}."
        )

    # Mask out cells that were NaN in the input.
    u = np.where(valid, u, np.nan)
    v = np.where(valid, v, np.nan)
    return u, v, valid


class Tracer(Medium):
    """Transport-family medium: passive scalar advection on a wave field.

    Parameters
    ----------
    flow_kind : {"gradient", "curl", "mixed"}, default "curl"
        How to derive a velocity field from the input scalar field.
        See module docstring.
    mixing : float, default 0.5
        Used only for ``flow_kind="mixed"``: weight of the gradient
        component vs. the curl component, in ``[0, 1]``.
    normalize : bool, default True
        If True, scale the velocity field so its 99th-percentile
        magnitude is 1. Useful when the input field's amplitude is
        arbitrary.
    output_mode : {"flow", "speed", "tracer_density"}, default "flow"
        ``"flow"`` returns ``vector_field_2d``;
        ``"speed"`` returns ``field_2d`` of ``|u|``;
        ``"tracer_density"`` returns ``field_2d`` of
        ``1 / (|u| + epsilon)`` normalized to ``[0, 1]``.
    epsilon : float, default 1e-3
        Regularizer in the tracer-density denominator (relative to the
        normalized speed scale).
    """

    family = "transport"

    def __init__(
        self,
        *,
        flow_kind: str = "curl",
        mixing: float = 0.5,
        normalize: bool = True,
        output_mode: str = "flow",
        epsilon: float = 1e-3,
    ) -> None:
        if flow_kind not in _VALID_FLOW_KINDS:
            raise ValueError(
                f"flow_kind must be one of {_VALID_FLOW_KINDS}; got "
                f"{flow_kind!r}."
            )
        if output_mode not in _VALID_OUTPUT_MODES:
            raise ValueError(
                f"output_mode must be one of {_VALID_OUTPUT_MODES}; got "
                f"{output_mode!r}."
            )
        if not (0.0 <= mixing <= 1.0):
            raise ValueError("mixing must be in [0, 1].")
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0.")

        self.flow_kind = flow_kind
        self.mixing = float(mixing)
        self.normalize = bool(normalize)
        self.output_mode = output_mode
        self.epsilon = float(epsilon)

    # ----------------------------------------------------------- contract

    def default_source(self) -> Medium:
        # Open-medium wave_field is the natural pairing — Tracer
        # produces the streamlines that swirl around a multi-source
        # interference pattern.
        from biotuner.harmonic_geometry.media.wave_field.interference import (
            Interference,
        )

        return Interference()

    def respond(
        self,
        forcing,
        **overrides: Any,
    ) -> GeometryData:
        flow_kind = overrides.pop("flow_kind", self.flow_kind)
        mixing = float(overrides.pop("mixing", self.mixing))
        normalize = bool(overrides.pop("normalize", self.normalize))
        output_mode = overrides.pop("output_mode", self.output_mode)
        epsilon = float(overrides.pop("epsilon", self.epsilon))

        if overrides:
            raise TypeError(
                f"Unexpected override keys: {sorted(overrides)}."
            )
        if flow_kind not in _VALID_FLOW_KINDS:
            raise ValueError(f"flow_kind must be one of {_VALID_FLOW_KINDS}.")
        if output_mode not in _VALID_OUTPUT_MODES:
            raise ValueError(
                f"output_mode must be one of {_VALID_OUTPUT_MODES}."
            )

        field_data = self._resolve_field(forcing)
        if field_data.geom_type != "field_2d":
            raise TypeError(
                "Tracer expects a 2-D scalar wave field "
                "(geom_type='field_2d'); got geom_type="
                f"{field_data.geom_type!r}."
            )

        field = np.asarray(field_data.coordinates, dtype=np.float64)
        field_grid = field_data.field_grid

        u, v, valid = _flow_from_field(field, field_grid, flow_kind, mixing)

        # Optional normalization of the magnitude scale.
        if normalize:
            speed = np.hypot(np.where(valid, u, 0.0), np.where(valid, v, 0.0))
            scale = float(np.nanpercentile(speed[valid], 99)) if valid.any() else 0.0
            if scale > 0:
                u = u / scale
                v = v / scale

        parameters = {
            "flow_kind": flow_kind,
            "mixing": mixing,
            "normalize": normalize,
            "output_mode": output_mode,
            "epsilon": epsilon,
        }
        metadata = {
            "kind": f"tracer_{output_mode}",
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

        # Both "speed" and "tracer_density" want a scalar field.
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

        # tracer_density: 1 / (|u| + ε), normalized to [0, 1].
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
            f"Tracer(flow_kind={self.flow_kind!r}, "
            f"output_mode={self.output_mode!r})"
        )
