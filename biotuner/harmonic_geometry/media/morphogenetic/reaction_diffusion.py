"""
Reaction–diffusion patterns (Gray–Scott).

Belongs to the ``morphogenetic`` family of
:mod:`biotuner.harmonic_geometry.media`. The Gray–Scott model describes
two interacting chemicals ``U`` and ``V`` undergoing the reaction
``U + 2V → 3V`` with feed/kill terms, on a 2-D diffusing substrate:

    ∂U/∂t = D_u · ∇²U − U·V² + F · (1 − U)
    ∂V/∂t = D_v · ∇²V + U·V² − (F + K) · V

The two scalar parameters ``F`` (feed) and ``K`` (kill) trace out a
rich phase diagram of stationary and travelling patterns: spots,
stripes, labyrinths, solitons, mitosis-like spot division, etc. See
Pearson, Science 261 (1993) for the canonical regime map.

Chord coupling
--------------
With ``feed`` / ``kill`` set to ``None`` (default), the chord drives
both from *structural* descriptors of the ratio set (not just scalar
reductions), so that chords with similar overall consonance still land
in qualitatively different Pearson regimes:

- ``feed``  F ← ``structure.prime_limit``        in ``[0.020, 0.062]``
            (low primes give the spotty / mitosis side;
             high primes give the stripe / replicating side).
- ``kill``  K ← ``log(structure.max_common_int)``
            in ``[0.045, 0.065]``  — chords with larger common-
            denominator integers (e.g. 11:7:5 → 200) get a higher K,
            pushing them into stripe / labyrinth territory; chords
            with smaller ones (Major = 4:5:6 → 6) get a lower K.
- ``diffusion_ratio`` (D_u / D_v) ← ``structure.max_common_int``
            in ``[1.5, 3.0]``  — relative spatial scales.

Initial conditions
------------------
``seed_strategy`` selects the trigger pattern:

- ``"single"``   small central V-blob — produces a single growing
                 pattern that fills the domain.
- ``"polygon"``  V-blobs planted at the chord's Tonnetz polygon
                 vertices — chord-structured initial condition
                 (default). Same idea as Crystallization's polygon
                 seed: different chord = different starting
                 configuration = visibly different attractor reached.
- ``"random"``   stochastic V-seeding (uniform density).

Outputs
-------
``output_mode='v'``  — V concentration field (``field_2d``); typically
the most visually striking channel.
``output_mode='u'``  — U concentration field.
``output_mode='difference'`` — ``V − U`` (signed; useful for nodal-
line-style rendering).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from biotuner.harmonic_geometry.geometry_data import GeometryData
from biotuner.harmonic_geometry.inputs import HarmonicInput
from biotuner.harmonic_geometry.media import coupling as _coupling
from biotuner.harmonic_geometry.media import structure as _structure
from biotuner.harmonic_geometry.media.base import Medium


_VALID_OUTPUT_MODES = ("u", "v", "difference")
_VALID_SEED_STRATEGIES = ("single", "polygon", "random")


def _laplacian5(field: np.ndarray) -> np.ndarray:
    """5-point Laplacian with periodic boundaries (unit spacing)."""
    return (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
        - 4.0 * field
    )


def _initial_condition(
    resolution: int,
    chord: HarmonicInput,
    seed_strategy: str,
    seed_scale: float,
    seed_radius_cells: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Build initial ``(U, V)`` fields at ``resolution × resolution``.

    Background: ``U = 1``, ``V = 0`` (the "no-pattern" steady state).
    Trigger blobs flip a local patch to ``U ≈ 0.5``, ``V ≈ 0.25``.
    """
    N = int(resolution)
    U = np.ones((N, N), dtype=np.float64)
    V = np.zeros((N, N), dtype=np.float64)

    def _plant(cx: int, cy: int, radius: int) -> None:
        r = max(int(radius), 1)
        xs = np.arange(-r, r + 1)
        ys = np.arange(-r, r + 1)
        XX, YY = np.meshgrid(xs, ys, indexing="ij")
        disk = (XX ** 2 + YY ** 2) <= r * r
        # Add a small noise so symmetry can break.
        for di, dj in zip(*np.where(disk)):
            ii = (cx + xs[di] + r) % N
            jj = (cy + ys[dj] + r) % N
            # Recompute integer offsets robustly.
        # Simpler: just patch using slicing with wrap.
        x_lo = (cx - r) % N
        y_lo = (cy - r) % N
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                if i * i + j * j <= r * r:
                    ii = (cx + i) % N
                    jj = (cy + j) % N
                    U[ii, jj] = 0.5 + 0.02 * rng.normal()
                    V[ii, jj] = 0.25 + 0.02 * rng.normal()

    if seed_strategy == "single":
        _plant(N // 2, N // 2, seed_radius_cells)
    elif seed_strategy == "polygon":
        polygon = _structure.tonnetz_polygon(chord)
        if not polygon:
            _plant(N // 2, N // 2, seed_radius_cells)
        else:
            cx = sum(x for x, _ in polygon) / len(polygon)
            cy = sum(y for _, y in polygon) / len(polygon)
            centered = [(x - cx, y - cy) for x, y in polygon]
            max_r = max(np.hypot(x, y) for x, y in centered)
            if max_r < 1e-6:
                _plant(N // 2, N // 2, seed_radius_cells)
            else:
                # Scale to seed_scale * (N/2) pixels.
                pix_scale = seed_scale * (N / 2) / max_r
                # Always plant centroid as the anchor.
                _plant(N // 2, N // 2, seed_radius_cells)
                for x, y in centered:
                    ci = int(N // 2 + round(x * pix_scale))
                    cj = int(N // 2 + round(y * pix_scale))
                    _plant(ci, cj, seed_radius_cells)
    elif seed_strategy == "random":
        # 1.5% area randomly seeded.
        n_seeds = max(int(0.015 * N * N / (np.pi * seed_radius_cells ** 2)), 1)
        for _ in range(n_seeds):
            ci = int(rng.integers(0, N))
            cj = int(rng.integers(0, N))
            _plant(ci, cj, seed_radius_cells)
    else:  # pragma: no cover - validated earlier
        raise ValueError(seed_strategy)

    return U, V


def _gray_scott_step(
    U: np.ndarray,
    V: np.ndarray,
    Du: float,
    Dv: float,
    F: float,
    K: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One forward-Euler step of Gray–Scott."""
    Lu = _laplacian5(U)
    Lv = _laplacian5(V)
    uvv = U * V * V
    U_new = U + dt * (Du * Lu - uvv + F * (1.0 - U))
    V_new = V + dt * (Dv * Lv + uvv - (F + K) * V)
    return U_new, V_new


class ReactionDiffusion(Medium):
    """Morphogenetic-family medium: Gray–Scott reaction–diffusion.

    Parameters
    ----------
    feed : float, optional
        Feed rate F. If ``None`` (default), derived from
        :func:`structure.prime_limit` mapped into ``[0.020, 0.062]`` —
        3-limit (Pythagorean) chords sit low (mitosis / spots side),
        11-limit chords sit high (stripes / replicating side).
        Typical Pearson regimes: F=0.022/K=0.051 → "mitosis spots",
        F=0.034/K=0.061 → "alpha labyrinth", F=0.062/K=0.061 → stripes.
    kill : float, optional
        Kill rate K. If ``None`` (default), derived from
        :func:`structure.max_common_int` (log-scaled) mapped into
        ``[0.045, 0.065]``.
    diffusion_u : float, default 0.16
        Diffusion coefficient of ``U``.
    diffusion_v : float, default 0.08
        Diffusion coefficient of ``V``. The ratio ``Du/Dv`` controls
        relative spatial scales (must be > 1 for Turing instability).
    dt : float, default 1.0
        Forward-Euler timestep.
    n_steps : int, default 4000
        Number of integration steps.
    resolution : int, default 192
        Grid resolution; the simulation domain is square with periodic
        boundaries.
    extent : float, default 1.0
        Half-side of the rendering extent (for the meshgrid).
    seed_strategy : {"single", "polygon", "random"}, default "polygon"
        Initial-condition pattern (see module docstring).
    seed_scale : float, default 0.30
        Radius (fraction of half-grid) at which polygon-vertex seeds
        are placed.
    seed_radius_cells : int, default 4
        Radius (in cells) of each trigger blob.
    rng_seed : int, optional
        Seed for noise in the initial condition; defaults to a
        deterministic per-chord seed.
    output_mode : {"u", "v", "difference"}, default "v"
        Which channel to return.
    """

    family = "morphogenetic"

    def __init__(
        self,
        *,
        feed: Optional[float] = None,
        kill: Optional[float] = None,
        diffusion_u: float = 0.16,
        diffusion_v: float = 0.08,
        dt: float = 1.0,
        n_steps: int = 4000,
        resolution: int = 192,
        extent: float = 1.0,
        seed_strategy: str = "polygon",
        seed_scale: float = 0.30,
        seed_radius_cells: int = 4,
        rng_seed: Optional[int] = None,
        output_mode: str = "v",
    ) -> None:
        if output_mode not in _VALID_OUTPUT_MODES:
            raise ValueError(
                f"output_mode must be one of {_VALID_OUTPUT_MODES}; "
                f"got {output_mode!r}."
            )
        if seed_strategy not in _VALID_SEED_STRATEGIES:
            raise ValueError(
                f"seed_strategy must be one of {_VALID_SEED_STRATEGIES}; "
                f"got {seed_strategy!r}."
            )
        if feed is not None and not (0.0 < feed < 0.2):
            raise ValueError("feed must be in (0, 0.2) when given.")
        if kill is not None and not (0.0 < kill < 0.2):
            raise ValueError("kill must be in (0, 0.2) when given.")
        if diffusion_u <= 0 or diffusion_v <= 0:
            raise ValueError("diffusion coefficients must be > 0.")
        if dt <= 0:
            raise ValueError("dt must be > 0.")
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1.")
        if resolution < 32:
            raise ValueError("resolution must be >= 32.")
        if extent <= 0:
            raise ValueError("extent must be > 0.")
        if not (0.0 < seed_scale < 1.0):
            raise ValueError("seed_scale must be in (0, 1).")
        if seed_radius_cells < 1:
            raise ValueError("seed_radius_cells must be >= 1.")

        self.feed = feed
        self.kill = kill
        self.diffusion_u = float(diffusion_u)
        self.diffusion_v = float(diffusion_v)
        self.dt = float(dt)
        self.n_steps = int(n_steps)
        self.resolution = int(resolution)
        self.extent = float(extent)
        self.seed_strategy = seed_strategy
        self.seed_scale = float(seed_scale)
        self.seed_radius_cells = int(seed_radius_cells)
        self.rng_seed = rng_seed
        self.output_mode = output_mode

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
                "ReactionDiffusion.respond requires a HarmonicInput; got "
                f"{type(forcing).__name__}."
            )

        feed_arg = overrides.pop("feed", self.feed)
        kill_arg = overrides.pop("kill", self.kill)
        Du = float(overrides.pop("diffusion_u", self.diffusion_u))
        Dv = float(overrides.pop("diffusion_v", self.diffusion_v))
        dt = float(overrides.pop("dt", self.dt))
        n_steps = int(overrides.pop("n_steps", self.n_steps))
        resolution = int(overrides.pop("resolution", self.resolution))
        extent = float(overrides.pop("extent", self.extent))
        seed_strategy = overrides.pop("seed_strategy", self.seed_strategy)
        seed_scale = float(overrides.pop("seed_scale", self.seed_scale))
        seed_radius_cells = int(overrides.pop(
            "seed_radius_cells", self.seed_radius_cells))
        rng_seed = overrides.pop("rng_seed", self.rng_seed)
        output_mode = overrides.pop("output_mode", self.output_mode)

        if overrides:
            raise TypeError(
                f"Unexpected override keys: {sorted(overrides)}."
            )
        if output_mode not in _VALID_OUTPUT_MODES:
            raise ValueError(
                f"output_mode must be one of {_VALID_OUTPUT_MODES}."
            )
        if seed_strategy not in _VALID_SEED_STRATEGIES:
            raise ValueError(
                f"seed_strategy must be one of {_VALID_SEED_STRATEGIES}."
            )

        # ------ chord-driven defaults (from structural descriptors)
        plim = _structure.prime_limit(forcing)
        mci = _structure.max_common_int(forcing)
        if feed_arg is None:
            # prime_limit 2 → 0.020 (degenerate); 3 → 0.024;
            # 5 → 0.032; 7 → 0.040; 11 → 0.056; 13 → 0.062.
            feed_arg = 0.020 + min(max(plim - 2, 0), 10) / 10.0 * (0.062 - 0.020)
        if kill_arg is None:
            # Log-scaled max_common_int so the 6 → 200 range stretches
            # reasonably: mci=6 → K≈0.048; mci=15 → K≈0.053; mci=60 → K≈0.060;
            # mci=200 → K≈0.065.
            t = min(np.log(max(mci, 2)) / np.log(200.0), 1.0)
            kill_arg = 0.045 + t * (0.065 - 0.045)
        F = float(feed_arg)
        K = float(kill_arg)

        # Diffusion ratio from max_common_int (varies relative spatial scale).
        ratio_target = 1.5 + min(max(mci - 3, 0) / 60.0, 1.0) * (3.0 - 1.5)
        Dv = Du / ratio_target

        # ------ initial conditions
        rng = (np.random.default_rng(int(rng_seed)) if rng_seed is not None
               else np.random.default_rng(
                    int(hash(tuple(forcing.to_ratios())) % (2 ** 31))))
        U, V = _initial_condition(
            resolution, forcing, seed_strategy,
            seed_scale, seed_radius_cells, rng,
        )

        # ------ integrate
        for _ in range(n_steps):
            U, V = _gray_scott_step(U, V, Du, Dv, F, K, dt)

        # ------ output channel
        if output_mode == "u":
            field = U
        elif output_mode == "v":
            field = V
        else:
            field = V - U

        x = np.linspace(-extent, extent, resolution)
        X, Y = np.meshgrid(x, x, indexing="xy")

        parameters = {
            "feed": F,
            "kill": K,
            "diffusion_u": Du,
            "diffusion_v": Dv,
            "diffusion_ratio": Du / Dv,
            "dt": dt,
            "n_steps": n_steps,
            "resolution": resolution,
            "extent": extent,
            "seed_strategy": seed_strategy,
            "seed_scale": seed_scale,
            "seed_radius_cells": seed_radius_cells,
            "rng_seed": rng_seed,
            "output_mode": output_mode,
        }
        metadata = {
            "kind": f"reaction_diffusion_{output_mode}",
            "family": "morphogenetic",
            "u_range": (float(U.min()), float(U.max())),
            "v_range": (float(V.min()), float(V.max())),
        }

        return GeometryData(
            geom_type="field_2d",
            coordinates=field,
            field_grid=(X, Y),
            parameters=parameters,
            metadata=metadata,
        )

    def __repr__(self) -> str:
        return (
            f"ReactionDiffusion(feed={self.feed}, kill={self.kill}, "
            f"seed_strategy={self.seed_strategy!r}, "
            f"output_mode={self.output_mode!r})"
        )
