"""
Coulomb / dusty-plasma crystal in a standing-wave potential.

Belongs to the ``eigenmode`` family of
:mod:`biotuner.harmonic_geometry.media`. Where :class:`RigidPlate`
returns a *continuous* eigenmode field, :class:`PlasmaLattice` returns
the *discrete* equilibrium of N mutually-repelling charged particles
sitting in a confining potential that is itself shaped by the chord's
eigenmode field — a Coulomb crystal trapped in a standing-wave
landscape.

Physics
-------
N point "ions" interact via repulsive ``1/r`` Coulomb pairs and feel an
external potential

    V_ext(x, y) = V_trap(x, y) + lambda * U_chord(x, y)

where ``V_trap`` is a parabolic harmonic trap (radius ``trap_radius``)
that keeps the crystal from flying apart, ``U_chord`` is the chord's
eigenmode amplitude (from :class:`RigidPlate`), and ``lambda`` is
the ``modulation_strength``. The equilibrium configuration is reached
by gradient descent on the total energy, with cooling.

Output
------
Returns ``point_cloud_2d`` with the equilibrium ion positions
(``coordinates`` is ``(N, 2)``). The metadata records the per-ion
potential energy and pair distances.

Chord coupling
--------------
With ``n_ions`` set to ``None`` (default), it is derived from
``structure.prime_limit`` mapped into ``[3, 31]`` — chords with higher
prime limit get more ions, so the crystal is denser and the
chord-modulated potential has more competing minima to populate.
Other parameters can be chord-driven by the user via overrides.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from biotuner.harmonic_geometry.geometry_data import GeometryData
from biotuner.harmonic_geometry.inputs import HarmonicInput
from biotuner.harmonic_geometry.media import structure as _structure
from biotuner.harmonic_geometry.media.base import Medium, Rectangular
from biotuner.harmonic_geometry.media.eigenmode.rigid_plate import RigidPlate


def _initial_positions(
    n: int, trap_radius: float, rng: np.random.Generator
) -> np.ndarray:
    """Initial ion positions on a small sub-grid of the trap."""
    # Hexagonal close-packed estimate sized to fit n.
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    xs = np.linspace(-0.5 * trap_radius, 0.5 * trap_radius, cols)
    ys = np.linspace(-0.5 * trap_radius, 0.5 * trap_radius, rows)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)[:n]
    # Add small jitter so identical-pair pathologies don't occur.
    pts = pts + rng.normal(scale=trap_radius * 0.01, size=pts.shape)
    return pts


def _interp_field(
    field: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the chord-modulation field and its gradient at points ``pts``.

    Uses bilinear interpolation on a regular grid.
    """
    x_lo = float(X[0, 0])
    y_lo = float(Y[0, 0])
    dx = float(X[0, 1] - X[0, 0])
    dy = float(Y[1, 0] - Y[0, 0])
    H, W = field.shape

    # Fractional grid indices.
    gx = (pts[:, 0] - x_lo) / dx
    gy = (pts[:, 1] - y_lo) / dy
    # Clamp to interior.
    gx = np.clip(gx, 0.0, W - 1.001)
    gy = np.clip(gy, 0.0, H - 1.001)
    i0 = gx.astype(int)
    j0 = gy.astype(int)
    fx = gx - i0
    fy = gy - j0

    # Bilinear value.
    F = field
    v00 = F[j0, i0]
    v01 = F[j0, i0 + 1]
    v10 = F[j0 + 1, i0]
    v11 = F[j0 + 1, i0 + 1]
    v = ((1 - fx) * (1 - fy) * v00 + fx * (1 - fy) * v01
         + (1 - fx) * fy * v10 + fx * fy * v11)

    # Gradient (central differences from the corner samples).
    grad_x = ((v01 - v00) * (1 - fy) + (v11 - v10) * fy) / dx
    grad_y = ((v10 - v00) * (1 - fx) + (v11 - v01) * fx) / dy
    return v, grad_x, grad_y


def _relax_ions(
    pts: np.ndarray,
    chord_field: np.ndarray,
    chord_X: np.ndarray,
    chord_Y: np.ndarray,
    trap_radius: float,
    modulation_strength: float,
    coulomb_strength: float,
    n_steps: int,
    learning_rate: float,
    cooling: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Gradient-descent / Langevin relaxation of N ions in the potential."""
    pts = pts.copy()
    lr = learning_rate
    for step in range(n_steps):
        # Pairwise Coulomb forces.
        n = len(pts)
        # Vectorized: (N, N, 2) diffs and (N, N) distances.
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = (diff ** 2).sum(axis=-1) + 1e-8
        # Force ~ +1/r^2 along (i->j); contribution on i is +d_ij / |d|^3
        inv_d3 = 1.0 / (d2 ** 1.5)
        # Zero out self-interaction.
        np.fill_diagonal(inv_d3, 0.0)
        force_coulomb = coulomb_strength * (diff * inv_d3[:, :, None]).sum(axis=1)

        # Harmonic trap pulling toward origin: F_trap = -k * r.
        # k chosen so equilibrium scale matches trap_radius.
        force_trap = -pts / (trap_radius ** 2)

        # Chord modulation gradient (descend the potential).
        _, gx, gy = _interp_field(chord_field, chord_X, chord_Y, pts)
        force_chord = -modulation_strength * np.stack([gx, gy], axis=1)

        total_force = force_coulomb + force_trap + force_chord
        # Add small thermal noise that anneals with `cooling`.
        T = max(1.0 - cooling * step / max(n_steps - 1, 1), 0.0)
        noise = rng.normal(scale=lr * T * 0.05, size=pts.shape)
        pts = pts + lr * total_force + noise

    return pts


class PlasmaLattice(Medium):
    """Eigenmode-family medium: discrete Coulomb crystal in a chord-shaped potential.

    Parameters
    ----------
    n_ions : int, optional
        Number of ions. If ``None`` (default), derived from
        :func:`structure.prime_limit` mapped into ``[6, 32]``.
    trap_radius : float, default 0.8
        Characteristic radius of the harmonic confining trap.
    modulation_strength : float, default 0.8
        Weight ``lambda`` of the chord-eigenmode contribution to the
        external potential.
    coulomb_strength : float, default 0.02
        Strength of the pairwise Coulomb repulsion.
    n_steps : int, default 600
        Gradient-descent steps.
    learning_rate : float, default 0.02
        Step size for the integrator.
    cooling : float, default 1.0
        Linear annealing rate (1.0 = full cooling by the final step).
    domain : Rectangular, optional
        Domain in which the chord-eigenmode field is computed
        (defaults to a square matching ``trap_radius``).
    chord_resolution : int, default 192
        Resolution for the chord-eigenmode field.
    rng_seed : int, optional
        Seed for the noise RNG; defaults to a deterministic per-chord
        hash.
    """

    family = "eigenmode"

    def __init__(
        self,
        *,
        n_ions: Optional[int] = None,
        trap_radius: float = 0.8,
        modulation_strength: float = 0.8,
        coulomb_strength: float = 0.02,
        n_steps: int = 600,
        learning_rate: float = 0.02,
        cooling: float = 1.0,
        domain: Optional[Rectangular] = None,
        chord_resolution: int = 192,
        rng_seed: Optional[int] = None,
    ) -> None:
        if n_ions is not None and n_ions < 2:
            raise ValueError("n_ions must be >= 2 when given.")
        if trap_radius <= 0:
            raise ValueError("trap_radius must be > 0.")
        if modulation_strength < 0:
            raise ValueError("modulation_strength must be >= 0.")
        if coulomb_strength < 0:
            raise ValueError("coulomb_strength must be >= 0.")
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if cooling < 0:
            raise ValueError("cooling must be >= 0.")
        if chord_resolution < 32:
            raise ValueError("chord_resolution must be >= 32.")
        if domain is None:
            # Square sized to fit the trap with a small margin.
            domain = Rectangular(2.0 * trap_radius, 2.0 * trap_radius)
        if not isinstance(domain, Rectangular):
            raise TypeError("PlasmaLattice expects a Rectangular domain.")

        self.n_ions = n_ions
        self.trap_radius = float(trap_radius)
        self.modulation_strength = float(modulation_strength)
        self.coulomb_strength = float(coulomb_strength)
        self.n_steps = int(n_steps)
        self.learning_rate = float(learning_rate)
        self.cooling = float(cooling)
        self.domain = domain
        self.chord_resolution = int(chord_resolution)
        self.rng_seed = rng_seed

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
                "PlasmaLattice.respond requires a HarmonicInput; got "
                f"{type(forcing).__name__}."
            )

        n_ions_arg = overrides.pop("n_ions", self.n_ions)
        trap_radius = float(overrides.pop("trap_radius", self.trap_radius))
        modulation_strength = float(overrides.pop(
            "modulation_strength", self.modulation_strength))
        coulomb_strength = float(overrides.pop(
            "coulomb_strength", self.coulomb_strength))
        n_steps = int(overrides.pop("n_steps", self.n_steps))
        learning_rate = float(overrides.pop(
            "learning_rate", self.learning_rate))
        cooling = float(overrides.pop("cooling", self.cooling))
        chord_resolution = int(overrides.pop(
            "chord_resolution", self.chord_resolution))
        rng_seed = overrides.pop("rng_seed", self.rng_seed)

        if overrides:
            raise TypeError(
                f"Unexpected override keys: {sorted(overrides)}."
            )

        # ----- chord-driven n_ions
        if n_ions_arg is None:
            plim = _structure.prime_limit(forcing)
            # Map prime_limit 2..13+ to n_ions 6..32 monotonically.
            n_ions_arg = int(np.clip(6 + (plim - 2) * 2.5, 6, 32))
        n_ions = int(n_ions_arg)

        # ----- chord-eigenmode source (use RigidPlate on the domain)
        rigid = RigidPlate(domain=self.domain,
                           resolution=chord_resolution)
        plate = rigid.respond(forcing)
        chord_field = np.nan_to_num(plate.coordinates, nan=0.0)
        chord_X, chord_Y = plate.field_grid

        # The plate domain is [0, Lx] x [0, Ly]; shift to centered
        # coords so it aligns with the trap (centered at origin).
        Lx = self.domain.Lx
        Ly = self.domain.Ly
        chord_X = chord_X - 0.5 * Lx
        chord_Y = chord_Y - 0.5 * Ly

        # ----- relax ions
        seed = (int(rng_seed) if rng_seed is not None
                else int(hash(tuple(forcing.to_ratios())) % (2 ** 31)))
        rng = np.random.default_rng(seed)
        pts0 = _initial_positions(n_ions, trap_radius, rng)
        pts = _relax_ions(
            pts0, chord_field, chord_X, chord_Y,
            trap_radius, modulation_strength, coulomb_strength,
            n_steps, learning_rate, cooling, rng,
        )

        # ----- per-ion potential energy at equilibrium
        v_chord, _, _ = _interp_field(chord_field, chord_X, chord_Y, pts)
        v_trap = (pts ** 2).sum(axis=1) / (2.0 * trap_radius ** 2)
        v_ext = v_trap + modulation_strength * v_chord
        # Pair energies.
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        pair_energy = coulomb_strength * (1.0 / d).sum(axis=1) * 0.5

        parameters = {
            "n_ions": n_ions,
            "trap_radius": trap_radius,
            "modulation_strength": modulation_strength,
            "coulomb_strength": coulomb_strength,
            "n_steps": n_steps,
            "learning_rate": learning_rate,
            "cooling": cooling,
            "domain_kind": "rectangular",
            "Lx": Lx,
            "Ly": Ly,
            "chord_resolution": chord_resolution,
            "rng_seed": seed,
        }
        metadata = {
            "kind": "plasma_lattice",
            "family": "eigenmode",
            "ion_potential_energy": v_ext,
            "ion_pair_energy": pair_energy,
            "min_pair_distance": float(d[np.isfinite(d)].min()),
            # Carry the underlying chord field so the renderer can
            # underlay it for context.
            "chord_field": chord_field,
            "chord_field_extent": (-0.5 * Lx, 0.5 * Lx,
                                    -0.5 * Ly, 0.5 * Ly),
        }
        return GeometryData(
            geom_type="point_cloud_2d",
            coordinates=pts,
            parameters=parameters,
            metadata=metadata,
        )

    def __repr__(self) -> str:
        return (
            f"PlasmaLattice(n_ions={self.n_ions}, "
            f"trap_radius={self.trap_radius}, "
            f"modulation_strength={self.modulation_strength})"
        )
