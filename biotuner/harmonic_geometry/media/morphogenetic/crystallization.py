"""
Reiter cellular-automaton snowflake — chord-shaped crystal growth.

Belongs to the ``morphogenetic`` family of
:mod:`biotuner.harmonic_geometry.media`: pattern grows through a
nonlinear update rule on a hexagonal grid rather than being read off as
a snapshot of a wave. The Reiter (2005) snowflake CA produces realistic
dendritic ice morphology from a small set of scalar parameters; here we
extend it with chord-driven anisotropy, asymmetry, and noise so the same
input chord can crystallize into materially different snowflakes by
varying the medium's parameters.

Algorithm
---------
Each hex cell ``i`` carries a scalar water content ``u_i``; cells with
``u_i ≥ 1`` are *frozen*. A cell is *receptive* if it is frozen or has
a frozen neighbor. Each step splits ``u_i`` into a receptive part
``v_i`` and a non-receptive part ``w_i``, then updates:

- ``v_i(t+1) = v_i(t) + γ(i)`` for receptive cells (ice phase gains
  humidity — optionally modulated per-cell by ``anisotropy_bias`` and
  by ``noise_temperature``)
- ``w_i(t+1) = w_i(t) + (α/2) · (mean_neighbors(w) − w_i(t))`` (diffusion)
- ``u_i(t+1) = v_i(t+1) + w_i(t+1)``

This is the original Reiter rule, [1]_, with 6-fold neighbor topology
on the hex grid.

Diversity controls
------------------
- ``humidity`` (γ) — background gain per step. Higher gives plumper
  shapes; lower gives sharper dendrites.
- ``diffusion`` (α) — non-receptive smoothing. Higher smooths the
  pattern; lower gives crisp finger growth.
- ``target_fill`` — stop when the fraction of frozen cells reaches this
  value (default ``0.20``). Keeps crystals from saturating into a hex
  blob; without this, dendrites grow until they hit the boundary and
  all chords end up looking the same.
- ``anisotropy_sectors`` — base symmetry of the angular humidity bias
  (3, 4, 5, 6, 8, 12, ...). Chord-driven by default from
  :func:`~biotuner.harmonic_geometry.media.coupling.ratio_complexity`:
  low-complexity chords (consonant) give clean 6-fold snowflakes;
  high-complexity chords give 3-, 5-, or 8-fold dendritic shapes.
- ``anisotropy_strength`` — magnitude of the angular humidity boost.
- ``anisotropy_kernel_width`` — angular width (radians) of each
  Gaussian bump in the bias. Narrower bumps give sharper, more
  dendritic rays.
- ``asymmetry`` — deterministic per-sector phase wobble seeded by the
  chord's ratios. Breaks perfect ``anisotropy_sectors``-fold symmetry,
  giving each chord a distinct fingerprint.
- ``noise_temperature`` — per-step humidity jitter standard deviation.
  Chord-driven from
  :func:`~biotuner.harmonic_geometry.media.coupling.amplitude_entropy`;
  introduces stochastic dendrite-tip splitting.

Chord coupling defaults
-----------------------
With the explicit knob set to ``None``, each parameter is derived from
the chord:

- ``humidity``           ← ``spectral_spread``    in ``[1.0e-3, 8.0e-3]``
- ``diffusion``          ← ``consonance``         in ``[0.15, 0.85]`` (consonant → low)
- ``anisotropy_sectors`` ← ``ratio_complexity``   in ``{3, 4, 5, 6, 8, 12}``
- ``asymmetry``          ← chord-deterministic    in ``[0, 0.45]``
- ``noise_temperature``  ← ``amplitude_entropy``  in ``[0, 3.0e-4]``

The chord-deterministic asymmetry is seeded by the chord's ratios so two
chords with the same ``sectors`` value still produce distinguishable
patterns: each chord prints a different phase-wobble fingerprint.

References
----------
.. [1] Reiter, C. A. (2005). A local cellular model for snow crystal growth.
       Chaos, Solitons & Fractals, 23, 1111-1119.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from biotuner.harmonic_geometry.geometry_data import GeometryData
from biotuner.harmonic_geometry.inputs import HarmonicInput
from biotuner.harmonic_geometry.media import coupling as _coupling
from biotuner.harmonic_geometry.media import structure as _structure
from biotuner.harmonic_geometry.media.base import Medium

_VALID_OUTPUTS = ("water", "frozen", "boundary")
_VALID_SEED_STRATEGIES = ("single", "polygon")

# Hard cap on prime_limit-driven sectors. Above 12 the angular bumps
# overlap so densely that diversity collapses back into "isotropic".
_MAX_SECTORS = 12


# ============================================================ hex helpers


def _hex_mask(R: int) -> np.ndarray:
    """Boolean mask of axial-coordinate cells inside the hex of radius ``R``.

    Axial coordinates ``(q, r)`` are indexed into a ``(2R+1, 2R+1)`` array
    by ``(q + R, r + R)``. The hex constraint is ``max(|q|, |r|, |q+r|) ≤ R``.
    """
    qs = np.arange(-R, R + 1)
    Q, RR = np.meshgrid(qs, qs, indexing="ij")
    return np.maximum.reduce([np.abs(Q), np.abs(RR), np.abs(Q + RR)]) <= R


def _hex_neighbor_sum(arr: np.ndarray) -> np.ndarray:
    """Sum of the 6 hex neighbors at every cell (zero-padded boundaries).

    Uses np.pad + slicing rather than np.roll to avoid wraparound from the
    opposite edge of the rectangular array.
    """
    padded = np.pad(arr, 1, mode="constant", constant_values=0.0)
    n, m = arr.shape
    return (
        padded[2:n + 2, 1:m + 1]
        + padded[0:n, 1:m + 1]
        + padded[1:n + 1, 2:m + 2]
        + padded[1:n + 1, 0:m]
        + padded[2:n + 2, 0:m]
        + padded[0:n, 2:m + 2]
    )


def _hex_neighbor_or(mask: np.ndarray) -> np.ndarray:
    """Boolean OR of the 6 hex neighbors at every cell (zero-padded)."""
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    n, m = mask.shape
    return (
        padded[2:n + 2, 1:m + 1]
        | padded[0:n, 1:m + 1]
        | padded[1:n + 1, 2:m + 2]
        | padded[1:n + 1, 0:m]
        | padded[2:n + 2, 0:m]
        | padded[0:n, 2:m + 2]
    )


# =================================================== CA core


def _crystallize_reiter(
    *,
    grid_radius: int,
    n_steps: int,
    humidity: float,
    diffusion: float,
    initial_water: float,
    anisotropy_bias: Optional[np.ndarray] = None,
    target_fill: float = 1.0,
    noise_temperature: float = 0.0,
    rng_seed: Optional[int] = None,
    seed_cells: Optional[Sequence[tuple[int, int]]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the Reiter snowflake CA on a hex grid; return ``(water, frozen)``.

    Parameters
    ----------
    target_fill : float
        Stop early when ``frozen.sum() / hex_mask.sum() >= target_fill``.
        Defaults to ``1.0`` (no early stop on fill). Used to keep crystals
        sub-saturated so dendritic morphology is visible rather than being
        masked by hex-boundary saturation.
    noise_temperature : float
        Standard deviation of additive humidity noise applied to receptive
        cells each step.
    rng_seed : int, optional
        Seed for the noise RNG (None → fresh entropy).
    seed_cells : sequence of (q, r), optional
        Additional axial-coordinate cells to freeze at initialization
        (in addition to the center cell). Used by ``seed_strategy=
        "polygon"`` to plant the chord's Tonnetz polygon as the
        starting frozen pattern.
    """
    R = int(grid_radius)
    mask = _hex_mask(R)
    n_cells = int(mask.sum())
    water = np.where(mask, float(initial_water), 0.0)
    water[R, R] = 1.0  # center seed (origin / unison)
    if seed_cells:
        for q, r in seed_cells:
            qi, ri = int(q), int(r)
            # Reject cells outside the hex.
            if (abs(qi) <= R and abs(ri) <= R and abs(qi + ri) <= R):
                water[qi + R, ri + R] = 1.0
    frozen = water >= 1.0
    base_humidity = float(humidity)
    rng = np.random.default_rng(rng_seed)

    for _ in range(int(n_steps)):
        recept = (frozen | _hex_neighbor_or(frozen)) & mask
        v = np.where(recept, water, 0.0)
        w = np.where(recept, 0.0, water)

        # Receptive: gain humidity (optionally per-direction-modulated
        # and optionally with per-step noise).
        gamma_field = base_humidity
        if anisotropy_bias is not None:
            gamma_field = base_humidity * anisotropy_bias
        if noise_temperature > 0:
            gamma_field = gamma_field + rng.normal(
                0.0, noise_temperature, size=v.shape
            )
        v_new = v + gamma_field * recept

        # Non-receptive: diffusion via (α/2) · (mean_neighbors − self).
        neighbor_sum = _hex_neighbor_sum(w)
        w_new = w + (float(diffusion) / 2.0) * (neighbor_sum / 6.0 - w)
        w_new = np.where(mask, w_new, 0.0)

        water = v_new + w_new
        frozen = water >= 1.0

        # Stop when target fill is reached OR boundary is touched.
        if target_fill < 1.0 and frozen.sum() >= target_fill * n_cells:
            break
        if (frozen[0, :].any() or frozen[-1, :].any()
                or frozen[:, 0].any() or frozen[:, -1].any()):
            break

    return water, frozen


# ============================================================ rasterization


def _hex_to_cartesian(
    values: np.ndarray,
    hex_mask: np.ndarray,
    grid_radius: int,
    resolution: int,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Project a hex-grid array to a cartesian ``(resolution, resolution)`` field.

    Uses nearest-neighbor axial sampling. The cartesian domain is
    ``[-extent, extent]²`` with ``extent = R + 0.5``. Cells outside the
    hex are returned as ``NaN`` so renderers can drop or mask them.
    """
    R = int(grid_radius)
    extent = R + 0.5
    x = np.linspace(-extent, extent, resolution)
    y = np.linspace(-extent, extent, resolution)
    X, Y = np.meshgrid(x, y, indexing="xy")

    # Inverse cartesian → axial: x = q + r/2, y = r · √3/2.
    inv_r = Y * 2.0 / np.sqrt(3.0)
    inv_q = X - inv_r / 2.0
    q_round = np.clip(np.round(inv_q).astype(int), -R, R)
    r_round = np.clip(np.round(inv_r).astype(int), -R, R)

    sampled = values[q_round + R, r_round + R]
    valid = hex_mask[q_round + R, r_round + R]
    out = np.where(valid, sampled.astype(np.float64), np.nan)
    return out, (X, Y)


def _frozen_boundary_contour(
    frozen: np.ndarray,
    hex_mask: np.ndarray,
    grid_radius: int,
    resolution: int,
) -> list[np.ndarray]:
    """Extract the frozen-region boundary as a list of (N_i, 2) polylines."""
    try:
        from skimage.measure import find_contours
    except ImportError as exc:
        raise ImportError(
            "Boundary extraction requires `scikit-image`. Install with "
            "`pip install scikit-image` and retry."
        ) from exc

    field, (X, Y) = _hex_to_cartesian(
        frozen.astype(np.float64), hex_mask, grid_radius, resolution
    )
    safe = np.where(np.isfinite(field), field, 0.0)
    contours_idx = find_contours(safe, level=0.5)
    x_axis = X[0, :]
    y_axis = Y[:, 0]
    curves: list[np.ndarray] = []
    for c in contours_idx:
        rows = c[:, 0]
        cols = c[:, 1]
        ys = np.interp(rows, np.arange(len(y_axis)), y_axis)
        xs = np.interp(cols, np.arange(len(x_axis)), x_axis)
        curves.append(np.stack([xs, ys], axis=1))
    return curves


# ============================================================== anisotropy


def _chord_seed(chord: HarmonicInput) -> int:
    """Deterministic 32-bit seed from the chord's ratios.

    Used so ``asymmetry`` wobbles and other random elements are
    reproducible per chord while still differing between chords.
    """
    ratios = [float(r) for r in chord.to_ratios()]
    accum = 0
    for r in ratios:
        # Pull bits from the float's mantissa.
        accum = (accum * 1_000_003 + int(r * 1e6)) & 0xFFFFFFFF
    return int(accum) or 1


def _select_sectors(chord: HarmonicInput) -> int:
    """Pick the anisotropy-sectors integer from the chord's structure.

    Uses the chord's *prime limit* — the highest prime in any
    numerator or denominator — as the base symmetry. Falls back to
    ``6`` for the degenerate 2-limit (octave-only) case so the
    rotational symmetry of the angular bias remains meaningful.

    Primes incompatible with the hex grid's 6-fold neighborhood
    (5, 7, 11) produce *geometric frustration* — the snowflake
    cannot be cleanly 5-fold on a hex grid, so the resulting
    morphology is visibly distinct from clean 6-fold growth.
    """
    p = _structure.prime_limit(chord)
    if p < 3:
        return 6
    return int(min(p, _MAX_SECTORS))


def _polygon_seed_cells(
    chord: HarmonicInput,
    grid_radius: int,
    seed_scale: float = 0.22,
    branch_length: int = 0,
) -> list[tuple[int, int]]:
    """Convert the chord's Tonnetz polygon to a list of axial seed cells.

    The polygon is centered, normalized to a unit max radius, then
    scaled to ``seed_scale * grid_radius`` cells from the center. Each
    polygon vertex becomes a frozen seed cell (in axial ``(q, r)``
    coordinates). With ``branch_length > 0``, an additional radial
    line of ``branch_length`` cells is planted between the center and
    each polygon vertex, giving the snowflake an initial dendritic
    skeleton.
    """
    polygon = _structure.tonnetz_polygon(chord)
    if not polygon:
        return []
    # Center polygon at its centroid.
    cx = sum(x for x, _ in polygon) / len(polygon)
    cy = sum(y for _, y in polygon) / len(polygon)
    centered = [(x - cx, y - cy) for x, y in polygon]
    max_r = max(np.hypot(x, y) for x, y in centered)
    if max_r < 1e-6:
        return []
    scale = seed_scale * grid_radius / max_r

    cells: list[tuple[int, int]] = []
    sqrt3 = np.sqrt(3.0)
    for x, y in centered:
        gx, gy = x * scale, y * scale
        # Inverse of axial → cartesian (X = q + r/2, Y = r·√3/2).
        ri = int(round(gy * 2.0 / sqrt3))
        qi = int(round(gx - ri / 2.0))
        cells.append((qi, ri))
        # Radial branch of length `branch_length` from origin to vertex.
        if branch_length > 0:
            steps = max(branch_length, 1)
            for k in range(1, steps + 1):
                f = k / float(steps + 1)
                bx, by = gx * f, gy * f
                br = int(round(by * 2.0 / sqrt3))
                bq = int(round(bx - br / 2.0))
                cells.append((bq, br))
    return cells


def _angular_bias(
    chord: HarmonicInput,
    hex_mask: np.ndarray,
    grid_radius: int,
    strength: float,
    sectors: int,
    kernel_width: float,
    asymmetry: float,
) -> np.ndarray:
    """Per-cell humidity multiplier from the chord's ratios.

    Each chord component is mapped to a pitch-class angle on the unit
    circle via ``2π · (log2(r_i) mod 1)``. Each angle is then replicated
    ``sectors`` times (giving the snowflake its base ``sectors``-fold
    symmetry) and weighted by the chord's normalized amplitudes. A
    Gaussian of width ``kernel_width`` is placed at each replicated
    angle; the resulting per-direction field is multiplied into the
    humidity term during the CA update.

    ``asymmetry`` adds a deterministic per-replica phase wobble
    (seeded by the chord) so different chords with the same sector
    count still print distinct fingerprints.
    """
    R = int(grid_radius)
    qs = np.arange(-R, R + 1)
    Q, RR = np.meshgrid(qs, qs, indexing="ij")
    X = Q + RR / 2.0
    Y = RR * np.sqrt(3.0) / 2.0
    theta = np.arctan2(Y, X)

    ratios = np.asarray(
        [float(r) for r in chord.to_ratios()], dtype=np.float64
    )
    amps = chord.normalized_amplitudes()
    if amps.size != ratios.size:
        amps = np.ones_like(ratios) / max(ratios.size, 1)
    # Per-ratio consonance weights — arms toward consonant chord
    # components grow more strongly. Combined with amplitudes so loud
    # consonant components dominate.
    cons_w = _structure.per_ratio_consonance_weights(chord)
    if cons_w.size != ratios.size:
        cons_w = np.ones_like(ratios)
    per_arm = amps * cons_w
    # Pitch-class angles in [0, 2π).
    base_angles = 2.0 * np.pi * np.mod(np.log2(np.maximum(ratios, 1e-12)), 1.0)

    rng = np.random.default_rng(_chord_seed(chord)) if asymmetry > 0 else None

    bias = np.ones_like(theta)
    for ang, w in zip(base_angles, per_arm):
        for k in range(int(sectors)):
            ang_k = ang + k * 2.0 * np.pi / float(sectors)
            if rng is not None:
                ang_k = ang_k + float(rng.uniform(-asymmetry, asymmetry))
            ang_k = np.mod(ang_k + np.pi, 2.0 * np.pi) - np.pi
            d = np.mod(theta - ang_k + np.pi, 2.0 * np.pi) - np.pi
            bias += strength * float(w) * np.exp(
                -(d ** 2) / (2.0 * kernel_width ** 2)
            )
    return np.where(hex_mask, bias, 1.0)


# ============================================================ Crystallization


class Crystallization(Medium):
    """Morphogenetic-family medium: chord-shaped Reiter CA snowflake.

    Parameters
    ----------
    humidity : float, optional
        Background humidity gain γ per step. Higher gives plumper
        shapes; lower gives sharper dendrites. If ``None`` (default),
        derived from the chord via
        ``coupling.spectral_spread(chord)`` mapped into ``[1.0e-3, 8.0e-3]``.
    diffusion : float, optional
        Water-diffusion coefficient α (in ``[0, 1]``). Higher smooths
        the crystal; lower gives crisp fingers. If ``None`` (default),
        derived from the chord via
        ``coupling.consonance(chord)`` mapped into ``[0.15, 0.85]``
        (consonant → lower diffusion → sharper).
    initial_water : float, default 0.4
        Uniform initial water content β in the unfrozen field.
    n_steps : int, default 2000
        Maximum CA steps. Simulation stops early when ``target_fill`` is
        reached or growth touches the hex boundary.
    grid_radius : int, default 110
        Half-extent of the hex grid (cells from center to edge along
        each axial direction). Total cells ≈ ``3·R²``.
    output_resolution : int, default 256
        Resolution of the cartesian rasterization.
    output_mode : {"water", "frozen", "boundary"}, default "water"
        ``"water"`` returns the continuous water-content field;
        ``"frozen"`` returns a binary {0, 1} mask;
        ``"boundary"`` returns the frozen-region outline as
        ``curve_set_2d``.
    target_fill : float, default 0.20
        Stop simulation when frozen fraction reaches this. ``1.0`` keeps
        only the boundary-touch stop. Lower values keep crystals
        sub-saturated so morphology stays visible.
    anisotropy_strength : float, default 0.8
        Strength of the per-direction humidity boost derived from the
        chord's ratios. ``0`` disables anisotropy (pure isotropic
        Reiter CA); larger values bias growth toward chord-specific
        angles.
    anisotropy_sectors : int, optional
        Number of equally-spaced replicas per chord-derived angle. Sets
        the base symmetry of the angular bias. If ``None`` (default),
        set to the chord's prime limit (highest prime in any p/q),
        clamped into ``[3, 12]``. Primes 5, 7, 11 are incompatible with
        the hex grid's 6-fold neighborhood, producing geometric
        frustration patterns that visibly differ from clean hex growth.
    anisotropy_kernel_width : float, default π/16
        Angular standard deviation (radians) of each Gaussian bump in
        the bias. Smaller → sharper dendritic rays.
    asymmetry : float, optional
        Per-replica phase wobble (radians), deterministically seeded by
        the chord. Breaks perfect ``sectors``-fold symmetry — two chords
        with the same ``sectors`` value still print distinct fingerprints.
        If ``None`` (default), derived from
        :func:`structure.max_common_int` mapped into ``[0, 0.6]`` — chords
        with larger common-denominator integers get more wobble.
    seed_strategy : {"single", "polygon"}, default "polygon"
        Initial frozen-cell pattern.

        - ``"single"`` plants only the center cell (classical Reiter).
        - ``"polygon"`` additionally plants the chord's Tonnetz polygon
          (each ratio projected to the just-intonation lattice) as
          frozen seed cells. This is the strongest source of chord-
          dependent diversity: major and minor triads, for instance,
          produce mirror-flipped polygons and therefore visibly
          mirror-flipped snowflakes.

    seed_scale : float, default 0.22
        Fraction of ``grid_radius`` used as the polygon seed's outer
        radius (only when ``seed_strategy="polygon"``). Larger values
        plant seeds farther from center; smaller keeps the polygon
        compact near the origin.
    seed_branch_length : int, default 0
        When ``> 0`` and ``seed_strategy="polygon"``, plant ``N`` extra
        frozen cells along the radial line from the center to each
        polygon vertex. Gives the snowflake an explicit initial
        dendritic skeleton.
    noise_temperature : float, optional
        Standard deviation of per-step Gaussian noise added to the
        humidity gain at receptive cells. If ``None`` (default), derived
        from ``coupling.amplitude_entropy(chord)`` in ``[0, 3.0e-4]``.
    rng_seed : int, optional
        Seed for the noise RNG. ``None`` (default) uses a deterministic
        per-chord seed so the same chord reproduces.
    """

    family = "morphogenetic"

    def __init__(
        self,
        *,
        humidity: Optional[float] = None,
        diffusion: Optional[float] = None,
        initial_water: float = 0.4,
        n_steps: int = 2000,
        grid_radius: int = 110,
        output_resolution: int = 256,
        output_mode: str = "water",
        target_fill: float = 0.20,
        anisotropy_strength: float = 0.8,
        anisotropy_sectors: Optional[int] = None,
        anisotropy_kernel_width: float = np.pi / 16.0,
        asymmetry: Optional[float] = None,
        noise_temperature: Optional[float] = None,
        rng_seed: Optional[int] = None,
        seed_strategy: str = "polygon",
        seed_scale: float = 0.22,
        seed_branch_length: int = 0,
    ) -> None:
        if output_mode not in _VALID_OUTPUTS:
            raise ValueError(
                f"output_mode must be one of {_VALID_OUTPUTS}; got "
                f"{output_mode!r}."
            )
        if humidity is not None and humidity <= 0:
            raise ValueError("humidity must be > 0 when given.")
        if diffusion is not None and not (0.0 <= diffusion <= 1.0):
            raise ValueError("diffusion must be in [0, 1] when given.")
        if not (0.0 < initial_water < 1.0):
            raise ValueError("initial_water must be in (0, 1).")
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1.")
        if grid_radius < 4:
            raise ValueError("grid_radius must be >= 4.")
        if output_resolution < 16:
            raise ValueError("output_resolution must be >= 16.")
        if anisotropy_strength < 0:
            raise ValueError("anisotropy_strength must be >= 0.")
        if not (0.0 < target_fill <= 1.0):
            raise ValueError("target_fill must be in (0, 1].")
        if anisotropy_sectors is not None and anisotropy_sectors < 1:
            raise ValueError("anisotropy_sectors must be >= 1 when given.")
        if anisotropy_kernel_width <= 0:
            raise ValueError("anisotropy_kernel_width must be > 0.")
        if asymmetry is not None and asymmetry < 0:
            raise ValueError("asymmetry must be >= 0 when given.")
        if noise_temperature is not None and noise_temperature < 0:
            raise ValueError("noise_temperature must be >= 0 when given.")
        if seed_strategy not in _VALID_SEED_STRATEGIES:
            raise ValueError(
                f"seed_strategy must be one of {_VALID_SEED_STRATEGIES}; "
                f"got {seed_strategy!r}."
            )
        if not (0.0 < seed_scale < 1.0):
            raise ValueError("seed_scale must be in (0, 1).")
        if seed_branch_length < 0:
            raise ValueError("seed_branch_length must be >= 0.")

        self.humidity = humidity
        self.diffusion = diffusion
        self.initial_water = float(initial_water)
        self.n_steps = int(n_steps)
        self.grid_radius = int(grid_radius)
        self.output_resolution = int(output_resolution)
        self.output_mode = output_mode
        self.target_fill = float(target_fill)
        self.anisotropy_strength = float(anisotropy_strength)
        self.anisotropy_sectors = anisotropy_sectors
        self.anisotropy_kernel_width = float(anisotropy_kernel_width)
        self.asymmetry = asymmetry
        self.noise_temperature = noise_temperature
        self.rng_seed = rng_seed
        self.seed_strategy = seed_strategy
        self.seed_scale = float(seed_scale)
        self.seed_branch_length = int(seed_branch_length)

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
                "Crystallization.respond requires a HarmonicInput; got "
                f"{type(forcing).__name__}."
            )

        humidity_arg = overrides.pop("humidity", self.humidity)
        diffusion_arg = overrides.pop("diffusion", self.diffusion)
        initial_water = float(overrides.pop("initial_water", self.initial_water))
        n_steps = int(overrides.pop("n_steps", self.n_steps))
        grid_radius = int(overrides.pop("grid_radius", self.grid_radius))
        output_resolution = int(overrides.pop(
            "output_resolution", self.output_resolution))
        output_mode = overrides.pop("output_mode", self.output_mode)
        target_fill = float(overrides.pop("target_fill", self.target_fill))
        anisotropy_strength = float(overrides.pop(
            "anisotropy_strength", self.anisotropy_strength))
        anisotropy_sectors = overrides.pop(
            "anisotropy_sectors", self.anisotropy_sectors)
        anisotropy_kernel_width = float(overrides.pop(
            "anisotropy_kernel_width", self.anisotropy_kernel_width))
        asymmetry_arg = overrides.pop("asymmetry", self.asymmetry)
        noise_temperature_arg = overrides.pop(
            "noise_temperature", self.noise_temperature)
        rng_seed = overrides.pop("rng_seed", self.rng_seed)
        seed_strategy = overrides.pop("seed_strategy", self.seed_strategy)
        seed_scale = float(overrides.pop("seed_scale", self.seed_scale))
        seed_branch_length = int(overrides.pop(
            "seed_branch_length", self.seed_branch_length))

        if seed_strategy not in _VALID_SEED_STRATEGIES:
            raise ValueError(
                f"seed_strategy must be one of {_VALID_SEED_STRATEGIES}."
            )

        if overrides:
            raise TypeError(
                f"Unexpected override keys: {sorted(overrides)}."
            )
        if output_mode not in _VALID_OUTPUTS:
            raise ValueError(f"output_mode must be one of {_VALID_OUTPUTS}.")

        # ----- chord-driven defaults
        if humidity_arg is None:
            spread = _coupling.spectral_spread(forcing)
            # spectral_spread typically in [0, ~2]; map to [1e-3, 8e-3].
            humidity_arg = 1.0e-3 + min(spread, 2.0) / 2.0 * (8.0e-3 - 1.0e-3)
        humidity = float(humidity_arg)

        if diffusion_arg is None:
            cons = _coupling.consonance(forcing)
            # consonance ∈ [0, 1]; consonant → low diffusion (sharper).
            diffusion_arg = 0.15 + (1.0 - cons) * (0.85 - 0.15)
        diffusion = float(diffusion_arg)

        if anisotropy_sectors is None:
            sectors = _select_sectors(forcing)
        else:
            sectors = int(anisotropy_sectors)

        if asymmetry_arg is None:
            # Chord-driven from max_common_int: chords with a larger
            # common-denominator integer form (e.g. minor 10:12:15 vs
            # major 4:5:6) get more wobble. Saturates at max_int=20
            # to keep the range bounded.
            max_int = _structure.max_common_int(forcing)
            asymmetry_arg = 0.6 * min(max(max_int - 3, 0) / 17.0, 1.0)
        asymmetry = float(asymmetry_arg)

        if noise_temperature_arg is None:
            ent = _coupling.amplitude_entropy(forcing)
            # entropy ∈ [0, 1]; flat amplitudes → more noise.
            noise_temperature_arg = ent * 3.0e-4
        noise_temperature = float(noise_temperature_arg)

        # Deterministic per-chord seed when rng_seed is None.
        seed = (int(rng_seed) if rng_seed is not None
                else _chord_seed(forcing))

        # ----- angular bias
        anisotropy = None
        if anisotropy_strength > 0:
            mask = _hex_mask(grid_radius)
            anisotropy = _angular_bias(
                forcing, mask, grid_radius,
                anisotropy_strength, sectors,
                anisotropy_kernel_width, asymmetry,
            )

        # ----- chord-polygon seed cells (Tonnetz projection)
        seed_cells: Optional[list[tuple[int, int]]] = None
        if seed_strategy == "polygon":
            seed_cells = _polygon_seed_cells(
                forcing, grid_radius, seed_scale, seed_branch_length
            )

        water, frozen = _crystallize_reiter(
            grid_radius=grid_radius,
            n_steps=n_steps,
            humidity=humidity,
            diffusion=diffusion,
            initial_water=initial_water,
            anisotropy_bias=anisotropy,
            target_fill=target_fill,
            noise_temperature=noise_temperature,
            rng_seed=seed,
            seed_cells=seed_cells,
        )
        mask = _hex_mask(grid_radius)

        parameters = {
            "humidity": humidity,
            "diffusion": diffusion,
            "initial_water": initial_water,
            "n_steps": n_steps,
            "grid_radius": grid_radius,
            "output_resolution": output_resolution,
            "output_mode": output_mode,
            "target_fill": target_fill,
            "anisotropy_strength": anisotropy_strength,
            "anisotropy_sectors": sectors,
            "anisotropy_kernel_width": anisotropy_kernel_width,
            "asymmetry": asymmetry,
            "noise_temperature": noise_temperature,
            "rng_seed": seed,
            "seed_strategy": seed_strategy,
            "seed_scale": seed_scale,
            "seed_branch_length": seed_branch_length,
        }
        metadata = {
            "kind": f"crystallization_{output_mode}",
            "family": "morphogenetic",
            "domain": "hexagonal_grid",
            "frozen_cells": int(frozen.sum()),
            "total_cells": int(mask.sum()),
            "frozen_fraction": float(frozen.sum()) / float(max(mask.sum(), 1)),
        }

        if output_mode == "boundary":
            curves = _frozen_boundary_contour(
                frozen, mask, grid_radius, output_resolution
            )
            return GeometryData(
                geom_type="curve_set_2d",
                coordinates=curves,
                parameters=parameters,
                metadata=metadata,
            )

        source = water if output_mode == "water" else frozen.astype(np.float64)
        field, grid = _hex_to_cartesian(
            source, mask, grid_radius, output_resolution
        )

        return GeometryData(
            geom_type="field_2d",
            coordinates=field,
            field_grid=grid,
            parameters=parameters,
            metadata=metadata,
        )

    def __repr__(self) -> str:
        return (
            f"Crystallization(humidity={self.humidity}, "
            f"diffusion={self.diffusion}, "
            f"anisotropy_strength={self.anisotropy_strength}, "
            f"anisotropy_sectors={self.anisotropy_sectors}, "
            f"target_fill={self.target_fill}, "
            f"seed_strategy={self.seed_strategy!r})"
        )
