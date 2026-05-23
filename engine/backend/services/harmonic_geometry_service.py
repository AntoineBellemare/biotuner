"""
Thin wrapper around `biotuner.harmonic_geometry` that exposes a uniform
JSON-shaped output for the engine API.

For each supported style we:
  1. Build a HarmonicInput from the caller's `tuning` and/or `peaks`.
  2. Dispatch to the corresponding biotuner function with the caller's
     style-specific parameters.
  3. Convert the returned GeometryData to a JSON-serialisable dict.

This keeps the frontend free to render any biotuner geometry without
duplicating the math — fast for compute-once patterns, with the engine
handling the heavy lifting.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from biotuner.harmonic_geometry.inputs import HarmonicInput
from biotuner.harmonic_geometry import (
    harmonic_knot,
    lsystem_3d,
    harmonic_point_cloud,
    recursive_polyhedron,
)
from biotuner.harmonic_geometry.fractal import subharmonic_tree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_input(
    tuning: Optional[List[float]],
    peaks: Optional[List[float]],
    base_freq: float = 1.0,
) -> HarmonicInput:
    """Build a HarmonicInput preferring peaks (absolute Hz) when provided."""
    if peaks:
        # Some peaks may be 0 / NaN — filter to positives.
        clean = [float(p) for p in peaks if p and p > 0]
        if clean:
            return HarmonicInput.from_peaks(peaks=clean)
    if tuning:
        clean = [float(r) for r in tuning if r and r > 0]
        if clean:
            return HarmonicInput.from_ratios(ratios=clean, base_freq=base_freq)
    # Fall-back: a simple harmonic series so the geometry still produces output
    return HarmonicInput.from_ratios(ratios=[1.0, 3 / 2, 5 / 4, 4 / 3])


def _to_list(arr) -> Any:
    """Recursively convert numpy values into JSON-serialisable equivalents.

    Handles: ndarray, lists/tuples, dicts, numpy scalars, Fractions, sets.
    Anything else (str, int, float, bool, None) passes through unchanged.
    Importantly we walk *inside* dicts so the parameters/metadata fields
    of GeometryData don't smuggle ndarrays through.
    """
    if arr is None:
        return None
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    if isinstance(arr, (np.floating, np.integer, np.bool_)):
        return arr.item()
    if isinstance(arr, dict):
        return {str(k): _to_list(v) for k, v in arr.items()}
    if isinstance(arr, (list, tuple)):
        return [_to_list(a) for a in arr]
    if isinstance(arr, set):
        return [_to_list(a) for a in sorted(arr, key=str)]
    # Fraction / sympy.Rational / anything with float() — coerce safely.
    try:
        import fractions
        if isinstance(arr, fractions.Fraction):
            return float(arr)
    except Exception:
        pass
    return arr


def _serialize_geometry(geom) -> Dict[str, Any]:
    """GeometryData → JSON-safe dict."""
    return {
        "geom_type": str(geom.geom_type),
        "coordinates": _to_list(geom.coordinates),
        "edges": _to_list(geom.edges),
        "faces": _to_list(geom.faces),
        "weights": _to_list(geom.weights),
        "field_grid": _to_list(geom.field_grid),
        "parameters": _to_list(geom.parameters),
        "metadata": _to_list(geom.metadata),
    }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

# Each entry: (callable, allowed-param keys, defaults). We intentionally
# whitelist params to avoid passing arbitrary kwargs from the network.
_STYLES = {
    "harmonic_knot": {
        "fn": harmonic_knot,
        "allowed": {"n_points", "tube_radius", "n_sides", "major_radius", "minor_radius"},
        "coerce": {
            "n_points":     int,
            "tube_radius":  float,
            "n_sides":      int,
            "major_radius": float,
            "minor_radius": float,
        },
        "clamp": {
            "n_points":     (50, 4000),
            "tube_radius":  (0.005, 0.5),
            "n_sides":      (4, 64),
            "major_radius": (0.5, 5.0),
            "minor_radius": (0.05, 3.0),
        },
    },
    "lsystem_3d": {
        "fn": lsystem_3d,
        "allowed": {"depth", "step_length", "axiom"},
        "coerce": {"depth": int, "step_length": float, "axiom": str},
        "clamp": {"depth": (1, 5), "step_length": (0.05, 5.0)},
    },
    "harmonic_point_cloud": {
        "fn": harmonic_point_cloud,
        "allowed": {"n_points", "surface"},
        "coerce": {"n_points": int, "surface": str},
        "clamp": {"n_points": (200, 10000)},
    },
    "recursive_polyhedron": {
        "fn": recursive_polyhedron,
        "allowed": {"depth", "solid", "per_face_bump", "apex_twist"},
        "coerce": {
            "depth": int,
            "solid": str,
            "per_face_bump": bool,
            "apex_twist": bool,
        },
        "clamp": {"depth": (0, 4)},
    },
    "subharmonic_tree": {
        "fn": subharmonic_tree,
        "allowed": {"depth", "n_harmonics", "min_freq", "layout"},
        "coerce": {
            "depth": int,
            "n_harmonics": int,
            "min_freq": float,
            "layout": str,
        },
        "clamp": {
            "depth": (1, 6),
            "n_harmonics": (2, 9),
            "min_freq": (0.001, 1000.0),
        },
    },
}


def _sanitize(params: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    """Apply whitelist + type coercion + range clamps."""
    out: Dict[str, Any] = {}
    for k, v in (params or {}).items():
        if k not in spec["allowed"] or v is None:
            continue
        coerce = spec.get("coerce", {}).get(k)
        if coerce is not None:
            try:
                v = coerce(v)
            except (TypeError, ValueError):
                continue
        clamp = spec.get("clamp", {}).get(k)
        if clamp is not None and isinstance(v, (int, float)):
            lo, hi = clamp
            if v < lo: v = lo
            if v > hi: v = hi
        out[k] = v
    return out


def compute(
    style: str,
    params: Optional[Dict[str, Any]] = None,
    tuning: Optional[List[float]] = None,
    peaks: Optional[List[float]] = None,
    base_freq: float = 1.0,
) -> Dict[str, Any]:
    if style not in _STYLES:
        raise ValueError(
            f"Unknown geometry style {style!r}. "
            f"Available: {sorted(_STYLES)}"
        )
    spec = _STYLES[style]
    kwargs = _sanitize(params, spec)
    h_input = _build_input(tuning, peaks, base_freq=base_freq)
    geom = spec["fn"](h_input, **kwargs)
    return _serialize_geometry(geom)


def list_styles() -> List[str]:
    return sorted(_STYLES.keys())
