"""
biotuner.harmonic_geometry.metrics
===================================
Quantitative measurements over :class:`GeometryData` (and time-resolved
:class:`HarmonicSequence` after a generator is applied) — the entry point
for running science on top of the harmonic-geometry module.

This module is **strictly geometry-side**. For harmonic-content metrics on
the underlying signal (consonance, dyad_similarity, euler, tenney,
subharmonic_tension), use :mod:`biotuner.metrics` and
:class:`biotuner.biotuner_group.BiotunerGroup` directly — those tools
already exist and are not duplicated here.

Public API
----------
geometry_metrics(geom)                  -- scalar dict of structural +
                                           per-method stats for a
                                           :class:`GeometryData`
list_supported_kinds()                  -- names of every recognised
                                           metadata['kind']
sequence_metrics(seq, generator, **kw)  -- apply ``generator`` to each
                                           :class:`HarmonicInput` frame and
                                           return per-frame
                                           :func:`geometry_metrics` as a
                                           dict of 1-D ndarrays
compare(geometries, labels=None)        -- N-way comparison table
                                           (column-oriented dict)
normalize_metrics(rows, ...)            -- min-max scale rows to ``[0, 1]``
                                           (data-driven; optional ``senses``
                                           argument inverts metric direction)
MetricsLog                              -- append-only log of metric rows
                                           (geometry / generator output);
                                           CSV / JSON export

All extractors fall back gracefully — missing fields produce ``nan`` rather
than raising.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from biotuner.harmonic_geometry.geometry_data import GeometryData
from biotuner.harmonic_geometry.inputs import HarmonicInput, HarmonicSequence


# ═══════════════════════════════════════════ per-kind extractors ═════════════
# Each `_extract_*` returns a dict of method-specific scalar metrics that gets
# merged onto the generic :func:`_geometry_metrics_generic` output.


def _safe_path_perimeter(coords: np.ndarray, closed: bool = False) -> float:
    if coords is None or len(coords) < 2:
        return 0.0
    diffs = np.diff(coords, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    if closed:
        seg = np.concatenate([seg, [np.linalg.norm(coords[0] - coords[-1])]])
    return float(seg.sum())


def _polygon_signed_area(coords: np.ndarray) -> float:
    """Shoelace area of a 2-D polygon (positive = CCW)."""
    if coords is None or len(coords) < 3:
        return 0.0
    x = coords[:, 0]; y = coords[:, 1]
    return 0.5 * float(np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def _box_count_dim(points: np.ndarray, n_scales: int = 8) -> float:
    """Box-counting fractal dimension of a 2-D point cloud (rough)."""
    if points is None or len(points) < 50:
        return float("nan")
    pts = np.asarray(points, dtype=np.float64)
    lo = pts.min(axis=0); hi = pts.max(axis=0)
    span = max(float((hi - lo).max()), 1e-9)
    pts01 = (pts - lo) / span
    sizes  = []
    counts = []
    for k in range(1, n_scales + 1):
        n = 2 ** k
        idx = np.clip((pts01 * n).astype(np.int64), 0, n - 1)
        keys = idx[:, 0] * n + idx[:, 1]
        n_occupied = len(np.unique(keys))
        if n_occupied <= 1:
            break
        sizes.append(1.0 / n)
        counts.append(n_occupied)
    if len(sizes) < 3:
        return float("nan")
    log_inv = np.log(1.0 / np.asarray(sizes))
    log_cnt = np.log(np.asarray(counts))
    slope, _ = np.polyfit(log_inv, log_cnt, 1)
    return float(slope)


def _extract_lissajous_2d(g: GeometryData) -> Dict[str, float]:
    md = g.metadata or {}
    coords = np.asarray(g.coordinates)
    return {
        "lobes_x":     float(md.get("lobes_x", float("nan"))),
        "lobes_y":     float(md.get("lobes_y", float("nan"))),
        "closed":      float(bool(md.get("closed", False))),
        "perimeter":   _safe_path_perimeter(coords, closed=False),
    }


def _extract_lissajous_3d(g: GeometryData) -> Dict[str, float]:
    md = g.metadata or {}
    coords = np.asarray(g.coordinates)
    return {
        "is_knot":     float(bool(md.get("knot", False))),
        "perimeter":   _safe_path_perimeter(coords),
    }


def _extract_lissajous_compound(g: GeometryData) -> Dict[str, float]:
    md = g.metadata or {}
    return {
        "n_components": float(md.get("n_components", float("nan"))),
        "perimeter":    _safe_path_perimeter(np.asarray(g.coordinates)),
    }


def _extract_lissajous_phase_drift(g: GeometryData) -> Dict[str, float]:
    md = g.metadata or {}
    return {
        "drift_active": float(bool(md.get("phase_drift", False))),
        "perimeter":    _safe_path_perimeter(np.asarray(g.coordinates)),
    }


def _extract_harmonograph(g: GeometryData) -> Dict[str, float]:
    """Energy decay: ratio of late-segment perimeter to early-segment."""
    coords = np.asarray(g.coordinates)
    n = len(coords)
    if n < 8:
        return {}
    early = coords[: n // 4]
    late  = coords[3 * n // 4 :]
    early_amp = float(np.linalg.norm(np.std(early, axis=0)))
    late_amp  = float(np.linalg.norm(np.std(late,  axis=0)))
    return {
        "early_amp":  early_amp,
        "late_amp":   late_amp,
        "decay_ratio": late_amp / early_amp if early_amp > 0 else float("nan"),
        "perimeter":  _safe_path_perimeter(coords),
    }


def _extract_chladni_field(g: GeometryData) -> Dict[str, float]:
    md = g.metadata or {}
    field = np.asarray(g.coordinates, dtype=np.float64)
    if field.ndim < 2:
        return {}
    abs_field = np.abs(field)
    energy = float(np.nansum(field ** 2))
    # Dominant amplitude location (peak abs value)
    if np.any(np.isfinite(abs_field)):
        flat_idx = int(np.nanargmax(abs_field))
        peak_val = float(np.nanmax(abs_field))
    else:
        peak_val = float("nan")
    out = {
        "n_modes":      float(md.get("n_modes", float("nan"))),
        "energy":       energy,
        "peak_abs":     peak_val,
        "active_frac":  float((abs_field > 0.1 * (peak_val or 1.0)).sum() / abs_field.size),
    }
    if md.get("eigenvalues"):
        eigs = np.asarray(md["eigenvalues"], dtype=np.float64)
        out["eig_min"]  = float(eigs.min())
        out["eig_max"]  = float(eigs.max())
        out["eig_span"] = float(eigs.max() - eigs.min())
    return out


def _extract_chladni_3d(g: GeometryData) -> Dict[str, float]:
    md = g.metadata or {}
    field = np.asarray(g.coordinates, dtype=np.float64)
    if field.ndim < 3:
        return {}
    abs_field = np.abs(field)
    energy    = float(np.nansum(field ** 2))
    # 3-D zero-crossing fraction (already approximated for 2-D in generic)
    sign = np.sign(field)
    zc = (np.abs(np.diff(sign, axis=0)).sum()
          + np.abs(np.diff(sign, axis=1)).sum()
          + np.abs(np.diff(sign, axis=2)).sum())
    return {
        "n_modes":            float(md.get("n_modes", float("nan"))),
        "energy":             energy,
        "peak_abs":           float(np.nanmax(abs_field)),
        "zero_crossing_frac_3d": float(zc / (3.0 * field.size)),
    }


def _extract_chladni_nodal_lines(g: GeometryData) -> Dict[str, float]:
    md = g.metadata or {}
    n_curves = int(md.get("n_contours", 0))
    total_len = 0.0
    for c in g.coordinates:
        arr = np.asarray(c)
        if arr.size:
            total_len += _safe_path_perimeter(arr)
    return {
        "n_contours":      float(n_curves),
        "total_arc_length": float(total_len),
    }


def _extract_chladni_nodal_surface(g: GeometryData) -> Dict[str, float]:
    md = g.metadata or {}
    return {
        "n_vertices":   float(md.get("n_vertices", len(g.coordinates))),
        "n_faces":      float(md.get("n_faces",
                                       len(g.faces) if g.faces is not None else 0)),
    }


def _extract_star_polygon(g: GeometryData) -> Dict[str, float]:
    md = g.metadata or {}
    return {
        "compound":              float(bool(md.get("compound", False))),
        "n_components":          float(md.get("n_components", 1)),
        "vertices_per_component": float(md.get("vertices_per_component",
                                                 md.get("n_vertices", float("nan")))),
    }


def _extract_times_table_circle(g: GeometryData) -> Dict[str, float]:
    md = g.metadata or {}
    params = g.parameters or {}
    edges = np.asarray(g.edges) if g.edges is not None else np.empty((0, 2), int)
    unique_targets = int(np.unique(edges[:, 1]).size) if edges.size else 0
    return {
        "multiplier":     float(params.get("multiplier", float("nan"))),
        "n_edges":        float(md.get("n_edges", len(edges))),
        "unique_targets": float(unique_targets),
    }


def _extract_times_table_from_input(g: GeometryData) -> Dict[str, float]:
    md = g.metadata or {}
    return {
        "n_ratios":       float(md.get("n_ratios", float("nan"))),
        "n_edges":        float(len(g.edges) if g.edges is not None else 0),
    }


def _extract_tuning_circle(g: GeometryData) -> Dict[str, float]:
    coords = np.asarray(g.coordinates)
    if coords.size == 0:
        return {}
    angles = np.arctan2(coords[:, 1], coords[:, 0])
    angles_unit = (angles / (2 * math.pi)) % 1.0
    return {
        "n_pitches":              float(len(coords)),
        "pitch_class_dispersion": float(np.std(angles_unit)),
    }


def _extract_rose(g: GeometryData) -> Dict[str, float]:
    md = g.metadata or {}
    return {
        "petal_count": float(md.get("petals", float("nan"))),
        "perimeter":   _safe_path_perimeter(np.asarray(g.coordinates), closed=True),
    }


def _extract_cycloid(g: GeometryData) -> Dict[str, float]:
    md = g.metadata or {}
    return {
        "cusps":     float(md.get("cusps", float("nan"))),
        "perimeter": _safe_path_perimeter(np.asarray(g.coordinates), closed=True),
    }


def _extract_polygon_chord_pattern(g: GeometryData) -> Dict[str, float]:
    return {
        "n_chord_edges": float(len(g.edges) if g.edges is not None else 0),
    }


def _extract_consonance_polygon(g: GeometryData) -> Dict[str, float]:
    coords = np.asarray(g.coordinates)
    return {
        "perimeter": _safe_path_perimeter(coords, closed=True),
        "area":      _polygon_signed_area(coords),
    }


def _extract_interval_vector_diagram(g: GeometryData) -> Dict[str, float]:
    edges = g.edges
    if edges is None or g.weights is None:
        return {}
    w = np.asarray(g.weights, dtype=np.float64)
    return {
        "n_intervals":      float(len(edges)),
        "metric_mean":      float(np.nanmean(w)),
        "metric_max":       float(np.nanmax(w)),
    }


def _extract_stern_brocot(g: GeometryData) -> Dict[str, float]:
    md = g.metadata or {}
    weights = g.weights
    out = {
        "max_depth": float(md.get("max_depth", float("nan"))),
    }
    if weights is not None and len(weights):
        h = np.asarray(weights, dtype=np.float64)
        out["harmonicity_mean"] = float(np.nanmean(h))
        out["harmonicity_std"]  = float(np.nanstd(h))
        out["harmonicity_max"]  = float(np.nanmax(h))
    if "nearest_input_dist_cents" in md:
        d = np.asarray(md["nearest_input_dist_cents"], dtype=np.float64)
        out["min_chord_dist_cents"] = float(np.nanmin(d))
        out["mean_chord_dist_cents"] = float(np.nanmean(d))
    return out


def _extract_continued_fraction(g: GeometryData) -> Dict[str, float]:
    md = g.metadata or {}
    cf = md.get("cf_expansion") or md.get("continued_fraction") or []
    return {
        "n_squares":      float(md.get("n_squares", float("nan"))),
        "cf_length":      float(len(cf)) if cf else float("nan"),
        "cf_max_coef":    float(max(cf)) if cf else float("nan"),
        "inverted":       float(bool(md.get("inverted", False))),
    }


def _extract_farey(g: GeometryData) -> Dict[str, float]:
    md = g.metadata or {}
    fracs = md.get("fractions") or []
    denoms = []
    for s in fracs:
        try:
            denoms.append(int(Fraction(s).denominator))
        except Exception:
            pass
    out = {
        "n_terms":   float(md.get("n_terms", len(fracs))),
    }
    if denoms:
        out["denom_max"]  = float(max(denoms))
        out["denom_mean"] = float(np.mean(denoms))
    return out


def _extract_subharmonic_tree(g: GeometryData) -> Dict[str, float]:
    md = g.metadata or {}
    freqs = md.get("frequencies_hz") or []
    if not freqs:
        return {"n_nodes": float(md.get("n_nodes", 0))}
    f = np.asarray(freqs, dtype=np.float64)
    f_pos = f[f > 0]
    octaves = (math.log2(float(f_pos.max()) / float(f_pos.min()))
               if len(f_pos) >= 2 else 0.0)
    depths = np.asarray(md.get("depth_per_node", []), dtype=np.int64)
    return {
        "n_nodes":           float(md.get("n_nodes", len(freqs))),
        "n_octaves_spanned": float(octaves),
        "min_freq":          float(f_pos.min()) if len(f_pos) else float("nan"),
        "max_freq":          float(f_pos.max()) if len(f_pos) else float("nan"),
        "max_depth_reached": float(depths.max()) if depths.size else float("nan"),
    }


def _extract_ifs(g: GeometryData) -> Dict[str, float]:
    md = g.metadata or {}
    coords = np.asarray(g.coordinates)
    return {
        "n_points":     float(len(coords)),
        "span":         float(md.get("span", float("nan"))),
        "fractal_dim":  _box_count_dim(coords) if coords.ndim == 2 and coords.shape[1] == 2 else float("nan"),
        "n_maps":       float(md.get("n_maps", float("nan"))),
    }


def _extract_lsystem(g: GeometryData) -> Dict[str, float]:
    md = g.metadata or {}
    coords = np.asarray(g.coordinates)
    edges  = np.asarray(g.edges) if g.edges is not None else np.empty((0, 2), int)
    total_len = 0.0
    if edges.size and coords.size:
        v0 = coords[edges[:, 0]]; v1 = coords[edges[:, 1]]
        total_len = float(np.linalg.norm(v1 - v0, axis=1).sum())
    return {
        "n_segments":     float(md.get("n_segments", len(edges))),
        "lstring_length": float(md.get("lstring_length",
                                         md.get("lstring_len", float("nan")))),
        "angle_deg":      float(md.get("angle_deg", float("nan"))),
        "total_length":   total_len,
    }


def _extract_recursive_polygon(g: GeometryData) -> Dict[str, float]:
    coords = np.asarray(g.coordinates)
    md = g.metadata or {}
    params = g.parameters or {}
    return {
        "n_vertices":   float(md.get("n_vertices", len(coords))),
        "perimeter":    _safe_path_perimeter(coords, closed=True),
        "area":         _polygon_signed_area(coords),
        "scale_factor": float(params.get("scale_factor", float("nan"))),
        "depth":        float(params.get("depth", float("nan"))),
    }


def _extract_self_similar_tuning(g: GeometryData) -> Dict[str, float]:
    md = g.metadata or {}
    n_per = md.get("n_nodes_per_level") or []
    return {
        "total_nodes":      float(md.get("total_nodes", len(g.coordinates))),
        "n_generations":    float(len(n_per)),
        "level_0_pitches":  float(len(md.get("pitches_level_0", []))),
    }


def _extract_lissajous_tube(g: GeometryData) -> Dict[str, float]:
    md = g.metadata or {}
    return {
        "rx":         float(md.get("rx", float("nan"))),
        "ry":         float(md.get("ry", float("nan"))),
        "rz":         float(md.get("rz", float("nan"))),
    }


def _extract_harmonic_knot(g: GeometryData) -> Dict[str, float]:
    md = g.metadata or {}
    return {
        "winding_p": float(md.get("winding_p", float("nan"))),
        "winding_q": float(md.get("winding_q", float("nan"))),
    }


def _extract_harmonic_surface(g: GeometryData) -> Dict[str, float]:
    md = g.metadata or {}
    params = g.parameters or {}
    coords = np.asarray(g.coordinates)
    deformation = float(np.std(np.linalg.norm(coords, axis=1))) if coords.size else 0.0
    return {
        "mode":         hash(str(md.get("mode", params.get("mode", "")))) % 100 / 100.0,  # ordinal stand-in
        "deformation":  deformation,
        "resolution":   float(params.get("resolution", float("nan"))),
    }


def _extract_lsystem_3d(g: GeometryData) -> Dict[str, float]:
    md = g.metadata or {}
    return {
        "n_segments":   float(md.get("n_segments", float("nan"))),
        "lstring_len":  float(md.get("lstring_len", float("nan"))),
    }


def _extract_recursive_polyhedron(g: GeometryData) -> Dict[str, float]:
    md = g.metadata or {}
    params = g.parameters or {}
    fri = md.get("face_ratio_index")
    out = {
        "depth":          float(params.get("depth", float("nan"))),
        "bump_scale":     float(params.get("bump_scale", float("nan"))),
        "per_face_bump":  float(bool(params.get("per_face_bump", False))),
        "apex_twist":     float(bool(params.get("apex_twist", False))),
        "n_ratios":       float(md.get("n_ratios", float("nan"))),
    }
    if fri is not None and len(fri):
        idx = np.asarray(fri, dtype=np.int64)
        unique, counts = np.unique(idx, return_counts=True)
        out["face_ratio_entropy"] = float(
            -np.sum((counts / counts.sum()) * np.log(counts / counts.sum() + 1e-12))
        )
        out["face_ratio_n_unique"] = float(len(unique))
    return out


def _extract_harmonic_point_cloud(g: GeometryData) -> Dict[str, float]:
    md = g.metadata or {}
    fr = md.get("field_range") or [float("nan"), float("nan")]
    return {
        "actual_n_points": float(md.get("actual_n_points", len(g.coordinates))),
        "field_min":       float(fr[0]),
        "field_max":       float(fr[1]),
    }


# Dispatch: kind -> extractor.
_GEOM_EXTRACTORS: Dict[str, Callable[[GeometryData], Dict[str, float]]] = {
    "lissajous_2d":              _extract_lissajous_2d,
    "lissajous_3d":              _extract_lissajous_3d,
    "lissajous_compound":        _extract_lissajous_compound,
    "lissajous_phase_drift":     _extract_lissajous_phase_drift,
    "harmonograph_lateral":      _extract_harmonograph,
    "harmonograph_rotary":       _extract_harmonograph,
    "harmonograph_3d":           _extract_harmonograph,
    "chladni_field_rectangular": _extract_chladni_field,
    "chladni_field_circular":    _extract_chladni_field,
    "chladni_field_polygon":     _extract_chladni_field,
    "chladni_field_3d_box":      _extract_chladni_3d,
    "nodal_lines":               _extract_chladni_nodal_lines,
    "nodal_surface":             _extract_chladni_nodal_surface,
    "star_polygon":              _extract_star_polygon,
    "times_table_circle":        _extract_times_table_circle,
    "times_table_from_input":    _extract_times_table_from_input,
    "tuning_circle":             _extract_tuning_circle,
    "rose":                      _extract_rose,
    "epicycloid":                _extract_cycloid,
    "hypocycloid":               _extract_cycloid,
    "polygon_chord_pattern":     _extract_polygon_chord_pattern,
    "consonance_polygon":        _extract_consonance_polygon,
    "interval_vector_diagram":   _extract_interval_vector_diagram,
    "stern_brocot_tree":         _extract_stern_brocot,
    "continued_fraction_rectangles": _extract_continued_fraction,
    "farey_sequence_layout":     _extract_farey,
    "subharmonic_tree":          _extract_subharmonic_tree,
    "ifs_harmonic":              _extract_ifs,
    "lsystem_from_ratios":       _extract_lsystem,
    "recursive_polygon":         _extract_recursive_polygon,
    "self_similar_tuning":       _extract_self_similar_tuning,
    "lissajous_tube":            _extract_lissajous_tube,
    "harmonic_knot":             _extract_harmonic_knot,
    "harmonic_surface":          _extract_harmonic_surface,
    "lsystem_3d":                _extract_lsystem_3d,
    "recursive_polyhedron":      _extract_recursive_polyhedron,
    "harmonic_point_cloud":      _extract_harmonic_point_cloud,
}


# ═══════════════════════════════════════════ geometry metrics ════════════════


def _geometry_metrics_generic(geom: GeometryData) -> Dict[str, float]:
    """Structural / quantitative stats for any :class:`GeometryData` type.

    The returned dict always contains ``n_vertices`` (or 0). Mesh-typed
    geometries also receive ``n_faces`` and edge-length statistics; tree /
    graph types receive ``n_edges`` and degree stats; field types receive
    field-value stats. Coordinates that are not 1-D arrays are recursively
    flattened to compute spatial extent (``span_x / span_y / span_z``).
    """
    out: Dict[str, float] = {}
    coords = geom.coordinates
    gt = geom.geom_type

    # n_vertices: leaf size of coordinates
    try:
        if isinstance(coords, np.ndarray):
            if coords.ndim == 2 and coords.shape[1] in (2, 3):
                n_v = coords.shape[0]
            elif coords.dtype == object:
                # array of polygons / objects
                n_v = sum(int(np.asarray(p).shape[0]) for p in coords)
            else:
                n_v = int(np.prod(coords.shape))
        else:
            n_v = len(coords)
    except Exception:
        n_v = 0
    out["n_vertices"] = float(n_v)

    # Spatial extent (only for vertex-shaped coordinates)
    pts = None
    if isinstance(coords, np.ndarray) and coords.ndim == 2 and coords.shape[1] in (2, 3):
        pts = coords
    elif isinstance(coords, np.ndarray) and coords.dtype == object and len(coords):
        try:
            pts = np.vstack([np.asarray(p) for p in coords if hasattr(p, "shape")])
        except Exception:
            pts = None
    if pts is not None and pts.size:
        rng = np.ptp(pts, axis=0)
        out["span_x"] = float(rng[0])
        if rng.shape[0] >= 2:
            out["span_y"] = float(rng[1])
        if rng.shape[0] >= 3:
            out["span_z"] = float(rng[2])

    # Mesh-specific
    if geom.faces is not None:
        faces = np.asarray(geom.faces)
        out["n_faces"] = float(len(faces))
        if pts is not None and len(faces):
            # per-edge length distribution from triangle edges
            v0 = pts[faces[:, 0]]
            v1 = pts[faces[:, 1]]
            v2 = pts[faces[:, 2]]
            d01 = np.linalg.norm(v1 - v0, axis=1)
            d12 = np.linalg.norm(v2 - v1, axis=1)
            d20 = np.linalg.norm(v0 - v2, axis=1)
            edges = np.concatenate([d01, d12, d20])
            out["edge_len_mean"] = float(edges.mean())
            out["edge_len_std"]  = float(edges.std())
            # cheap surface-area estimate (sum of triangle areas)
            cross = np.cross(v1 - v0, v2 - v0)
            tri_area = 0.5 * np.linalg.norm(cross, axis=1)
            out["surface_area"] = float(tri_area.sum())

    # Edge / graph stats
    if geom.edges is not None:
        edges = np.asarray(geom.edges)
        out["n_edges"] = float(len(edges))
        if len(edges) and pts is not None:
            v0 = pts[edges[:, 0]]
            v1 = pts[edges[:, 1]]
            d = np.linalg.norm(v1 - v0, axis=1)
            out.setdefault("edge_len_mean", float(d.mean()))
            out.setdefault("edge_len_std", float(d.std()))
            # node degrees
            n_nodes = pts.shape[0]
            deg = np.bincount(edges.ravel(), minlength=n_nodes)
            out["degree_mean"] = float(deg.mean())
            out["degree_max"]  = float(deg.max())

    # Field-specific stats: field_2d / field_3d store the array in coordinates
    if gt in {"field_2d", "field_3d"} and isinstance(coords, np.ndarray):
        f = coords.astype(np.float64, copy=False)
        out["field_min"]  = float(np.nanmin(f))
        out["field_max"]  = float(np.nanmax(f))
        out["field_mean"] = float(np.nanmean(f))
        out["field_std"]  = float(np.nanstd(f))
        # Nodal-line proxy: fraction of cells that change sign vs neighbour
        if f.ndim >= 2:
            sign = np.sign(f)
            diff_x = np.abs(np.diff(sign, axis=-1))
            diff_y = np.abs(np.diff(sign, axis=-2))
            out["zero_crossing_frac"] = float(
                (diff_x.sum() + diff_y.sum()) / (2.0 * f.size)
            )

    # Weights stats (point-cloud field amplitudes etc.)
    if geom.weights is not None and len(geom.weights):
        w = np.asarray(geom.weights, dtype=np.float64)
        out["weight_mean"] = float(np.nanmean(w))
        out["weight_std"]  = float(np.nanstd(w))

    return out


def geometry_metrics(geom: GeometryData) -> Dict[str, float]:
    """Structural + per-method metrics for any :class:`GeometryData`.

    Always returns the **generic** stats (``n_vertices``, ``n_faces`` /
    ``n_edges`` when applicable, spatial spans, edge-length and field stats).
    When the geometry's ``metadata['kind']`` matches a registered generator
    (lissajous_2d, chladni_field_*, stern_brocot_tree, ifs_harmonic,
    recursive_polyhedron, harmonic_point_cloud, …), method-specific scalars
    are merged on top.

    The list of recognised kinds is :data:`_GEOM_EXTRACTORS`. Geometries
    without a recognised kind still receive the generic stats.

    Parameters
    ----------
    geom : GeometryData

    Returns
    -------
    dict[str, float]
    """
    base = _geometry_metrics_generic(geom)
    md = geom.metadata or {}
    kind = md.get("kind")
    extractor = _GEOM_EXTRACTORS.get(kind) if kind else None
    if extractor is not None:
        try:
            base.update(extractor(geom))
        except Exception:
            # Extractors are best-effort; never break the generic output.
            pass
    if kind:
        base["kind"] = kind  # informational
    return base


def list_supported_kinds() -> List[str]:
    """Return every metadata['kind'] string recognised by :func:`geometry_metrics`.

    Useful for sanity-checking instrumentation coverage across the module.
    """
    return sorted(_GEOM_EXTRACTORS.keys())


# ═══════════════════════════════════════════ sequence metrics ════════════════


def sequence_metrics(
    seq: HarmonicSequence,
    generator: Callable[..., GeometryData],
    **generator_kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Apply ``generator`` to every frame of ``seq`` and return per-frame
    geometry-metric trajectories.

    Parameters
    ----------
    seq : HarmonicSequence
    generator : callable
        Any harmonic-geometry generator that takes a
        :class:`HarmonicInput` as its first positional argument and returns a
        :class:`GeometryData` (e.g. ``harmonic_knot``, ``chladni_from_input``,
        ``recursive_polyhedron``, …).
    **generator_kwargs
        Forwarded to ``generator`` on every frame.

    Returns
    -------
    dict[str, ndarray]
        ``{metric_name: 1-D ndarray of length n_frames}``. The metric set is
        the union over all frames; missing values are ``nan``.

    Examples
    --------
    >>> traj = sequence_metrics(seq, harmonic_knot, n_points=400)
    >>> traj["winding_p"].shape
    (T,)
    """
    if not callable(generator):
        raise TypeError(
            "sequence_metrics requires a callable generator (e.g. "
            "harmonic_knot, chladni_from_input, recursive_polyhedron). "
            "For raw harmonic-content stats over a windowed signal, see "
            "biotuner.metrics and biotuner.biotuner_group.BiotunerGroup."
        )
    rows: List[Dict[str, float]] = []
    for frame in seq.frames:
        try:
            geom = generator(frame, **generator_kwargs)
            rows.append(geometry_metrics(geom))
        except Exception:
            rows.append({})
    keys: List[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    out: Dict[str, np.ndarray] = {}
    for k in keys:
        vals = []
        for row in rows:
            v = row.get(k, float("nan"))
            try:
                vals.append(float(v) if v is not None else float("nan"))
            except (TypeError, ValueError):
                vals.append(float("nan"))
        out[k] = np.asarray(vals, dtype=np.float64)
    return out


# ═══════════════════════════════════════════ comparison ═══════════════════════


def compare(
    geometries: Sequence[GeometryData],
    labels: Optional[Sequence[str]] = None,
) -> Dict[str, List[float]]:
    """Side-by-side metric comparison across a list of :class:`GeometryData`.

    Parameters
    ----------
    geometries : sequence of GeometryData
    labels : sequence of str, optional
        One label per geometry. Defaults to ``geom_0, geom_1, …``.

    Returns
    -------
    dict
        ``{metric_name: [v_for_item_1, …]}`` plus a special ``'__labels__'``
        key carrying the user-supplied or auto labels.
    """
    if not geometries:
        raise ValueError("compare(): empty geometries list.")
    if not all(isinstance(g, GeometryData) for g in geometries):
        raise TypeError(
            "compare(): every item must be a GeometryData. For raw harmonic "
            "comparisons across HarmonicInputs, use biotuner.metrics directly."
        )
    labels = list(labels) if labels else [f"geom_{i}" for i in range(len(geometries))]
    if len(labels) != len(geometries):
        raise ValueError(
            f"len(labels)={len(labels)} does not match len(geometries)="
            f"{len(geometries)}."
        )
    rows = [geometry_metrics(g) for g in geometries]
    keys: List[str] = []
    for row in rows:
        for k in row:
            if k not in keys and k != "kind":
                keys.append(k)
    out: Dict[str, List[float]] = {"__labels__": labels}
    for k in keys:
        out[k] = [row.get(k, float("nan")) for row in rows]
    return out


# ═══════════════════════════════════════════ normalize_metrics ═══════════════


def normalize_metrics(
    rows: Sequence[Dict[str, float]],
    metrics: Optional[Sequence[str]] = None,
    bounds: Optional[Dict[str, tuple]] = None,
    senses: Optional[Dict[str, int]] = None,
) -> List[Dict[str, float]]:
    """Map metric values to ``[0, 1]`` for radar plotting.

    Bounds are data-driven by default (per-metric min/max across rows).  Pass
    ``bounds={name: (lo, hi)}`` to override.

    Parameters
    ----------
    rows : sequence of dict
    metrics : sequence of str, optional
        Subset of metric keys to normalise; defaults to ``rows[0]`` keys.
    bounds : dict, optional
        Explicit ``{metric: (lo, hi)}`` per-metric scales.
    senses : dict, optional
        ``{metric: +1 | -1}``. ``+1`` (default) keeps the data-driven mapping;
        ``-1`` inverts (i.e. higher raw value → lower normalised value).
        Useful for "lower is better" metrics like edge irregularity.

    Behaviour notes
    ---------------
    * Non-finite raw values stay ``nan`` (not rendered on a radar).
    * Zero-variance metrics (all rows agree) map to ``0.5`` — interpreted as
      "consensus / no information" rather than misleadingly collapsing to 0.
    """
    if not rows:
        return []
    keys = list(metrics) if metrics else list(rows[0].keys())
    bounds = dict(bounds or {})
    senses = dict(senses or {})

    out: List[Dict[str, float]] = []
    scales: Dict[str, tuple] = {}
    for k in keys:
        if k in bounds:
            lo, hi = bounds[k]
        else:
            v = np.asarray([r.get(k, float("nan")) for r in rows],
                           dtype=np.float64)
            v = v[np.isfinite(v)]
            if v.size == 0:
                lo, hi = 0.0, 1.0
            else:
                lo = float(v.min()); hi = float(v.max())
        scales[k] = (lo, hi)
    for r in rows:
        norm: Dict[str, float] = {}
        for k in keys:
            lo, hi = scales[k]
            v = float(r.get(k, float("nan")))
            if not math.isfinite(v):
                norm[k] = float("nan")
                continue
            if hi <= lo:
                norm[k] = 0.5
                continue
            x = (v - lo) / (hi - lo)
            if senses.get(k, +1) == -1:
                x = 1.0 - x
            norm[k] = float(np.clip(x, 0.0, 1.0))
        out.append(norm)
    return out


# ═══════════════════════════════════════════ MetricsLog ══════════════════════


@dataclass
class MetricsLog:
    """Append-only log of metric measurements with CSV / JSON export.

    Each row is a ``dict`` of metric values plus optional metadata fields
    (``label``, ``timestamp``, anything user-supplied via ``log()``).
    Suitable for accumulating measurements across many chords / frames /
    experiments and exporting them for downstream analysis.
    """

    rows: List[Dict[str, Any]] = field(default_factory=list)

    def log(self, **fields: Any) -> None:
        """Append one row. ``fields`` is the per-row data."""
        self.rows.append(dict(fields))

    def log_geometry(
        self,
        geom: GeometryData,
        label: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Convenience: compute :func:`geometry_metrics` and append the row."""
        row = geometry_metrics(geom)
        row.setdefault("label", label or f"geom_{len(self.rows)}")
        row["geom_type"] = geom.geom_type
        row.update(extra)
        self.rows.append(row)

    def log_sequence(
        self,
        seq: HarmonicSequence,
        generator: Callable[..., GeometryData],
        label_prefix: str = "frame",
        generator_kwargs: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> None:
        """Apply ``generator`` to each frame and log the resulting
        geometry-metrics row.

        Parameters
        ----------
        seq : HarmonicSequence
        generator : callable(HarmonicInput, **kw) -> GeometryData
        label_prefix : str, default ``"frame"``
        generator_kwargs : dict, optional
            Forwarded to ``generator`` for every frame.
        **extra : Any
            Constant fields appended to every row (e.g. ``trial=1``).
        """
        if not callable(generator):
            raise TypeError(
                "log_sequence requires a callable generator. For raw "
                "harmonic-content stats over a windowed signal use "
                "biotuner.metrics directly."
            )
        gen_kwargs = dict(generator_kwargs or {})
        for i, frame in enumerate(seq.frames):
            try:
                geom = generator(frame, **gen_kwargs)
                row = geometry_metrics(geom)
                row["geom_type"] = geom.geom_type
            except Exception as exc:
                row = {"error": str(exc)}
            row["label"] = f"{label_prefix}_{i}"
            row["t"] = float(seq.times[i]) if seq.times is not None else float(i)
            row.update(extra)
            self.rows.append(row)

    def to_dict(self) -> Dict[str, List[Any]]:
        """Return a column-oriented dict (DataFrame-like)."""
        if not self.rows:
            return {}
        keys: List[str] = []
        for row in self.rows:
            for k in row:
                if k not in keys:
                    keys.append(k)
        return {k: [row.get(k) for row in self.rows] for k in keys}

    def to_csv(self, path: Union[str, Path]) -> Path:
        """Write the log as CSV (header = column names, one row per entry)."""
        import csv
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = list(self.to_dict().keys())
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(cols)
            for row in self.rows:
                writer.writerow([row.get(c, "") for c in cols])
        return path

    def to_json(self, path: Union[str, Path]) -> Path:
        """Write the log as JSON (a list of row dicts)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.rows, fh, indent=2, default=str)
        return path

    def __len__(self) -> int:
        return len(self.rows)


__all__ = [
    "geometry_metrics",
    "list_supported_kinds",
    "sequence_metrics",
    "compare",
    "normalize_metrics",
    "MetricsLog",
]
