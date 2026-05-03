"""Tests for biotuner.harmonic_geometry.metrics (geometry-only API)."""
from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from biotuner.harmonic_geometry import (
    HarmonicInput, HarmonicSequence,
    MetricsLog, compare, geometry_metrics,
    list_supported_kinds, normalize_metrics, sequence_metrics,
)
from biotuner.harmonic_geometry.lissajous import (
    lissajous_2d, lissajous_3d, lissajous_compound,
)
from biotuner.harmonic_geometry.harmonograph import harmonograph_lateral
from biotuner.harmonic_geometry.polygon_circular import (
    star_polygon, times_table_circle, tuning_circle, rose_curve, epicycloid,
)
from biotuner.harmonic_geometry.fractal import (
    stern_brocot_tree, continued_fraction_rectangles, farey_sequence_layout,
    subharmonic_tree, ifs_harmonic,
)
from biotuner.harmonic_geometry.generative import (
    lsystem_from_ratios, recursive_polygon, self_similar_tuning,
)
from biotuner.harmonic_geometry.geometry_3d import (
    lissajous_tube, harmonic_knot, harmonic_surface, lsystem_3d,
    recursive_polyhedron, harmonic_point_cloud,
)
from biotuner.harmonic_geometry.chladni import (
    chladni_field_rectangular, chladni_field_circular, chladni_field_3d_box,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def major():
    return HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)])

@pytest.fixture
def minor():
    return HarmonicInput(ratios=[Fraction(1), Fraction(6, 5), Fraction(3, 2)])

@pytest.fixture
def dom7():
    return HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2),
                                  Fraction(7, 4)])


# ── geometry_metrics + per-method extractors ─────────────────────────────────

class TestGeometryMetrics:
    def test_curve_2d(self):
        g = lissajous_2d(Fraction(3, 2), n_points=200)
        m = geometry_metrics(g)
        assert m["n_vertices"] == 200.0
        assert "span_x" in m and "span_y" in m

    def test_mesh_3d(self, major):
        g = lissajous_tube(major, n_points=100, n_sides=8)
        m = geometry_metrics(g)
        assert m["n_vertices"] == float(100 * 8)
        assert m["n_faces"] > 0
        assert m["edge_len_mean"] > 0
        assert "surface_area" in m

    def test_field_2d(self):
        g = chladni_field_rectangular([(2, 3), (3, 5)], resolution=64)
        m = geometry_metrics(g)
        for k in ("field_min", "field_max", "field_mean", "field_std",
                  "zero_crossing_frac"):
            assert k in m
        assert 0.0 <= m["zero_crossing_frac"] <= 1.0


class TestPerMethodExtractors:
    """Smoke test per registered kind."""

    def test_supported_kinds_count(self):
        assert len(list_supported_kinds()) >= 30

    def test_lissajous_2d(self):
        g = lissajous_2d(Fraction(3, 2), n_points=200)
        m = geometry_metrics(g)
        assert m["kind"] == "lissajous_2d"
        for k in ("lobes_x", "lobes_y", "closed", "perimeter"):
            assert k in m

    def test_lissajous_3d(self):
        g = lissajous_3d([3, 4, 5], n_points=400)
        m = geometry_metrics(g)
        assert m["kind"] == "lissajous_3d"
        assert "is_knot" in m and "perimeter" in m

    def test_lissajous_compound(self, major):
        g = lissajous_compound(major, n_points=400)
        m = geometry_metrics(g)
        assert m["kind"] == "lissajous_compound"
        assert "n_components" in m

    def test_harmonograph_lateral(self):
        inp = HarmonicInput(peaks=[2.01, 3.02, 5.0],
                            damping=[0.05, 0.04, 0.06])
        g = harmonograph_lateral(inp, sr=200)
        m = geometry_metrics(g)
        assert m["kind"] == "harmonograph_lateral"
        for k in ("early_amp", "late_amp", "decay_ratio", "perimeter"):
            assert k in m

    def test_chladni_rectangular(self):
        g = chladni_field_rectangular([(2, 3), (3, 5), (4, 1)], resolution=64)
        m = geometry_metrics(g)
        assert m["kind"] == "chladni_field_rectangular"
        assert "n_modes" in m and "energy" in m and "peak_abs" in m

    def test_chladni_circular(self):
        g = chladni_field_circular([1, 2], [0, 1], R=1.0, resolution=64)
        m = geometry_metrics(g)
        assert m["kind"] == "chladni_field_circular"

    def test_chladni_3d_box(self):
        g = chladni_field_3d_box([(2, 3, 4)], resolution=24)
        m = geometry_metrics(g)
        assert m["kind"] == "chladni_field_3d_box"
        assert "zero_crossing_frac_3d" in m

    def test_star_polygon(self):
        g = star_polygon(7, 2)
        m = geometry_metrics(g)
        assert m["kind"] == "star_polygon"
        assert "compound" in m

    def test_times_table_circle(self):
        g = times_table_circle(n_points=60, multiplier=2.0)
        m = geometry_metrics(g)
        assert m["kind"] == "times_table_circle"
        assert "multiplier" in m and "unique_targets" in m

    def test_tuning_circle(self, major):
        g = tuning_circle(major)
        m = geometry_metrics(g)
        assert m["kind"] == "tuning_circle"
        assert "n_pitches" in m and "pitch_class_dispersion" in m

    def test_rose(self):
        g = rose_curve(Fraction(5, 2), n_points=400)
        m = geometry_metrics(g)
        assert m["kind"] == "rose"
        assert "perimeter" in m

    def test_epicycloid(self):
        g = epicycloid(Fraction(1, 4), n_points=400)
        m = geometry_metrics(g)
        assert m["kind"] == "epicycloid"
        assert "cusps" in m

    def test_stern_brocot(self, major):
        g = stern_brocot_tree(input=major, max_depth=5)
        m = geometry_metrics(g)
        assert m["kind"] == "stern_brocot_tree"
        for k in ("harmonicity_mean", "harmonicity_std",
                  "min_chord_dist_cents"):
            assert k in m

    def test_continued_fraction(self):
        g = continued_fraction_rectangles(Fraction(7, 4))
        m = geometry_metrics(g)
        assert m["kind"] == "continued_fraction_rectangles"
        assert "n_squares" in m

    def test_farey(self):
        g = farey_sequence_layout(8)
        m = geometry_metrics(g)
        assert m["kind"] == "farey_sequence_layout"
        assert "n_terms" in m and "denom_max" in m

    def test_subharmonic_tree(self, major):
        g = subharmonic_tree(major, depth=3, n_harmonics=3)
        m = geometry_metrics(g)
        assert m["kind"] == "subharmonic_tree"
        for k in ("n_nodes", "n_octaves_spanned", "min_freq", "max_freq"):
            assert k in m

    def test_ifs(self, major):
        g = ifs_harmonic(major, n_points=2000, rng=np.random.default_rng(0))
        m = geometry_metrics(g)
        assert m["kind"] == "ifs_harmonic"
        assert "fractal_dim" in m and "span" in m

    def test_lsystem(self, major):
        g = lsystem_from_ratios(major, depth=3)
        m = geometry_metrics(g)
        assert m["kind"] == "lsystem_from_ratios"
        for k in ("n_segments", "lstring_length", "angle_deg", "total_length"):
            assert k in m

    def test_recursive_polygon(self, major):
        g = recursive_polygon(major, depth=2)
        m = geometry_metrics(g)
        assert m["kind"] == "recursive_polygon"
        for k in ("perimeter", "area", "scale_factor", "depth"):
            assert k in m

    def test_self_similar_tuning(self, major):
        g = self_similar_tuning(major, n_levels=3)
        m = geometry_metrics(g)
        assert m["kind"] == "self_similar_tuning"
        assert "total_nodes" in m and "n_generations" in m

    def test_lissajous_tube(self, major):
        g = lissajous_tube(major, n_points=100, n_sides=8)
        m = geometry_metrics(g)
        assert m["kind"] == "lissajous_tube"
        assert "rx" in m and "ry" in m and "rz" in m

    def test_harmonic_knot(self, major):
        g = harmonic_knot(major, n_points=200, n_sides=8)
        m = geometry_metrics(g)
        assert m["kind"] == "harmonic_knot"
        assert "winding_p" in m and "winding_q" in m

    def test_harmonic_surface(self, major):
        g = harmonic_surface(major, mode="torus", resolution=24)
        m = geometry_metrics(g)
        assert m["kind"] == "harmonic_surface"
        assert "deformation" in m

    def test_lsystem_3d(self, major):
        g = lsystem_3d(major, depth=2)
        m = geometry_metrics(g)
        assert m["kind"] == "lsystem_3d"
        assert "n_segments" in m

    def test_recursive_polyhedron(self, major):
        g = recursive_polyhedron(major, depth=1, solid="icosahedron")
        m = geometry_metrics(g)
        assert m["kind"] == "recursive_polyhedron"
        for k in ("face_ratio_entropy", "face_ratio_n_unique",
                  "depth", "per_face_bump"):
            assert k in m

    def test_harmonic_point_cloud(self, major):
        g = harmonic_point_cloud(major, n_points=200, surface="sphere")
        m = geometry_metrics(g)
        assert m["kind"] == "harmonic_point_cloud"
        assert "actual_n_points" in m
        assert "field_min" in m and "field_max" in m


# ── sequence_metrics ──────────────────────────────────────────────────────────

class TestSequenceMetrics:
    def test_geometry_trajectory(self, major, minor, dom7):
        seq = HarmonicSequence(frames=[major, minor, dom7])
        traj = sequence_metrics(seq, harmonic_knot, n_points=100, n_sides=6)
        # All knots produce winding_p and winding_q
        assert "winding_p" in traj
        assert "winding_q" in traj
        assert traj["winding_p"].shape == (3,)
        assert traj["winding_q"].shape == (3,)

    def test_with_chladni_generator(self, major, dom7):
        from biotuner.harmonic_geometry.chladni import chladni_from_input
        seq = HarmonicSequence(frames=[major, dom7])
        traj = sequence_metrics(
            seq, chladni_from_input,
            plate="rectangular",
            plate_kwargs={"resolution": 32},
        )
        assert "energy" in traj
        assert traj["energy"].shape == (2,)

    def test_no_generator_raises(self, major):
        seq = HarmonicSequence(frames=[major])
        with pytest.raises(TypeError):
            sequence_metrics(seq)            # type: ignore[call-arg]

    def test_failing_generator_keeps_alignment(self, major):
        seq = HarmonicSequence(frames=[major, major])
        def bad_generator(inp, **kw):
            raise RuntimeError("boom")
        traj = sequence_metrics(seq, bad_generator)
        assert traj == {}    # all frames failed → no keys


# ── compare (geometries only) ─────────────────────────────────────────────────

class TestCompare:
    def test_geometries(self):
        gs = [lissajous_2d(Fraction(3, 2), n_points=100),
              lissajous_2d(Fraction(5, 4), n_points=100)]
        c = compare(gs, labels=["3/2", "5/4"])
        assert c["__labels__"] == ["3/2", "5/4"]
        assert "n_vertices" in c
        assert len(c["n_vertices"]) == 2

    def test_input_raises(self, major):
        with pytest.raises(TypeError):
            compare([major])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compare([])


# ── normalize_metrics ─────────────────────────────────────────────────────────

class TestNormalize:
    def test_data_driven_default(self):
        rows = [{"a": 1.0}, {"a": 5.0}, {"a": 9.0}]
        norm = normalize_metrics(rows)
        assert norm[0]["a"] == pytest.approx(0.0, abs=1e-9)
        assert norm[2]["a"] == pytest.approx(1.0, abs=1e-9)

    def test_zero_variance_maps_to_half(self):
        rows = [{"a": 4.0}, {"a": 4.0}, {"a": 4.0}]
        norm = normalize_metrics(rows)
        for n in norm:
            assert n["a"] == 0.5

    def test_senses_invert(self):
        rows = [{"a": 1.0}, {"a": 9.0}]
        norm = normalize_metrics(rows, senses={"a": -1})
        # With sense=-1, the smaller value becomes the higher normalised value
        assert norm[0]["a"] == pytest.approx(1.0, abs=1e-9)
        assert norm[1]["a"] == pytest.approx(0.0, abs=1e-9)

    def test_explicit_bounds(self):
        rows = [{"a": 5.0}]
        norm = normalize_metrics(rows, bounds={"a": (0.0, 10.0)})
        assert norm[0]["a"] == pytest.approx(0.5, abs=1e-9)

    def test_nan_passes_through(self):
        rows = [{"a": float("nan")}, {"a": 4.0}, {"a": 8.0}]
        norm = normalize_metrics(rows)
        assert math.isnan(norm[0]["a"])


# ── MetricsLog ────────────────────────────────────────────────────────────────

class TestMetricsLog:
    def test_log_geometry(self, major):
        g = lissajous_tube(major, n_points=50, n_sides=6)
        log = MetricsLog()
        log.log_geometry(g, label="major_tube")
        d = log.to_dict()
        assert d["label"] == ["major_tube"]
        assert d["geom_type"] == ["mesh_3d"]
        assert "n_faces" in d

    def test_log_sequence_with_generator(self, major, minor, dom7):
        seq = HarmonicSequence(
            frames=[major, minor, dom7],
            times=np.array([0.0, 1.5, 3.0]),
        )
        log = MetricsLog()
        log.log_sequence(seq, harmonic_knot,
                          generator_kwargs={"n_points": 80, "n_sides": 6})
        d = log.to_dict()
        assert len(log) == 3
        assert d["t"] == [0.0, 1.5, 3.0]
        assert "winding_p" in d

    def test_log_sequence_requires_generator(self, major):
        seq = HarmonicSequence(frames=[major])
        log = MetricsLog()
        with pytest.raises(TypeError):
            log.log_sequence(seq, generator=None)   # type: ignore[arg-type]

    def test_to_csv(self, major, tmp_path):
        g = lissajous_tube(major, n_points=50, n_sides=6)
        log = MetricsLog()
        log.log_geometry(g, label="m")
        out = log.to_csv(tmp_path / "log.csv")
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "n_faces" in content and "m" in content

    def test_to_json(self, major, tmp_path):
        g = lissajous_tube(major, n_points=50, n_sides=6)
        log = MetricsLog()
        log.log_geometry(g, label="m")
        out = log.to_json(tmp_path / "log.json")
        import json
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(loaded, list) and len(loaded) == 1
        assert loaded[0]["label"] == "m"

    def test_extra_fields(self, major):
        g = lissajous_tube(major, n_points=50, n_sides=6)
        log = MetricsLog()
        log.log_geometry(g, label="m", trial=1, condition="rest")
        d = log.to_dict()
        assert d["trial"] == [1]
        assert d["condition"] == ["rest"]


# ── plotting helpers (smoke) ──────────────────────────────────────────────────

class TestPlottingHelpers:
    def test_radar_geometry_rows(self, major, minor, dom7):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from biotuner.harmonic_geometry import plotting

        # Build geometries and a metrics row per chord
        knots = [harmonic_knot(c, n_points=100, n_sides=6)
                 for c in (major, minor, dom7)]
        rows = [geometry_metrics(g) for g in knots]
        fig, ax = plotting.plot_metric_radar(
            rows, labels=["maj", "min", "dom7"],
            metrics=["winding_p", "winding_q",
                     "n_vertices", "edge_len_mean", "surface_area"],
        )
        plt.close(fig)

    def test_trajectory_with_generator(self, major, minor, dom7):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from biotuner.harmonic_geometry import plotting

        seq = HarmonicSequence(frames=[major, minor, dom7])
        fig, ax = plotting.plot_metric_trajectory(
            seq, generator=harmonic_knot,
            generator_kwargs={"n_points": 100, "n_sides": 6},
            metrics=["winding_p", "n_vertices"],
            normalize=True,
        )
        plt.close(fig)

    def test_trajectory_pre_computed(self, major):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from biotuner.harmonic_geometry import plotting

        traj = {"winding_p": np.array([2.0, 3.0, 5.0]),
                "n_vertices": np.array([100.0, 200.0, 400.0])}
        fig, ax = plotting.plot_metric_trajectory(traj, normalize=True)
        plt.close(fig)
