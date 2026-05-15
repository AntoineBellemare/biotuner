"""Tests for biotuner.harmonic_geometry.media.base — Medium / Pipeline / Domains."""

from fractions import Fraction

import numpy as np
import pytest

from biotuner.harmonic_geometry import HarmonicInput
from biotuner.harmonic_geometry.geometry_data import GeometryData
from biotuner.harmonic_geometry.media import (
    Box3D,
    Circular,
    ClosedSurface,
    Domain,
    Granular,
    Interference,
    Medium,
    Pipeline,
    PolygonDomain,
    Rectangular,
    RigidPlate,
    Sphere,
)


@pytest.fixture
def major():
    return HarmonicInput(
        ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)],
        amplitudes=[1.0, 0.7, 0.8],
    )


# ============================================================== Domains


class TestDomains:
    def test_domain_subclasses(self):
        for cls in (Rectangular, Circular, PolygonDomain, Box3D, Sphere):
            assert issubclass(cls, Domain)

    def test_rectangular_defaults(self):
        r = Rectangular()
        assert r.Lx == 1.0 and r.Ly == 1.0

    def test_immutable(self):
        # frozen dataclass — assignment must raise.
        r = Rectangular(Lx=2.0, Ly=3.0)
        with pytest.raises((AttributeError, Exception)):
            r.Lx = 5.0  # type: ignore[misc]


# ============================================================ RigidPlate


class TestRigidPlate:
    def test_family(self):
        assert RigidPlate.family == "eigenmode"

    def test_default_source_is_none(self):
        assert RigidPlate().default_source() is None

    def test_rectangular_field(self, major):
        out = RigidPlate(domain=Rectangular(1.0, 1.0), resolution=64).respond(major)
        assert isinstance(out, GeometryData)
        assert out.geom_type == "field_2d"
        assert out.coordinates.shape == (64, 64)
        assert np.all(np.isfinite(out.coordinates))

    def test_circular_field_has_nans(self, major):
        out = RigidPlate(domain=Circular(1.0), resolution=64).respond(major)
        # Outside-disk cells are NaN.
        assert np.isnan(out.coordinates).any()
        assert np.isfinite(out.coordinates).any()

    def test_box_3d(self, major):
        out = RigidPlate(domain=Box3D(1, 1, 1), resolution=16).respond(major)
        assert out.geom_type == "field_3d"
        assert out.coordinates.shape == (16, 16, 16)

    def test_polygon(self, major):
        out = RigidPlate(domain=PolygonDomain(n_sides=6, radius=1.0),
                         resolution=32).respond(major)
        assert out.geom_type == "field_2d"

    def test_rejects_non_harmonic_input(self):
        with pytest.raises(TypeError):
            RigidPlate().respond(np.zeros((4, 4)))

    def test_rejects_wrong_domain(self):
        with pytest.raises(TypeError):
            RigidPlate(domain=Sphere())  # eigenmode plate doesn't take Sphere

    def test_overrides_forwarded(self, major):
        out = RigidPlate(domain=Rectangular(1, 1)).respond(major, resolution=32)
        assert out.coordinates.shape == (32, 32)

    def test_callable_shorthand(self, major):
        out = RigidPlate(resolution=32)(major)
        assert out.geom_type == "field_2d"


# ========================================================== ClosedSurface


class TestClosedSurface:
    def test_family(self):
        assert ClosedSurface.family == "eigenmode"

    def test_field_output(self, major):
        out = ClosedSurface(n_theta=32, n_phi=64).respond(major)
        assert out.geom_type == "field_2d"
        assert out.coordinates.shape == (32, 64)

    def test_mesh_output(self, major):
        out = ClosedSurface(n_theta=32, n_phi=64, output="mesh").respond(major)
        assert out.geom_type == "mesh_3d"

    def test_rejects_bad_output(self):
        with pytest.raises(ValueError):
            ClosedSurface(output="bogus")


# ========================================================== Interference


class TestInterference:
    def test_family(self):
        assert Interference.family == "wave_field"

    @pytest.mark.parametrize(
        "paradigm",
        ["harmonic", "quasicrystal", "standing_lattice", "vortex", "sources"],
    )
    def test_each_paradigm(self, paradigm, major):
        out = Interference(paradigm=paradigm).respond(major, resolution=64)
        assert out.geom_type == "field_2d"
        assert out.coordinates.shape == (64, 64)

    def test_rejects_unknown_paradigm(self):
        with pytest.raises(ValueError):
            Interference(paradigm="nope")


# ============================================================== Pipeline


class _IdentityMedium(Medium):
    family = "transport"

    def default_source(self) -> Medium:
        return RigidPlate(resolution=32)

    def respond(self, forcing, **overrides):
        field_data = self._resolve_field(forcing)
        return field_data


class TestPipeline:
    def test_empty_pipeline_rejected(self):
        with pytest.raises(ValueError):
            Pipeline()

    def test_non_medium_stage_rejected(self):
        with pytest.raises(TypeError):
            Pipeline(RigidPlate(), object())  # type: ignore[arg-type]

    def test_chains_correctly(self, major):
        plate = RigidPlate(resolution=32)
        sand = Granular(temperature=0.1)
        pipe = Pipeline(plate, sand)
        out = pipe(major)
        assert out.geom_type == "field_2d"
        assert out.metadata["family"] == "transport"

    def test_repr(self):
        pipe = Pipeline(RigidPlate(), Granular())
        assert "RigidPlate" in repr(pipe)
        assert "Granular" in repr(pipe)


# ====================================================== _resolve_field


class TestResolveField:
    def test_passes_geometrydata_through(self, major):
        plate = RigidPlate(resolution=32).respond(major)
        # An identity medium that just returns the resolved field.
        ident = _IdentityMedium()
        out = ident.respond(plate)
        assert out is plate

    def test_auto_wraps_harmonic_input(self, major):
        ident = _IdentityMedium()
        out = ident.respond(major)
        # auto-wrap calls default_source().respond(forcing); for _IdentityMedium
        # the default source is RigidPlate(32).
        assert isinstance(out, GeometryData)
        assert out.geom_type == "field_2d"
        assert out.coordinates.shape == (32, 32)

    def test_no_default_source_raises(self, major):
        class NoSource(Medium):
            family = "transport"

            def respond(self, forcing, **overrides):
                return self._resolve_field(forcing)

        with pytest.raises(TypeError, match="default_source"):
            NoSource().respond(major)

    def test_bad_type_raises(self):
        ident = _IdentityMedium()
        with pytest.raises(TypeError, match="HarmonicInput or GeometryData"):
            ident.respond(42)  # type: ignore[arg-type]
