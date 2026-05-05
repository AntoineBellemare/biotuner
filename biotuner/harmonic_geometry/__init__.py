"""
biotuner.harmonic_geometry
==========================

Module type: Subpackage

Pure-data geometric structures derived from harmonic inputs (ratios, peaks,
amplitudes, phases). This module produces structured numpy / dataclass output;
rendering is the responsibility of downstream layers.

Submodules: chladni, fractal, generative, geometry_3d, geometry_data,
harmonograph, inputs, lissajous, metrics, plotting, polygon_circular,
spherical_harmonics.
"""

from biotuner.harmonic_geometry.geometry_data import GeometryData, GeomType
from biotuner.harmonic_geometry.inputs import HarmonicInput, HarmonicSequence

# Lissajous
from biotuner.harmonic_geometry.lissajous import (
    lissajous_2d,
    lissajous_3d,
    lissajous_compound,
    lissajous_pairwise_grid,
    lissajous_phase_drift,
    lissajous_topology,
)

# Harmonograph
from biotuner.harmonic_geometry.harmonograph import (
    derive_damping_from_linewidth,
    harmonograph_3d,
    harmonograph_from_peaks,
    harmonograph_lateral,
    harmonograph_rotary,
)

# Chladni
from biotuner.harmonic_geometry.chladni import (
    chladni_field_3d_box,
    chladni_field_circular,
    chladni_field_polygon,
    chladni_field_rectangular,
    chladni_from_input,
    chladni_nodal_lines,
    chladni_nodal_surfaces,
    chladni_temporal,
    ratios_to_modes,
)

# Spherical harmonics (eigenmodes of the Laplacian on the unit sphere —
# the closed-surface analogue of Chladni plate modes; also the basis used
# in higher-order ambisonics).
from biotuner.harmonic_geometry.spherical_harmonics import (
    ratios_to_modes_lm,
    single_spherical_harmonic,
    spherical_harmonic_field,
    spherical_harmonic_from_input,
    spherical_harmonic_mesh,
    spherical_harmonic_temporal,
)

# Polygon / circular
from biotuner.harmonic_geometry.polygon_circular import (
    consonance_polygon,
    epicycloid,
    hypocycloid,
    interval_vector_diagram,
    polygon_chord_pattern,
    rose_curve,
    star_polygon,
    times_table_circle,
    times_table_from_input,
    tuning_circle,
)

# Fractal (Phase 4 deterministic)
from biotuner.harmonic_geometry.fractal import (
    continued_fraction_rectangles,
    farey_sequence_layout,
    ifs_harmonic,
    stern_brocot_tree,
    subharmonic_tree,
)

# Generative / fractal (Phase 5) + pipeline utility
from biotuner.harmonic_geometry.generative import (
    geometry_sequence,
    lsystem_from_ratios,
    recursive_polygon,
    self_similar_tuning,
)

# 3-D geometry (Phase 7)
from biotuner.harmonic_geometry.geometry_3d import (
    harmonic_knot,
    harmonic_point_cloud,
    harmonic_surface,
    lissajous_tube,
    lsystem_3d,
    recursive_polyhedron,
)

# Plotting (cross-phase visualisation)
from biotuner.harmonic_geometry import plotting  # noqa: F401

# Metrics monitoring (do science with it)
from biotuner.harmonic_geometry import metrics  # noqa: F401
from biotuner.harmonic_geometry.metrics import (
    MetricsLog,
    compare,
    geometry_metrics,
    list_supported_kinds,
    normalize_metrics,
    sequence_metrics,
)

__all__ = [
    # Core data
    "GeometryData",
    "GeomType",
    "HarmonicInput",
    "HarmonicSequence",
    # Lissajous
    "lissajous_2d",
    "lissajous_3d",
    "lissajous_compound",
    "lissajous_pairwise_grid",
    "lissajous_phase_drift",
    "lissajous_topology",
    # Harmonograph
    "derive_damping_from_linewidth",
    "harmonograph_3d",
    "harmonograph_from_peaks",
    "harmonograph_lateral",
    "harmonograph_rotary",
    # Chladni
    "chladni_field_3d_box",
    "chladni_field_circular",
    "chladni_field_polygon",
    "chladni_field_rectangular",
    "chladni_from_input",
    "chladni_nodal_lines",
    "chladni_nodal_surfaces",
    "chladni_temporal",
    "ratios_to_modes",
    # Spherical harmonics (closed-surface eigenmodes)
    "ratios_to_modes_lm",
    "single_spherical_harmonic",
    "spherical_harmonic_field",
    "spherical_harmonic_from_input",
    "spherical_harmonic_mesh",
    "spherical_harmonic_temporal",
    # Polygon / circular
    "consonance_polygon",
    "epicycloid",
    "hypocycloid",
    "interval_vector_diagram",
    "polygon_chord_pattern",
    "rose_curve",
    "star_polygon",
    "times_table_circle",
    "times_table_from_input",
    "tuning_circle",
    # Fractal (Phase 4)
    "continued_fraction_rectangles",
    "farey_sequence_layout",
    "ifs_harmonic",
    "stern_brocot_tree",
    "subharmonic_tree",
    # Generative / fractal (Phase 5) + pipeline utility
    "geometry_sequence",
    "lsystem_from_ratios",
    "recursive_polygon",
    "self_similar_tuning",
    # 3-D geometry (Phase 7)
    "harmonic_knot",
    "harmonic_point_cloud",
    "harmonic_surface",
    "lissajous_tube",
    "lsystem_3d",
    "recursive_polyhedron",
    # Plotting submodule
    "plotting",
    # Metrics monitoring
    "metrics",
    "MetricsLog",
    "compare",
    "geometry_metrics",
    "list_supported_kinds",
    "normalize_metrics",
    "sequence_metrics",
]
