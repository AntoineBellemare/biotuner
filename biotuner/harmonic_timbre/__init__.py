"""biotuner.harmonic_timbre

Synthesize spectra (timbres) matched to tunings derived from biosignal analysis.

The module is organized around a single canonical object — a ratio set,
regardless of how it was produced — with multiple projections that share
matching code, consonance metrics, and provenance metadata.

Phase 1 surface (v1.0 foundation):
- ``Timbre``, ``TimbreSequence``, ``Modulator`` dataclasses.
- Matching algorithms: ``match_direct``, ``match_consonance_weighted``,
  ``match_sethares``, ``match_harmonic_entropy``, ``match_hybrid``,
  ``match_timbre`` (dispatcher).
- Inharmonic partial series: ``stretched_partials``, ``compressed_partials``,
  ``inharmonic_string``, ``gamelan_partials``, ``custom_partial_series``,
  ``inharmonic_timbre``.
- Synthesis: ``render_additive``, ``render_with_envelope``,
  ``render_band_limited``, ``render_wavetable_cycle``.
- Biotuner mapping (v1 fields): ``timbre_from_biotuner``, ``timbre_from_ratios``,
  per-mapping helpers, ``DEFAULT_MAPPING_V1``, ``ALL_MAPPINGS``.
- Cross-modal sidecar: ``palette_from_peaks``, ``geometry_signature_image``,
  ``write_sidecar``.
- Scale source abstraction: ``SCALE_SOURCES``, ``resolve_scale``.
"""

from biotuner.harmonic_timbre.timbre import (
    Timbre,
    TimbreSequence,
    Modulator,
)
from biotuner.harmonic_timbre.matching import (
    match_timbre,
    match_direct,
    match_sethares,
    match_harmonic_entropy,
    match_consonance_weighted,
    match_hybrid,
)
from biotuner.harmonic_timbre.biotuner_mapping import (
    timbre_from_biotuner,
    timbre_from_ratios,
    map_peaks_to_partials,
    map_amps_to_amplitudes,
    map_phases,
    map_linewidth_to_decay,
    map_aperiodic_to_tilt,
    map_flatness_to_noise,
    map_harmonicity_weights,
    map_consonance_priors,
    map_pac_to_am_modulators,
    map_cfc_to_fm_modulators,
    map_intermod_to_modulators,
    DEFAULT_MAPPING_V1,
    ALL_MAPPINGS,
)
from biotuner.harmonic_timbre.inharmonic import (
    stretched_partials,
    compressed_partials,
    inharmonic_string,
    gamelan_partials,
    custom_partial_series,
    inharmonic_timbre,
)
from biotuner.harmonic_timbre.synthesis import (
    render_additive,
    render_with_envelope,
    render_band_limited,
    render_modulated,
    render_wavetable_cycle,
)
from biotuner.harmonic_timbre.cross_modal import (
    geometry_signature_image,
    write_sidecar,
)
from biotuner.harmonic_timbre.direct_synth import (
    hilbert_instrument,
    fm_patch_from_tuning,
)
from biotuner.harmonic_timbre.sequence_sources import (
    timbre_sequence_from_ratio_frames,
    timbre_sequence_from_signal,
    timbre_sequence_from_transitional_harmony,
    timbre_sequence_from_markov_walk,
)
from biotuner.harmonic_timbre._utils import (
    SCALE_SOURCES,
    resolve_scale,
)
from biotuner.harmonic_timbre import exporters  # noqa: F401

__all__ = [
    "Timbre",
    "TimbreSequence",
    "Modulator",
    "match_timbre",
    "match_direct",
    "match_sethares",
    "match_harmonic_entropy",
    "match_consonance_weighted",
    "match_hybrid",
    "timbre_from_biotuner",
    "timbre_from_ratios",
    "map_peaks_to_partials",
    "map_amps_to_amplitudes",
    "map_phases",
    "map_linewidth_to_decay",
    "map_aperiodic_to_tilt",
    "map_flatness_to_noise",
    "map_harmonicity_weights",
    "map_consonance_priors",
    "map_pac_to_am_modulators",
    "map_cfc_to_fm_modulators",
    "map_intermod_to_modulators",
    "DEFAULT_MAPPING_V1",
    "ALL_MAPPINGS",
    "stretched_partials",
    "compressed_partials",
    "inharmonic_string",
    "gamelan_partials",
    "custom_partial_series",
    "inharmonic_timbre",
    "render_additive",
    "render_with_envelope",
    "render_band_limited",
    "render_modulated",
    "render_wavetable_cycle",
    "geometry_signature_image",
    "write_sidecar",
    "hilbert_instrument",
    "fm_patch_from_tuning",
    "timbre_sequence_from_ratio_frames",
    "timbre_sequence_from_signal",
    "timbre_sequence_from_transitional_harmony",
    "timbre_sequence_from_markov_walk",
    "SCALE_SOURCES",
    "resolve_scale",
]
