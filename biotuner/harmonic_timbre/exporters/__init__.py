"""biotuner.harmonic_timbre.exporters — write Timbres to musician-facing formats.

Module type: Subpackage

Each exporter takes a :class:`~biotuner.harmonic_timbre.Timbre` and writes
files into a target directory. Every exporter calls
:func:`~biotuner.harmonic_timbre.cross_modal.write_sidecar` so the output
bundle always travels with provenance metadata and a visual fingerprint.

v1 exporters
------------
tuning_files
    ``.scl`` + ``.kbm`` (Scala scale + keyboard mapping) for any synth
    that supports microtuning import.
to_wav
    Multi-pitch WAV sample pack + JSON manifest. Foundation for SFZ
    and Kontakt-style samplers.
to_sfz
    SFZ document + WAV samples + tuning files + sidecar; loads
    directly into Sforzando, sfizz, Decent Sampler.
to_wavetable
    Single-cycle wavetable WAV with Vital / Serum / Surge XT / generic
    header conventions.
to_surge
    Wavetable + ``.scl`` + ``.kbm`` + README bundle aimed at the Surge
    XT microtonal community.
to_csound
    Self-contained ``.csd`` (CsOptions + CsInstruments + optional CsScore).
to_supercollider
    ``.scd`` with a SynthDef + ``Tuning.new`` + optional ``Pbind`` demo.

Convenience
-----------
export_full_bundle
    One call → multi-format kit ready for DAW / live use.
"""

from biotuner.harmonic_timbre.exporters._common import (
    write_manifest,
    bundle_paths,
)
from biotuner.harmonic_timbre.exporters.tuning_files import (
    export_scl,
    export_kbm,
    export_tuning_files,
)
from biotuner.harmonic_timbre.exporters.to_wav import export_wav_pack
from biotuner.harmonic_timbre.exporters.to_sfz import export_sfz
from biotuner.harmonic_timbre.exporters.to_wavetable import (
    export_wavetable,
    export_wavetable_from_imfs,
    export_wavetable_morph,
)
from biotuner.harmonic_timbre.exporters.to_surge import export_surge_bundle
from biotuner.harmonic_timbre.exporters.to_csound import export_csound
from biotuner.harmonic_timbre.exporters.to_supercollider import (
    export_supercollider,
)
from biotuner.harmonic_timbre.exporters.to_vital import (
    to_vital_spectral,
    to_vital_inharmonic,
    to_vital_wavetable_morph,
    to_vital_fm,
    to_vital_pac_driven,
    to_vital_cfc_driven,
    to_vital_hilbert_modulator,
    to_vital_imf_lfo_bank,
    to_vital_imf_stack,
    to_vital_hilbert_sample,
    to_vital_polyrhythm_gated,
    to_vital_markov_macro,
    to_vital_ensemble,
)
from biotuner.harmonic_timbre.exporters._bundle import (
    export_full_bundle,
    export_full_bundle_sequence,
)
from biotuner.harmonic_timbre.exporters.combinatorial import (
    EffectSpec,
    EnvelopeSpec,
    FilterSpec,
    LFOSpec,
    MacroSpec,
    OscSpec,
    PresetSpec,
    build_combinatorial_sources,
    random_spec,
    spec_from_biotuner,
    to_vital_combinatorial,
)

__all__ = [
    "write_manifest",
    "bundle_paths",
    "export_scl",
    "export_kbm",
    "export_tuning_files",
    "export_wav_pack",
    "export_sfz",
    "export_wavetable",
    "export_wavetable_from_imfs",
    "export_wavetable_morph",
    "export_surge_bundle",
    "export_csound",
    "export_supercollider",
    "to_vital_spectral",
    "to_vital_inharmonic",
    "to_vital_wavetable_morph",
    "to_vital_fm",
    "to_vital_pac_driven",
    "to_vital_cfc_driven",
    "to_vital_hilbert_modulator",
    "to_vital_imf_lfo_bank",
    "to_vital_imf_stack",
    "to_vital_hilbert_sample",
    "to_vital_polyrhythm_gated",
    "to_vital_markov_macro",
    "to_vital_ensemble",
    "export_full_bundle",
    "export_full_bundle_sequence",
    "PresetSpec",
    "OscSpec",
    "LFOSpec",
    "FilterSpec",
    "EnvelopeSpec",
    "EffectSpec",
    "MacroSpec",
    "to_vital_combinatorial",
    "random_spec",
    "build_combinatorial_sources",
    "spec_from_biotuner",
]
