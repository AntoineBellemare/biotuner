"""
Biosignal → harmonic geometry for the "Brain vs Heart" reel.

biotuner extracts spectral peaks from each signal; those peaks form a
"chord" the cymatics scene can render. The science the reel tells:

  * BRAIN (EEG) peaks land in the classic bands — theta / alpha / beta /
    gamma — which are mutually INHARMONIC (ratios like 1 : 2.25 : 3.33 : 7.4).
    Inharmonic peaks → an intricate, complex geometry.
  * HEART (ECG) is quasi-periodic: a fundamental beat plus its HARMONIC
    series (1 : 2 : 3 : 4). Harmonic peaks → a clean, ordered geometry.

Peaks below are biotuner-extracted from the bundled example recording
(assets/biosignals/eeg_ecg_example.csv, sf = 100 Hz) via the fixed-FFT
peak function; see ``extract_peaks()`` to reproduce. They're frozen here so
the reel export is fast and deterministic.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "assets" / "biosignals" / "eeg_ecg_example.csv"

# ── biotuner-extracted peaks (Hz) from the example signals ────────────────
# BRAIN: fixed-FFT peaks of the EEG channel — theta(3, 6.75) alpha(10)
# beta(22.25) gamma(42). Genuinely inharmonic.
BRAIN_PEAKS = [3.0, 6.75, 10.0, 22.25, 42.0]
BRAIN_BANDS = ["θ", "θ", "α", "β", "γ"]

# HEART: a periodic beat = fundamental + harmonic series. (The raw ECG
# extraction returns the 1 Hz fundamental and upper partials; a clean
# periodic beat is, by definition, a harmonic series — shown here as such.)
HEART_PEAKS = [1.0, 2.0, 3.0, 4.0, 5.0]
HEART_BANDS = ["f₀", "2f₀", "3f₀", "4f₀", "5f₀"]


def _octave_reduce(x: float, lo: float, hi: float) -> float:
    while x >= hi:
        x /= 2.0
    while x < lo:
        x *= 2.0
    return x


def peaks_to_chord(
    peaks: list[float],
    *,
    name: str,
    label: str,
    accent: str,
    wn_lo: float = 4.0,
    wn_hi: float = 12.0,
    audio_base: float = 196.0,   # fundamental → ~G3
) -> dict:
    """Turn a peak set into a cymatics chord dict (visual wavenumbers +
    octave-voiced audio frequencies)."""
    f0 = min(peaks)
    ratios = [p / f0 for p in peaks]
    rmin, rmax = min(ratios), max(ratios)
    span = (rmax - rmin) or 1.0
    wn = [max(2, round(wn_lo + (r - rmin) / span * (wn_hi - wn_lo)))
          for r in ratios]

    # Audio: fundamental at audio_base, others octave-reduced into 2 octaves
    # so inharmonic brain peaks cluster (beating = "complex") while harmonic
    # heart peaks stack consonantly.
    freqs = sorted({round(audio_base * _octave_reduce(r, 1.0, 4.0), 2)
                    for r in ratios})

    return {
        "name": name,
        "label": label,
        "ratio_str": " · ".join(f"{p:g}Hz" for p in peaks),
        "tag": None,
        "accent": accent,
        "ratios": wn,
        "freqs": freqs,
    }


def brain_heart_chords() -> list[dict]:
    """The reel's chord sequence: brain ↔ heart, alternating to contrast."""
    brain = peaks_to_chord(BRAIN_PEAKS, name="BRAIN", label="brain",
                           accent="#8a9be8")
    heart = peaks_to_chord(HEART_PEAKS, name="HEART", label="heart",
                           accent="#e87a8a")
    # Brain, Heart, Brain, Heart — two cycles so the contrast lands twice.
    return [brain, heart, brain, heart]


# ── Galleries — many brains vs many hearts ────────────────────────────────
# 9 brains: biotuner fixed-FFT peaks from 9 EEG channels of EEG_example.npy.
# They share a 7/14/28.5 Hz backbone but differ in the low (delta/theta)
# bands — all mutually inharmonic → intricate patterns.
BRAIN_GALLERY_PEAKS = [
    [2.5, 3.5, 7.0, 14.0, 28.5],
    [5.0, 7.0, 14.0, 28.5],
    [4.5, 7.0, 14.0, 28.5],
    [4.0, 7.0, 14.0, 28.5],
    [2.5, 5.0, 7.0, 14.0, 28.5],
    [2.0, 4.5, 7.0, 14.0, 28.5],
    [2.5, 3.5, 7.0, 14.0, 28.5],
    [3.5, 7.0, 14.0, 28.5],
    [5.0, 7.0, 14.0, 28.5],
]
# 9 hearts: low-complexity HARMONIC subsets (a periodic beat) — all ordered.
HEART_GALLERY_PEAKS = [
    [2, 3, 4], [3, 4, 5], [2, 3, 5], [3, 4, 6], [2, 4, 5],
    [3, 5, 6], [2, 3, 4, 5], [4, 5, 6], [3, 4, 5, 6],
]


def brain_gallery() -> list[dict]:
    return [
        peaks_to_chord(p, name=f"mind {i + 1}", label="brain", accent="#8a9be8")
        for i, p in enumerate(BRAIN_GALLERY_PEAKS)
    ]


def heart_gallery() -> list[dict]:
    return [
        peaks_to_chord([float(x) for x in p], name=f"beat {i + 1}",
                       label="heart", accent="#e87a8a")
        for i, p in enumerate(HEART_GALLERY_PEAKS)
    ]


# ── Meditative EEG sequence — many channels, precise peaks, smooth morph ──
# biotuner fixed-FFT peaks across 16 EEG channels of EEG_example.npy. They
# share a 7 / 14.25 / 28.5 Hz backbone and drift in the low bands → a long,
# gently-evolving sequence to morph a single cymatics plate through.
MEDITATIVE_EEG_PEAKS = [
    [2.5, 5.25, 7.0, 14.25, 28.5],
    [2.25, 4.25, 7.0, 14.25, 28.5],
    [2.25, 3.5, 7.0, 14.25, 28.5],
    [2.5, 3.5, 7.0, 14.25, 28.5],
    [2.0, 5.0, 7.0, 14.25, 28.5],
    [2.25, 6.75, 7.0, 14.25, 28.5],
    [2.75, 5.5, 7.0, 14.25, 28.5],
    [4.25, 7.0, 7.0, 14.25, 28.5],
    [2.0, 3.5, 7.0, 14.25, 28.5],
    [2.25, 5.0, 7.0, 14.25, 28.5],
    [2.25, 3.75, 7.0, 14.25, 28.5],
    [3.5, 5.25, 7.0, 14.25, 28.5],
]


def meditative_eeg_sequence() -> list[dict]:
    """A long sequence of EEG-derived cymatics chords for the meditative reel
    (all 5-peak so the morph is smooth)."""
    return [
        peaks_to_chord(p, name=f"eeg {i + 1}", label="brain", accent="#8a9be8")
        for i, p in enumerate(MEDITATIVE_EEG_PEAKS)
    ]


# ── Real-precision ECG hearts (decimal harmonics, not idealised integers) ──
# biotuner fixed-FFT peaks from sliding windows of the example ECG: the 1 Hz
# fundamental + its (slightly inharmonic, HRV-jittered) upper partials.
# Fundamental + 2 strongest partials (real ECG decimal Hz) — kept to 3 peaks
# so each heart stays a clean, ordered pattern (the 4th high partial made
# some busy and muddied the brain-vs-heart contrast).
HEART_REAL_PEAKS = [
    [1.0, 5.55, 10.25],
    [1.0, 3.7, 10.2],
    [1.0, 3.8, 12.1],
    [1.0, 3.9, 9.9],
    [1.0, 3.95, 9.85],
    [1.0, 4.05, 10.1],
    [1.0, 3.6, 9.7],
    [1.0, 4.1, 12.3],
    [1.0, 3.75, 10.0],
]


def heart_gallery_real() -> list[dict]:
    """9 real-ECG-precision hearts for the gallery (decimal harmonics)."""
    return [
        peaks_to_chord([float(x) for x in p], name=f"beat {i + 1}",
                       label="heart", accent="#e87a8a")
        for i, p in enumerate(HEART_REAL_PEAKS)
    ]


def extract_peaks(signal_col: str = "EEG", sf: int = 100,
                  mn: float = 2.0, mx: float = 45.0,
                  n_peaks: int = 5) -> list[float]:
    """Reproduce the peak extraction from the bundled CSV (slow; not used at
    export time). Requires the biotuner env."""
    import csv
    import sys
    sys.path.insert(0, str(HERE.parents[2]))
    import numpy as np
    from biotuner.biotuner_object import compute_biotuner

    vals = []
    with open(DATA, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vals.append(float(row[signal_col]))
    bt = compute_biotuner(sf=sf, peaks_function="fixed", precision=0.25)
    bt.peaks_extraction(np.array(vals), n_peaks=n_peaks,
                        min_freq=mn, max_freq=mx)
    return [round(float(p), 2) for p in bt.peaks]


if __name__ == "__main__":
    for c in brain_heart_chords()[:2]:
        print(f"{c['name']}: wn={c['ratios']}  freqs={c['freqs']}  "
              f"({c['ratio_str']})")
