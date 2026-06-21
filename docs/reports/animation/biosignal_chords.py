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
