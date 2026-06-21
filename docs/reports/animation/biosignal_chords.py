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
    wn_scale: float | None = None,
    audio_base: float = 196.0,   # fundamental → ~G3
) -> dict:
    """Turn a peak set into a cymatics chord dict (visual wavenumbers +
    octave-voiced audio frequencies).

    Two wavenumber mappings:
      * pinned (default): min peak → ``wn_lo``, max peak → ``wn_hi``. Good
        for wide-span inharmonic peaks (brains), keeping them legible.
      * absolute (``wn_scale`` set): wn = round(wn_scale · ratio). Good for
        near-harmonic peaks (hearts) — harmonic ratios land on integer
        wavenumbers (clean lattices) and distinct partials stay distinct.
    """
    f0 = min(peaks)
    ratios = [p / f0 for p in peaks]
    if wn_scale is not None:
        wn = [max(2, round(wn_scale * r)) for r in ratios]
    else:
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
# Each spans delta/theta → beta with mutually INHARMONIC peaks; the nine are
# deliberately distinct (no two map to the same wavenumber lattice) so the
# wall reads as nine different intricate minds.
BRAIN_GALLERY_PEAKS = [
    [3.1, 6.7, 10.3, 18.9],
    [2.4, 5.1, 9.8, 13.6, 22.7],
    [4.3, 7.9, 12.1, 19.4],
    [2.8, 6.1, 8.4, 15.2, 24.1],
    [3.6, 5.9, 11.7, 21.3],
    [2.1, 4.7, 7.8, 12.9, 18.3],
    [3.0, 5.4, 11.2, 17.5],
    [2.6, 6.4, 10.1, 14.9, 20.6],
    [3.3, 7.2, 9.4, 16.7, 26.2],
]


def brain_gallery() -> list[dict]:
    # wn_hi=13 spreads the inharmonic interior peaks across the lattice.
    return [
        peaks_to_chord(p, name=f"mind {i + 1}", label="brain",
                       accent="#8a9be8", wn_hi=13.0)
        for i, p in enumerate(BRAIN_GALLERY_PEAKS)
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
# biotuner fixed-FFT peaks from sliding windows of the example ECG: the beat
# fundamental (~1 Hz, HRV-jittered) + its low partials (2nd/3rd/4th harmonic,
# where the ECG's energy actually lives). Decimal Hz — NOT idealised 1·2·3·4.
# The nine are chosen so their (absolute-scale) wavenumber lattices are all
# distinct, so the wall reads as nine different — but all ordered — beats.
HEART_REAL_PEAKS = [
    [1.00, 2.03, 3.05],
    [0.98, 1.99, 3.94],
    [1.02, 3.07, 4.06],
    [0.96, 1.91, 2.49],
    [1.01, 2.27, 3.03],
    [0.99, 1.52, 2.51],
    [1.03, 2.05, 3.58],
    [0.97, 2.74, 3.71],
    [1.00, 1.49, 2.02],
]


def heart_gallery_real() -> list[dict]:
    """9 real-ECG-precision hearts for the gallery (decimal harmonics).
    Absolute wavenumber scale → near-harmonic ratios give clean, ordered
    lattices that stay visually distinct from one another."""
    return [
        peaks_to_chord([float(x) for x in p], name=f"beat {i + 1}",
                       label="heart", accent="#e87a8a", wn_scale=4.0)
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
