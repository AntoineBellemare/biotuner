"""
Export biotuner rhythm-module data for two biorhythm reels:

  ① brain_polyrhythm.json  — EEG peaks → ratios → a 5:7:11 polyrhythm
       (scale2polyrhythm_continuous: each peak-ratio becomes an evenly-pulsed
        voice; coincidences_continuous marks where voices reunite). Plus the
        Euclidean form (scale2polyrhythm / bjorklund) for the "even spread" beat.

  ② heart_brain.json       — ECG heartbeat (real R-peaks + HRV) as a master
       pulse, with the SAME brain voices layered on top, and the grid columns
       where brain pulses land on a heartbeat (the groove accents).

Run: python export_polyrhythm.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch, find_peaks

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
warnings.filterwarnings("ignore")

from biotuner.rhythm_construction import (  # noqa: E402
    scale2polyrhythm, scale2polyrhythm_continuous, coincidences_continuous,
    bjorklund, scale2frac,
)

ASSETS = HERE / "assets" / "biosignals"
PUB = HERE / "public"


def downsample(a, n):
    a = np.asarray(a, float)
    idx = np.linspace(0, len(a) - 1, n).astype(int)
    return a[idx]


def eeg_peaks(n_peaks=4):
    eeg = np.load(ASSETS / "EEG_example.npy")
    sf = 250
    fE, pE = welch(eeg, fs=sf, nperseg=sf * 4, axis=1)
    ch = int(np.argmax(pE[:, (fE >= 8) & (fE <= 12)].sum(axis=1)))
    sig = eeg[ch] * 1e6
    f, p = welch(sig, fs=sf, nperseg=sf * 4)
    fmax = 45
    m = f <= fmax
    f, p = f[m], p[m] / p[m].max()
    band = f >= 4
    pk, _ = find_peaks(np.where(band, p, 0), prominence=0.02, distance=max(1, len(p) // 60))
    pk = np.sort(pk[np.argsort(p[pk])[::-1][:n_peaks]])
    peaks = [round(float(f[i]), 2) for i in pk]
    amps = [round(float(p[i]), 3) for i in pk]
    # waveform snippet + spectrum for the didactic extraction beat
    wave = sig[: int(2.4 * sf)]
    wave = wave / (np.max(np.abs(wave)) or 1)
    spectrum = {
        "wave": [round(float(x), 4) for x in downsample(wave, 520)],
        "spec_f": [round(float(x), 3) for x in downsample(f, 300)],
        "spec_mag": [round(float(x), 4) for x in downsample(p, 300)],
        "fmax": fmax,
    }
    return peaks, amps, spectrum


def brain_voices(ratios):
    onset_times, labels, counts, dur = scale2polyrhythm_continuous(ratios, duration=1.0)
    voices = []
    for onsets, p in zip(onset_times, counts):
        voices.append({"pulses": int(p),
                       "onsets": [round(float(t), 5) for t in onsets],
                       "ioi_ms": round(1000.0 / p, 1)})
    return voices, counts


def main() -> None:
    peaks, amps, spectrum = eeg_peaks()
    minf = min(peaks)
    ratios = [round(p / minf, 4) for p in peaks]
    voices, counts = brain_voices(ratios)

    # Euclidean form (for the "spread evenly" beat)
    e_voices, coinc, cpos, e_labels, lcm = scale2polyrhythm(ratios)
    euclid = []
    for pat, lab in zip(e_voices, e_labels):
        # one native period of the pattern
        # lab = "E(d,n)" → d pulses over n steps
        d, n = (int(x) for x in lab[2:-1].split(","))
        euclid.append({"label": lab, "steps": n, "pulses": d, "pattern": bjorklund(n, d)})

    # continuous coincidences (where ≥2 brain voices reunite within a cycle)
    onset_times = [v["onsets"] for v in voices]
    cc = coincidences_continuous(onset_times, tolerance_sec=0.012)
    brain_coinc = [{"t": round(c["time"], 5), "voices": c["voices"]} for c in cc]

    brain = {
        "peaks": peaks, "amps": amps, "ratios": ratios,
        "voices": voices, "counts": [int(c) for c in counts],
        "poly_label": ":".join(str(c) for c in counts),
        "euclid": euclid, "lcm": int(lcm),
        "coincidences": brain_coinc,
        **spectrum,
    }
    (PUB / "brain_polyrhythm.json").write_text(json.dumps(brain, separators=(",", ":")), encoding="utf-8")

    # ── ECG heartbeat ────────────────────────────────────────────────────────
    df = pd.read_csv(ASSETS / "eeg_ecg_example.csv")
    ecg = df["ECG"].values.astype(float)
    sf = 100
    ecg = ecg - ecg.mean()
    ecg_n = ecg / (np.max(np.abs(ecg)) or 1)
    rpk, _ = find_peaks(ecg, height=ecg.std() * 1.5, distance=int(0.4 * sf))
    r_times = (rpk / sf).tolist()
    ibi = np.diff(rpk) / sf
    hr = 60.0 / float(np.mean(ibi))
    hrv = float(np.std(ibi) * 1000)

    heart = {
        "ecg": [round(float(x), 4) for x in downsample(ecg_n, 1000)],
        "dur_s": round(len(ecg) / sf, 3),
        "r_times": [round(t, 4) for t in r_times],
        "ibi_s": [round(float(x), 4) for x in ibi],
        "hr_bpm": round(hr, 1), "hrv_ms": round(hrv, 1),
        "brain_voices": voices, "counts": [int(c) for c in counts],
        "poly_label": ":".join(str(c) for c in counts),
    }
    (PUB / "heart_brain.json").write_text(json.dumps(heart, separators=(",", ":")), encoding="utf-8")

    print(f"brain  peaks {peaks}  ratios {ratios}  poly {brain['poly_label']}")
    print(f"       euclid {[e['label'] for e in euclid]}  coincidences {len(brain_coinc)}")
    print(f"heart  {hr:.1f} bpm  HRV {hrv:.0f} ms  {len(r_times)} beats over {heart['dur_s']}s")


if __name__ == "__main__":
    main()
