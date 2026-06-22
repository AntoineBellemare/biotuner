"""
Export data for the "signal → peaks → dissonance curve" reel (the biotuner
Sethares pipeline), for a synthetic signal AND real EEG.

Pipeline per signal:
  1. spectrum  (Welch PSD)
  2. spectral PEAKS + their amplitudes  (the harmonic content)
  3. the dissonance COMB: hold the peaks, sweep a copy unison→octave, sum
     Plomp-Levelt roughness over every peak pair  →  biotuner dissmeasure
  4. the curve's valleys = the consonant scale that signal implies

Writes public/diss_pipeline.json. Run: python export_diss_pipeline.py
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import welch, find_peaks

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from biotuner.scale_construction import dissmeasure  # noqa: E402

OUT = HERE / "public" / "diss_pipeline.json"
GRID = [(6, 5), (5, 4), (4, 3), (3, 2), (5, 3), (2, 1)]


def downsample(a, n):
    a = np.asarray(a, float)
    if len(a) <= n:
        return a
    idx = np.linspace(0, len(a) - 1, n).astype(int)
    return a[idx]


def spectrum(sig, sf, fmax, n_peaks, fmin=0.0):
    f, p = welch(sig, fs=sf, nperseg=min(len(sig), int(sf * 4)))
    m = f <= fmax
    f, p = f[m], p[m]
    p = p / (p.max() or 1)
    band = f >= fmin  # drop sub-band drift (e.g. EEG delta) from peak picking
    pk, _ = find_peaks(np.where(band, p, 0), prominence=0.02, distance=max(1, len(p) // 60))
    order = np.argsort(p[pk])[::-1][:n_peaks]
    pk = np.sort(pk[order])
    peaks = [{"f": round(float(f[i]), 2), "a": round(float(np.sqrt(p[i])), 3)} for i in pk]
    return f, p, peaks


AUDIO_F = 261.63  # peak RATIOS are transposed to an audible fundamental so the
# Plomp-Levelt critical band has meaningful structure (biotuner's EEG→tuning idea)


def diss_curve(peaks, n=460, max_ratio=2):
    pf = np.array([p["f"] for p in peaks], float)
    pf = pf / pf.min() * AUDIO_F          # ratios → audio
    pa = np.array([p["a"] for p in peaks], float)
    amps = np.concatenate((pa, pa))
    alphas = np.linspace(1.0, max_ratio, n)
    d = np.empty(n)
    for i, al in enumerate(alphas):
        d[i] = dissmeasure(np.concatenate((pf, al * pf)), amps, model="min")
    dm = float(d.max()) or 1.0
    dn = d / dm
    # valleys = prominent local minima of the curve (prominence, not absolute)
    vi, _ = find_peaks(-dn, prominence=0.03, distance=max(1, n // 30))
    out = []
    for i in vi:
        a = float(alphas[i])
        fr = Fraction(a).limit_denominator(9)
        on = any(abs(a - x / y) < 0.013 for x, y in GRID)
        out.append({"alpha": round(a, 4),
                    "label": f"{fr.numerator}/{fr.denominator}" if on else f"{a:.2f}",
                    "on_grid": on, "diss": round(float(dn[i]), 4)})
    return dn.round(4).tolist(), out


def build(label, sub, sig, sf, fmax, n_peaks, wave_secs, fmin=0.0):
    f, p, peaks = spectrum(sig, sf, fmax, n_peaks, fmin=fmin)
    curve, valleys = diss_curve(peaks)
    wave = sig[: int(wave_secs * sf)]
    wave = wave / (np.max(np.abs(wave)) or 1)
    return {
        "label": label, "sublabel": sub, "sf": sf, "fmax": fmax,
        "wave": [round(float(x), 4) for x in downsample(wave, 520)],
        "spec_f": [round(float(x), 3) for x in downsample(f, 300)],
        "spec_mag": [round(float(x), 4) for x in downsample(p, 300)],
        "peaks": peaks, "curve": curve, "valleys": valleys,
    }


def main() -> None:
    rng = np.random.RandomState(3)

    # ── synthetic signal: a few partials + noise ─────────────────────────────
    sf = 250
    t = np.arange(10 * sf) / sf
    comps = [(18, 1.0), (27, 0.8), (36, 0.6), (54, 0.45)]  # 2:3:4:6 — structured
    sig = sum(a * np.sin(2 * np.pi * fr * t) for fr, a in comps) + 0.04 * rng.randn(len(t))
    signal = build("a signal", "synthetic · 4 components", sig, sf, 80, 4, 0.7)

    # ── real EEG (occipital-ish channel with clear rhythms) ──────────────────
    eeg = np.load(HERE / "assets" / "biosignals" / "EEG_example.npy")
    sf_e = 250
    # pick the channel with the most alpha-band power
    fE, pE = welch(eeg, fs=sf_e, nperseg=min(eeg.shape[1], sf_e * 4), axis=1)
    alpha_band = (fE >= 8) & (fE <= 12)
    ch = int(np.argmax(pE[:, alpha_band].sum(axis=1)))
    eeg_sig = eeg[ch] * 1e6  # → microvolts
    eeg_out = build("your brain", "EEG · same pipeline", eeg_sig, sf_e, 45, 4, 2.0, fmin=4.0)

    payload = {"grid": [{"num": n, "den": d, "ratio": n / d} for n, d in GRID],
               "signal": signal, "eeg": eeg_out}
    OUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB)")
    for s in (signal, eeg_out):
        print(f"  {s['label']:10s} peaks {[p['f'] for p in s['peaks']]}  "
              f"valleys {[v['label'] for v in s['valleys']]}")


if __name__ == "__main__":
    main()
