"""
Export data for "Brain Grooves" — the world's rhythms hidden in EEG.

biotuner's rhythm module turns each EEG peak-ratio into a Euclidean rhythm
(Bjorklund), then names it via dict_rhythms (Toussaint's world-rhythm catalog).
Scanning EEG_example.npy across channels, the brain reproduces real named
rhythms from Cuba, Arabia, India, Bulgaria — even a Frank Zappa meter. We curate
a striking selection (each verified to appear from the EEG) + the didactic
front-end (waveform → spectrum → peaks).

Writes public/brain_grooves.json. Run: python export_brain_grooves.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, welch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
warnings.filterwarnings("ignore")

from biotuner.dictionaries import dict_rhythms  # noqa: E402
from biotuner.rhythm_construction import bjorklund, interval_vector, scale2euclid  # noqa: E402

OUT = HERE / "public" / "brain_grooves.json"
ASSETS = HERE / "assets" / "biosignals"

# curated brain rhythms (label, pulses, steps, short name, region, blurb)
PICKS = [
    ("E(3,8)", 3, 8, "Tresillo", "Cuba", "the Habanera — under a thousand rock'n'roll songs"),
    ("E(5,7)", 5, 7, "Nawakhat", "Arabia", "a popular Arabic rhythm · 'Al Noht' in Nubia"),
    ("E(4,7)", 4, 7, "Ruchenitza", "Bulgaria", "a Bulgarian folk-dance"),
    ("E(5,11)", 5, 11, "Savari tala", "India", "Hindustani classical · also Bulgaria & Serbia"),
]


def downsample(a, n):
    a = np.asarray(a, float)
    return a[np.linspace(0, len(a) - 1, n).astype(int)]


def eeg_front_end():
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
    pk = np.sort(find_peaks(np.where(band, p, 0), prominence=0.02, distance=max(1, len(p) // 60))[0]
                 [np.argsort(p[find_peaks(np.where(band, p, 0), prominence=0.02, distance=max(1, len(p) // 60))[0]])[::-1][:4]])
    peaks = [round(float(f[i]), 2) for i in pk]
    amps = [round(float(p[i]), 3) for i in pk]
    wave = sig[: int(2.4 * sf)]
    wave = wave / (np.max(np.abs(wave)) or 1)
    return {
        "wave": [round(float(x), 4) for x in downsample(wave, 520)],
        "spec_f": [round(float(x), 3) for x in downsample(f, 300)],
        "spec_mag": [round(float(x), 4) for x in downsample(p, 300)],
        "fmax": fmax, "peaks": peaks, "amps": amps,
    }


def verify_in_eeg():
    """Confirm each picked rhythm really is produced by some EEG channel."""
    eeg = np.load(ASSETS / "EEG_example.npy")
    sf = 250
    seen = set()
    for c in range(eeg.shape[0]):
        f, p = welch(eeg[c] * 1e6, fs=sf, nperseg=sf * 4)
        m = f <= 45
        f, p = f[m], p[m] / (p[m].max() or 1)
        band = f >= 4
        pk = find_peaks(np.where(band, p, 0), prominence=0.02, distance=max(1, len(p) // 60))[0]
        pk = np.sort(pk[np.argsort(p[pk])[::-1][:4]])
        fr = f[pk]
        if len(fr) < 2:
            continue
        ratios = [float(x / fr.min()) for x in fr]
        for pat in scale2euclid(ratios, 16, "normal"):
            if sum(pat) >= 2 and len(set(pat)) > 1:
                seen.add(f"E({sum(pat)},{len(pat)})")
    return seen


def main() -> None:
    seen = verify_in_eeg()
    rhythms = []
    for label, k, n, name, region, blurb in PICKS:
        pat = bjorklund(n, k)
        rhythms.append({
            "label": label, "pulses": k, "steps": n,
            "pattern": pat, "ivec": [int(x) for x in interval_vector(pat)],
            "name": name, "region": region, "blurb": blurb,
            "from_eeg": label in seen,
            "dict": dict_rhythms.get(label, "")[:180],
        })
        print(f"  {label:8s} {name:14s} {region:14s} {'[EEG]' if label in seen else '[ - ]'}  {pat}")

    payload = {"front": eeg_front_end(), "rhythms": rhythms}
    OUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
