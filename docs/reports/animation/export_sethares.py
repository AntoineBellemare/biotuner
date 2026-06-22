"""
Export Sethares / Plomp-Levelt sensory-dissonance curves for several TIMBRES,
to show that the consonant intervals (the scale) match the timbre.

For each timbre (a set of partial ratios + amplitudes) we hold a lower tone and
sweep an upper copy from unison to the octave, summing Plomp-Levelt roughness
over every partial pair (biotuner.scale_construction.dissmeasure). The curve's
VALLEYS are the consonances that timbre prefers:
  * few harmonics  → a few broad valleys (only octave / fifth)
  * many harmonics → many valleys, exactly on the just-intonation grid
  * stretched/inharmonic partials → valleys SHIFT off the just grid (Sethares 1993)

Writes public/sethares.json. Run: python export_sethares.py
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from biotuner.scale_construction import dissmeasure  # noqa: E402

OUT = HERE / "public" / "sethares.json"
F0 = 261.63
N = 600

# just-ratio reference grid drawn on every panel
GRID = [(6, 5), (5, 4), (4, 3), (3, 2), (5, 3), (2, 1)]


def harmonic(n, decay=0.88):
    ks = np.arange(1, n + 1)
    return ks.astype(float), decay ** (ks - 1)


def stretched(n, s=1.0704, decay=0.88):
    # m-th partial at m**s  (s = log2(2.1) → a stretched 2.1:1 "octave")
    ks = np.arange(1, n + 1)
    return ks.astype(float) ** s, decay ** (ks - 1)


TIMBRES = [
    ("few partials · 3", *harmonic(3)),
    ("rich harmonic · 7", *harmonic(7)),
    ("stretched · inharmonic", *stretched(7)),
]


def curve_for(partials, base_a):
    base_f = F0 * np.asarray(partials)
    amps = np.concatenate((base_a, base_a))
    alphas = np.linspace(1.0, 2.0, N)
    diss = np.empty(N)
    for i, alpha in enumerate(alphas):
        fvec = np.concatenate((base_f, alpha * base_f))
        diss[i] = dissmeasure(fvec, amps, model="min")
    dmax = float(diss.max()) or 1.0
    norm = diss / dmax

    valleys = []
    for i in range(3, N - 3):
        if norm[i] < norm[i - 1] and norm[i] <= norm[i + 1] and norm[i] < 0.6:
            a = float(alphas[i])
            fr = Fraction(a).limit_denominator(9)
            on_grid = any(abs(a - n / d) < 0.012 for n, d in GRID)
            valleys.append({"alpha": round(a, 4),
                            "label": f"{fr.numerator}/{fr.denominator}" if on_grid else f"{a:.2f}",
                            "on_grid": on_grid, "diss": round(float(norm[i]), 4)})
    # de-dupe near-duplicate alphas (keep deepest)
    valleys.sort(key=lambda v: v["alpha"])
    dedup = []
    for v in valleys:
        if dedup and abs(v["alpha"] - dedup[-1]["alpha"]) < 0.02:
            if v["diss"] < dedup[-1]["diss"]:
                dedup[-1] = v
        else:
            dedup.append(v)
    return norm.round(4).tolist(), dedup, float(np.argmax(diss) / (N - 1) + 1)


def main() -> None:
    timbres = []
    for label, partials, amps in TIMBRES:
        curve, valleys, peak = curve_for(partials, amps)
        timbres.append({
            "label": label,
            "partials": [round(float(p), 4) for p in partials],
            "amps": [round(float(a), 4) for a in amps],
            "curve": curve, "valleys": valleys, "peak_alpha": round(peak, 4),
        })
        vs = " ".join(v["label"] + ("" if v["on_grid"] else "*") for v in valleys)
        print(f"  {label:24s} valleys: {vs}")

    payload = {"f0": F0, "grid": [{"num": n, "den": d, "ratio": n / d} for n, d in GRID],
               "timbres": timbres}
    OUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB)   (* = off the just grid)")


if __name__ == "__main__":
    main()
