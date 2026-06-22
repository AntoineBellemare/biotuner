"""
Export the real Sethares / Plomp-Levelt sensory-dissonance curve for reel ③.

Two complex tones (a harmonic timbre: fundamental + N partials, decaying amps).
The lower tone is fixed at f0; the upper sweeps from f0 to 2·f0. At each interval
the total sensory dissonance is the sum of Plomp-Levelt roughness over every pair
of partials — biotuner.scale_construction.dissmeasure. The curve dips into
VALLEYS at the simple ratios (octave, fifth, fourth, thirds…): those minima are
the consonant intervals a given timbre "wants". (Sethares 1993.)

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
F0 = 261.63       # C4 — dissonance is frequency-dependent (critical band)
N_PARTIALS = 6
N = 600           # sweep resolution over the octave


def main() -> None:
    ks = np.arange(1, N_PARTIALS + 1)
    base_f = F0 * ks
    base_a = 0.88 ** (ks - 1)            # gently decaying harmonic timbre
    amps = np.concatenate((base_a, base_a))

    alphas = np.linspace(1.0, 2.0, N)
    diss = np.empty(N)
    for i, alpha in enumerate(alphas):
        fvec = np.concatenate((base_f, alpha * base_f))
        diss[i] = dissmeasure(fvec, amps, model="min")

    dmax = float(diss.max()) or 1.0
    diss_norm = (diss / dmax).round(4).tolist()

    # local minima = consonant valleys; label by nearest simple ratio
    valleys = []
    for i in range(2, N - 2):
        if diss[i] < diss[i - 1] and diss[i] <= diss[i + 1] and diss[i] < 0.55 * dmax:
            a = alphas[i]
            fr = Fraction(a).limit_denominator(9)
            valleys.append({
                "alpha": round(float(a), 4),
                "num": fr.numerator, "den": fr.denominator,
                "diss_norm": round(float(diss[i] / dmax), 4),
            })
    # de-dupe valleys that round to the same ratio (keep the deepest)
    best: dict[str, dict] = {}
    for v in valleys:
        key = f"{v['num']}/{v['den']}"
        if key not in best or v["diss_norm"] < best[key]["diss_norm"]:
            best[key] = {**v, "label": key}
    valleys = sorted(best.values(), key=lambda v: v["alpha"])

    payload = {
        "f0": F0, "n_partials": N_PARTIALS,
        "curve": diss_norm,            # N samples of normalised dissonance over α∈[1,2]
        "diss_max": round(dmax, 5),
        "valleys": valleys,
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB)")
    for v in valleys:
        print(f"  valley a={v['alpha']:.3f}  {v['label']:5s}  diss={v['diss_norm']:.2f}")


if __name__ == "__main__":
    main()
