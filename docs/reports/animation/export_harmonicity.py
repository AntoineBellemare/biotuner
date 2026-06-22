"""
Export the real biotuner harmonic-similarity curve for the didactic reel.

dyad_similarity(ratio) = ((x + y - 1) / (x*y)) * 100 on the rationalised ratio
(Gill & Purves 2009) — high where two tones share many overtones. We sample it
densely across one octave (the "consonance landscape" backdrop) and at the
named interval stops the animation parks on.

Writes public/harmonicity.json. Run: python export_harmonicity.py
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # repo root

from biotuner.metrics import dyad_similarity  # noqa: E402

# Perceptual tolerance for the continuous landscape: snap each ratio to the
# nearest fraction with denominator ≤ this before scoring, so a *band* of
# nearby ratios reads as the simple interval (otherwise dyad_similarity's
# limit_denominator(1000) makes the curve zero except at exact fractions).
LANDSCAPE_MAXDEN = 16

OUT = HERE / "public" / "harmonicity.json"

# Named just-intonation stops the slide parks on (unison → octave).
STOPS = [
    (1, 1, "unison"),
    (6, 5, "minor third"),
    (5, 4, "major third"),
    (4, 3, "perfect fourth"),
    (3, 2, "perfect fifth"),
    (5, 3, "major sixth"),
    (2, 1, "octave"),
]


def main() -> None:
    n = 700
    curve = []
    for i in range(n):
        r = 1.0 + (i / (n - 1))  # 1.0 .. 2.0
        snapped = float(Fraction(r).limit_denominator(LANDSCAPE_MAXDEN))
        curve.append(round(float(dyad_similarity(snapped)), 3))

    stops = [
        {
            "num": a, "den": b, "ratio": a / b, "label": lab,
            "sim": round(float(dyad_similarity(a / b)), 2),
        }
        for (a, b, lab) in STOPS
    ]

    payload = {
        "curve": curve,            # 700 samples of dyad_similarity over r∈[1,2]
        "curve_min_r": 1.0,
        "curve_max_r": 2.0,
        "sim_max": 100.0,
        "n_harmonics": 6,          # teeth per comb
        "stops": stops,
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB)")
    for s in stops:
        print(f"  {s['num']}/{s['den']:<2} {s['label']:14s} sim={s['sim']:.1f}")


if __name__ == "__main__":
    main()
