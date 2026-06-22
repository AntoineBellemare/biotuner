"""
Export biotuner subharmonic-tension data for the "Subharmonic Tension" reel.

compute_subharmonic_tension (Chan et al. 2019, via biotuner.metrics) scores how
hard a chord is to fold onto one common subharmonic (a shared virtual
fundamental): ~0 when the notes resolve to a single low fundamental, higher
when they cannot. Scored on the canonical small-integer form (the metric is
scale-sensitive); the reel positions/sounds the notes at ratio × base_freq.

Writes public/subharmonicity.json. Run: python export_subharm.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
warnings.filterwarnings("ignore")

from biotuner.metrics import compute_subharmonic_tension  # noqa: E402

OUT = HERE / "public" / "subharmonicity.json"
N_HARM = 10
DELTA = 20.0

# harmonic (one clear home) → inharmonic (no home), by the metric.
CHORDS = [
    ("Major  4:5:6",        [4, 5, 6]),
    ("Detuned",             [4, 5.3, 6.6]),
    ("Augmented 16:20:25",  [16, 20, 25]),
    ("Inharmonic 31:51:71", [31, 51, 71]),
]


def tension_of(chord):
    _, _, t, _ = compute_subharmonic_tension(list(chord), N_HARM, DELTA, min_notes=2)
    try:
        return float(t[0])
    except Exception:
        return float("nan")


def main() -> None:
    rows = []
    for label, chord in CHORDS:
        t = tension_of(chord)
        m = min(chord)
        ratios = [round(c / m, 4) for c in chord]
        rows.append({"label": label, "ratios": ratios, "tension": round(t, 5)})
        print(f"  {label:22s} tension={t:.4f}  ratios={ratios}")

    tmax = max(r["tension"] for r in rows) or 1.0
    for r in rows:
        r["tension_norm"] = round(min(1.0, r["tension"] / tmax), 4)

    payload = {
        "base_freq": 174.61,   # F3 — bottom voice
        "n_subharm": 15,       # subharmonic ticks drawn per note (deep stacks)
        "chords": rows,
        "tension_max": round(tmax, 5),
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB)  tmax={tmax:.4f}")


if __name__ == "__main__":
    main()
