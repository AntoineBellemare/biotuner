"""
Export biotuner subharmonic-tension data for the "Subharmonic Tension" reel.

compute_subharmonic_tension (Chan et al. 2019, via biotuner.metrics) scores how
hard a chord is to fold onto common subharmonics (shared virtual fundamentals):
~0 when the notes' subharmonic ladders line up, higher when they cannot.

We also export the ACTUAL shared subharmonics the metric sees. Following the
metric, subharmonics live in PERIOD space (ms): note i's k-th subharmonic has
period 1000·k/i. Two subharmonics from different notes are "shared" when their
periods fall within `delta_lim` ms — so NEAR-alignments count, not just exact
ones. We cluster them (single-linkage in period) and keep clusters touched by
≥2 notes; a cluster touched by ALL notes is a common fundamental ("home").
Each subharmonic maps to a plot frequency = (i/k)·base/min(chord).

Writes public/subharmonicity.json. Run: python export_subharm.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
warnings.filterwarnings("ignore")

from biotuner.metrics import compute_subharmonic_tension  # noqa: E402

OUT = HERE / "public" / "subharmonicity.json"
N_HARM = 12          # subharmonics per note
DELTA = 45.0         # ms — near-alignment tolerance (larger ⇒ more shared subs)
BASE = 174.61        # F3 — bottom voice

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


def alignments(chord):
    """Shared-subharmonic clusters in period space (faithful to the metric).

    Complete-linkage (every member within DELTA ms of every other ⇒ no chaining),
    ≥2 distinct notes, and the shared frequency must sit BELOW the lowest note
    (a genuine virtual fundamental, not the notes themselves)."""
    minc = min(chord)
    scale = BASE / minc
    note_lo = minc * scale  # = BASE
    subs = []  # (note_i, k, plot_freq, period_ms)
    for i, c in enumerate(chord):
        for k in range(1, N_HARM + 1):
            intf = c / k
            subs.append((i, k, intf * scale, 1000.0 / intf))
    subs.sort(key=lambda s: s[3])  # by period

    # complete-linkage clustering: grow while the whole window stays within DELTA
    clusters, cur = [], [subs[0]]
    for s in subs[1:]:
        if s[3] - cur[0][3] < DELTA:
            cur.append(s)
        else:
            clusters.append(cur)
            cur = [s]
    clusters.append(cur)

    aligns = []
    n_notes = len(chord)
    for cl in clusters:
        notes = {s[0] for s in cl}
        if len(notes) < 2:
            continue
        periods = [s[3] for s in cl]
        freq = float(np.mean([s[2] for s in cl]))
        if freq > note_lo * 0.985:       # keep only sub-fundamentals
            continue
        spread = max(periods) - min(periods)
        tight = round(max(0.0, 1.0 - spread / DELTA), 3)  # 1 = perfect alignment
        aligns.append({
            "freq": round(freq, 2),
            "members": [{"i": s[0], "k": s[1]} for s in cl],
            "n_notes": len(notes),
            "spread": round(spread, 2),
            "tight": tight,
            "full": len(notes) == n_notes,
        })
    aligns.sort(key=lambda a: a["freq"])
    return aligns


def main() -> None:
    rows = []
    for label, chord in CHORDS:
        t = tension_of(chord)
        minc = min(chord)
        ratios = [round(c / minc, 4) for c in chord]
        al = alignments(chord)
        full = [a for a in al if a["full"]]
        home = min((a["freq"] for a in full), default=None)
        rows.append({
            "label": label, "ratios": ratios, "tension": round(t, 5),
            "alignments": al, "home": home,
            "n_shared": len(al), "n_full": len(full),
        })
        print(f"  {label:22s} tension={t:.4f}  shared={len(al):2d}  full={len(full)}  home={home}")

    tmax = max(r["tension"] for r in rows) or 1.0
    for r in rows:
        r["tension_norm"] = round(min(1.0, r["tension"] / tmax), 4)

    payload = {
        "base_freq": BASE,
        "n_subharm": N_HARM,
        "delta_lim": DELTA,
        "chords": rows,
        "tension_max": round(tmax, 5),
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB)  tmax={tmax:.4f}")


if __name__ == "__main__":
    main()
