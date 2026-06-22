"""
Export data for "Fractal Rhythm" — biotuner's second_order_polyrhythm.

A polyrhythm's multi-voice coincidences form a new rhythm; its IOI ratios seed a
fresh polyrhythm — rhythm nested inside rhythm. The just-major scale gives a rich
two-level hierarchy (8 voices / 12 coincidences → 5 voices). We export each
level's voice onsets + coincidences, normalised to one cycle [0,1).

Writes public/fractal.json. Run: python export_fractal.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

from biotuner.rhythm_construction import second_order_polyrhythm  # noqa: E402

OUT = Path(__file__).resolve().parent / "public" / "fractal.json"
SCALE = [1, 9 / 8, 5 / 4, 4 / 3, 3 / 2, 5 / 3, 15 / 8, 2]   # just major
DUR = 3.0


def main() -> None:
    res = second_order_polyrhythm(SCALE, duration=DUR, n_orders=2, max_denom=16)
    orders = res[0] if isinstance(res, tuple) else res

    out_orders = []
    for o in orders:
        voices = [{"pulses": int(p), "onsets": [round(t / DUR, 5) for t in onsets]}
                  for p, onsets in zip(o["pulse_counts"], o["onset_times"])]
        coinc = [{"t": round(t / DUR, 5), "n": int(nv)}
                 for t, nv in zip(o["coinc_times"], o["voice_count"])]
        out_orders.append({
            "order": int(o["order"]),
            "pulse_counts": [int(p) for p in o["pulse_counts"]],
            "poly_label": " : ".join(str(p) for p in o["pulse_counts"]),
            "voices": voices,
            "coincidences": coinc,
            "n_coinc": int(o["n_coinc"]),
            "ioi_scale": [round(float(x), 3) for x in o.get("ioi_scale", [])],
        })
        print(f"  order {o['order']}: {len(voices)} voices {o['pulse_counts']}  "
              f"{o['n_coinc']} coincidences  ioi_scale={[round(x,2) for x in o.get('ioi_scale',[])]}")

    OUT.write_text(json.dumps({"orders": out_orders}, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
