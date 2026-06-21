"""
Re-render the intro-sound auditions to ~/Downloads from the canonical
modules (intro_synths.py + soundscape.py), for choosing/refining the reel
opening. Not part of the render pipeline — a convenience tool.

    python audition_sounds.py            # bed + each of the 5 synths
    python audition_sounds.py --dry      # synths only, no nature bed
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from intro_synths import SYNTHS, write_wav
from soundscape import build_bed

DOWNLOADS = Path.home() / "Downloads"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry", action="store_true",
                    help="Render synths alone, without the nature bed.")
    args = ap.parse_args()

    bed = None if args.dry else build_bed(3.6, peak=0.5)
    if bed is not None:
        write_wav(bed, DOWNLOADS / "intro_natural_bed.wav")
        print("  intro_natural_bed.wav")

    for name, fn in SYNTHS.items():
        synth = fn()
        if bed is None:
            mix = synth
            tag = "intro"
        else:
            n = min(bed.shape[0], synth.shape[0])
            mix = bed[:n] + 0.9 * synth[:n]
            tag = "intro_natural"
        peak = float(np.max(np.abs(mix)))
        if peak > 1e-9:
            mix = mix * (10 ** (-1.0 / 20) / peak)
        out = DOWNLOADS / f"{tag}_{name}.wav"
        write_wav(mix, out)
        print(f"  {out.name}")
    print(f"\nAuditions in: {DOWNLOADS}")


if __name__ == "__main__":
    main()
