"""Preprocess the Sleep-EDF SC4012 recording into a compact per-stage EEG dataset
for the bioelements notebook (so the notebook is reproducible without the raw EDF).

Writes docs/examples/bioelements/sleep_stages.npz:
    epochs : (n, 3000) float32   — 30 s EEG (Fpz-Cz, µV) epochs
    stages : (n,) <U3            — 'W' | 'N1' | 'N2' | 'N3' | 'REM'
    sf     : float               — 100.0
"""
import warnings; warnings.filterwarnings("ignore")
from collections import defaultdict
from pathlib import Path

import numpy as np
import mne

PSG = r"C:/Users/skite/Documents/Github/goofi-pipe/SC4012E0-PSG.edf"
HYP = r"C:/Users/skite/Documents/Github/goofi-pipe/SC4012EC-Hypnogram.edf"
OUT = Path(__file__).resolve().parents[1] / "docs" / "examples" / "bioelements" / "sleep_stages.npz"

SF = 100
EPOCH = 30 * SF  # 3000 samples
N_PER_STAGE = 12
STAGE_MAP = {
    "Sleep stage W": "W", "Sleep stage 1": "N1", "Sleep stage 2": "N2",
    "Sleep stage 3": "N3", "Sleep stage 4": "N3", "Sleep stage R": "REM",
}
ORDER = ["W", "N1", "N2", "N3", "REM"]

raw = mne.io.read_raw_edf(PSG, preload=True, verbose=False)
sig = raw.get_data(picks="EEG Fpz-Cz")[0] * 1e6   # -> microvolts
ann = mne.read_annotations(HYP)

# per-epoch stage labels from the annotation segments
labels = {}
for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
    st = STAGE_MAP.get(str(desc))
    if st is None:
        continue
    for e in range(int(onset // 30), int((onset + dur) // 30)):
        labels[e] = st

by = defaultdict(list)
for e, st in sorted(labels.items()):
    s = e * EPOCH
    if s + EPOCH <= len(sig):
        seg = sig[s:s + EPOCH]
        if np.std(seg) > 1e-3:            # drop flat/artefact epochs
            by[st].append(seg.astype(np.float32))

epochs, stages = [], []
for st in ORDER:
    arr = by.get(st, [])
    if not arr:
        print(f"  WARNING: no clean epochs for {st}")
        continue
    idx = np.linspace(0, len(arr) - 1, min(N_PER_STAGE, len(arr))).astype(int)
    for i in idx:
        epochs.append(arr[i]); stages.append(st)
    print(f"  {st:4s}: {len(arr):3d} epochs available, {len(idx)} sampled")

epochs = np.asarray(epochs, np.float32)
stages = np.asarray(stages)

# --- contiguous night sequence (for the continuous resonance river) -------- #
# crop to the sleep window (first -> last non-wake epoch, +/- 20 epochs of wake),
# then take one epoch every STRIDE (4 min) so the river stays light to compute.
labelled = sorted(labels.items())
sleep_e = [e for e, st in labelled if st != "W"]
lo = max(0, min(sleep_e) - 20)
hi = min(max(labels), max(sleep_e) + 20)
STRIDE = 8  # every 8th 30 s epoch = every 4 min
night_x, night_s, night_t = [], [], []
for e in range(lo, hi + 1, STRIDE):
    st = labels.get(e)
    s = e * EPOCH
    if st is None or s + EPOCH > len(sig):
        continue
    seg = sig[s:s + EPOCH]
    if np.std(seg) > 1e-3:
        night_x.append(seg.astype(np.float32)); night_s.append(st); night_t.append(e * 30 / 3600.0)
print(f"  night sequence: {len(night_x)} contiguous epochs "
      f"({night_t[0]:.1f}-{night_t[-1]:.1f} h)")

OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT, epochs=epochs, stages=stages, sf=float(SF),
                    night_epochs=np.asarray(night_x, np.float32),
                    night_stages=np.asarray(night_s),
                    night_times_h=np.asarray(night_t, np.float32))
print(f"\nwrote {OUT.name}: {epochs.shape} stage-epochs + {len(night_x)} night-epochs, "
      f"{OUT.stat().st_size/1024:.0f} KB")
