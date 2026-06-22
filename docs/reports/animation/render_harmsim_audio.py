"""
Soundtrack for the "Harmonic Similarity" reel (HarmonicSimilarity-IG).

A modern 2-operator FM "key" plays each dyad (root + the interval tone) as the
slide parks on it, over a soft root drone — so you HEAR the consonance land in
sync with the visual bloom. Timeline constants must match
src/scenes/HarmonicSimilarity.tsx.

Run: python render_harmsim_audio.py  →  public/audio/harmonicity.wav
"""
from __future__ import annotations

import json
from pathlib import Path
from wave import open as wave_open

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "public" / "audio" / "harmonicity.wav"
SR = 44_100
FPS = 30

# ── must match HarmonicSimilarity.tsx ───────────────────────────────────────
INTRO = 36
APPROACH = 18
DWELL = 58
BEAT = APPROACH + DWELL
TAIL = 56
ROOT_HZ = 261.63  # C4

stops = json.load(open(HERE / "public" / "harmonicity.json", encoding="utf-8"))["stops"]
N_FRAMES = INTRO + len(stops) * BEAT + TAIL


def fm_key(freq: float, dur: float, *, detune: float = 0.004,
           bright: float = 3.2, decay: float = 3.0,
           attack: float = 0.006, release: float = 0.9) -> np.ndarray:
    """Modern 2-op FM key/bell: carrier + 2× modulator with a bright,
    fast-decaying index (digital pluck → mellow tail). Stereo-detuned. (n,2)."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    left = np.zeros(n)
    right = np.zeros(n)
    for acc, df in ((0, 1.0 - detune), (1, 1.0 + detune)):
        fc = freq * df
        fm = fc * 2.0  # harmonic modulator → clean, glassy FM
        idx = bright * np.exp(-t * decay) + 0.4
        sig = np.sin(2 * np.pi * fc * t + idx * np.sin(2 * np.pi * fm * t))
        # a touch of a 3× layer for modern sparkle
        sig += 0.18 * np.sin(2 * np.pi * fc * t + (idx * 0.6) * np.sin(2 * np.pi * fc * 3 * t))
        env = np.ones(n)
        a = max(int(attack * SR), 1)
        r = max(int(release * SR), 1)
        env[:a] = np.linspace(0, 1, a)
        # gentle exponential body decay so it doesn't sound static
        env *= 0.55 + 0.45 * np.exp(-t * 0.8)
        if r < n:
            env[-r:] *= np.linspace(1, 0, r)
        if acc == 0:
            left += sig * env
        else:
            right += sig * env
    return np.stack([left, right], axis=1)


def root_drone(dur: float) -> np.ndarray:
    """Soft sustained root pad (sub + body) under the whole piece."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    trem = 0.85 + 0.15 * np.sin(2 * np.pi * 0.1 * t)
    mono = (0.5 * np.sin(2 * np.pi * (ROOT_HZ / 2) * t)
            + 0.28 * np.sin(2 * np.pi * ROOT_HZ * t)) * 0.10 * trem
    head = int(1.0 * SR)
    mono[:head] *= np.linspace(0, 1, head)
    return np.stack([mono, mono], axis=1)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    total_n = int(N_FRAMES / FPS * SR) + int(1.0 * SR)
    out = root_drone(total_n / SR)

    for i, s in enumerate(stops):
        park_frame = INTRO + i * BEAT + APPROACH
        start = int(park_frame / FPS * SR)
        dur = (DWELL / FPS) + 1.2
        root = fm_key(ROOT_HZ, dur, bright=2.6) * 0.5
        upper = fm_key(ROOT_HZ * s["ratio"], dur, bright=3.4) * 0.5
        dyad = root + upper
        end = min(start + dyad.shape[0], out.shape[0])
        out[start:end] += dyad[: end - start]

    out = out[: int(N_FRAMES / FPS * SR)]
    peak = float(np.max(np.abs(out)))
    if peak > 1e-6:
        out *= (10 ** (-1.5 / 20.0)) / peak
    head = int(0.03 * SR)
    tail = int(0.4 * SR)
    out[:head] *= np.linspace(0, 1, head)[:, None]
    out[-tail:] *= np.linspace(1, 0, tail)[:, None]

    arr16 = np.clip(out * 32767.0, -32768, 32767).astype("<i2")
    with wave_open(str(OUT), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(arr16.tobytes())
    print(f"Wrote {OUT}  ({out.shape[0]/SR:.2f}s, {OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
