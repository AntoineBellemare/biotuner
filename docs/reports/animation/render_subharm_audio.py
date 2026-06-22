"""
Soundtrack for the "Subharmonic Tension" reel (SubharmonicTension-IG).

Each chord (ratios × base_freq) is played with the modern FM voice as it
appears, so you HEAR the resolved → tense gradient. Timeline constants must
match src/scenes/SubharmonicTension.tsx.

Run: python render_subharm_audio.py  →  public/audio/subharmonicity.wav
"""
from __future__ import annotations

import json
from pathlib import Path
from wave import open as wave_open

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "public" / "audio" / "subharmonicity.wav"
SR = 44_100
FPS = 30

INTRO = 30
DRAW = 50
DWELL = 92
BEAT = DRAW + DWELL
TAIL = 46

dat = json.load(open(HERE / "public" / "subharmonicity.json", encoding="utf-8"))
BASE = dat["base_freq"]
CHORDS = dat["chords"]
N_FRAMES = INTRO + len(CHORDS) * BEAT + TAIL


def fm_voice(freq: float, dur: float, *, detune: float = 0.004,
             bright: float = 3.0, decay: float = 2.6) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    left = np.zeros(n)
    right = np.zeros(n)
    for acc, df in ((0, 1.0 - detune), (1, 1.0 + detune)):
        fc = freq * df
        idx = bright * np.exp(-t * decay) + 0.4
        sig = np.sin(2 * np.pi * fc * t + idx * np.sin(2 * np.pi * fc * 2 * t))
        env = np.ones(n)
        a = max(int(0.008 * SR), 1)
        r = max(int(1.0 * SR), 1)
        env[:a] = np.linspace(0, 1, a)
        env *= 0.55 + 0.45 * np.exp(-t * 0.7)
        if r < n:
            env[-r:] *= np.linspace(1, 0, r)
        (left if acc == 0 else right)[:] += sig * env
    return np.stack([left, right], axis=1)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    total_n = int(N_FRAMES / FPS * SR) + int(1.0 * SR)
    out = np.zeros((total_n, 2))

    for ci, ch in enumerate(CHORDS):
        start = int((INTRO + ci * BEAT + 6) / FPS * SR)
        dur = (DWELL / FPS) + 1.4
        freqs = [r * BASE for r in ch["ratios"]]
        chord = np.zeros((int(dur * SR), 2))
        for f in freqs:
            v = fm_voice(f, dur)
            chord[: v.shape[0]] += v / len(freqs)
        end = min(start + chord.shape[0], out.shape[0])
        out[start:end] += chord[: end - start] * 0.6

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
    print(f"Wrote {OUT}  ({out.shape[0]/SR:.2f}s)")


if __name__ == "__main__":
    main()
