"""
Soundtrack for the "Sethares dissonance" reel (SetharesDissonance-IG).

For each timbre, a fixed lower complex tone + an upper copy gliding unison →
octave, both built from that timbre's partials — so you HEAR the roughness curve
(and how the harmonic vs stretched spectra sound different) in sync with the
sweeping marker. Timeline constants must match SetharesDissonance.tsx; the glide
matches the scene's eased sweep (0.5·(1−cos(π·t))).

Run: python render_sethares_audio.py  →  public/audio/sethares.wav
"""
from __future__ import annotations

import json
from pathlib import Path
from wave import open as wave_open

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "public" / "audio" / "sethares.wav"
SR = 44_100
FPS = 30

INTRO = 24
SWEEP = 150
HOLD = 70
BEAT = SWEEP + HOLD
OUTRO = 34

dat = json.load(open(HERE / "public" / "sethares.json", encoding="utf-8"))
F0 = dat["f0"]
TIMBRES = dat["timbres"]
N_FRAMES = INTRO + len(TIMBRES) * BEAT + OUTRO


def render_beat(partials, amps) -> np.ndarray:
    dur = BEAT / FPS + 0.6
    n = int(dur * SR)
    t = np.arange(n) / SR
    frame = t * FPS
    u = np.clip(frame / SWEEP, 0.0, 1.0)
    alpha = 1.0 + 0.5 * (1.0 - np.cos(np.pi * u))   # eased sweep, holds at 2

    lower = np.zeros(n)
    upper = np.zeros(n)
    for p, a in zip(partials, amps):
        lower += a * np.sin(2 * np.pi * p * F0 * t)
        f_inst = p * F0 * alpha
        upper += a * np.sin(2 * np.pi * np.cumsum(f_inst) / SR)

    env = np.ones(n)
    at = int(0.04 * SR)
    rl = int(0.5 * SR)
    env[:at] = np.linspace(0, 1, at)
    env[-rl:] *= np.linspace(1, 0, rl)
    out = np.stack([(0.5 * lower + 0.25 * upper) * env,
                    (0.25 * lower + 0.5 * upper) * env], axis=1)
    return out


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    total_n = int(N_FRAMES / FPS * SR) + int(0.8 * SR)
    master = np.zeros((total_n, 2))

    for ti, T in enumerate(TIMBRES):
        beat = render_beat(T["partials"], T["amps"])
        start = int((INTRO + ti * BEAT) / FPS * SR)
        end = min(start + beat.shape[0], master.shape[0])
        master[start:end] += beat[: end - start]

    master = master[: int(N_FRAMES / FPS * SR)]
    peak = float(np.max(np.abs(master)))
    if peak > 1e-6:
        master *= (10 ** (-2.0 / 20.0)) / peak
    head = int(0.05 * SR)
    tail = int(0.4 * SR)
    master[:head] *= np.linspace(0, 1, head)[:, None]
    master[-tail:] *= np.linspace(1, 0, tail)[:, None]

    arr16 = np.clip(master * 32767.0, -32768, 32767).astype("<i2")
    with wave_open(str(OUT), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(arr16.tobytes())
    print(f"Wrote {OUT}  ({master.shape[0]/SR:.2f}s)")


if __name__ == "__main__":
    main()
