"""
Soundtrack for "Heart × Brain" (HeartBrainDuet-IG).

The ECG heartbeat (real R-peaks, lub-dub kick) is the master pulse; the brain's
5:7:11 polyrhythm plays pitched mallets on top; when a brain pulse lands on a
heartbeat the mallet is accented. Timeline matches HeartBrainDuet.tsx (audio is
offset by TITLE via Sequence; events sound when they cross the playhead).

Run: python render_heartbrain_audio.py  →  public/audio/heart_brain.wav
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from wave import open as wave_open

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "public" / "audio" / "heart_brain.wav"
SR = 44_100
FPS = 30

INTRO = 18
LEADIN = 1.2
CYCLE = 2.0
COINC_TOL = 0.07
PITCH = [220.0, 330.0, 440.0]  # A3 · E4 · A4

dat = json.load(open(HERE / "public" / "heart_brain.json", encoding="utf-8"))
R = dat["r_times"]
VOICES = dat["brain_voices"]
DUR = dat["dur_s"]
PLAYS = math.ceil((DUR + 1.0) * FPS) + INTRO
N_FRAMES = PLAYS + 36


def mallet(freq, amp, dur=0.5, bright=2.2, decay=9.0):
    n = int(dur * SR); t = np.arange(n) / SR
    idx = bright * np.exp(-t * 6)
    sig = np.sin(2 * np.pi * freq * t + idx * np.sin(2 * np.pi * freq * 2 * t))
    env = np.exp(-t * decay) * (1 - np.exp(-t * 400))
    return amp * sig * env


def thump(amp, dur=0.32, f0=92, k=24):
    n = int(dur * SR); t = np.arange(n) / SR
    f = f0 * np.exp(-t * k) + 40
    return amp * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9)


def add(buf, t_sec, sig):
    s = int((INTRO / FPS + LEADIN + t_sec) * SR)  # event sounds when it hits the playhead
    if s < 0:
        return
    e = min(s + len(sig), buf.shape[0])
    if e <= 0:
        return
    buf[s:e, 0] += sig[: e - s]
    buf[s:e, 1] += sig[: e - s]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = int(N_FRAMES / FPS * SR) + int(0.6 * SR)
    buf = np.zeros((n, 2))

    # heart: lub-dub per R-peak
    for r in R:
        add(buf, r, thump(0.75))
        add(buf, r + 0.13, thump(0.4, f0=78))

    # brain: metronomic 5:7:11; accent the coincidences
    for vi, v in enumerate(VOICES):
        nC = int(DUR / CYCLE) + 2
        for c in range(nC):
            for o in v["onsets"]:
                t = (c + o) * CYCLE
                if t > DUR + 0.5:
                    continue
                coincide = any(abs(r - t) < COINC_TOL for r in R)
                add(buf, t, mallet(PITCH[vi % len(PITCH)], 0.5 if coincide else 0.26))

    buf = buf[: int(N_FRAMES / FPS * SR)]
    peak = float(np.max(np.abs(buf)))
    if peak > 1e-6:
        buf *= (10 ** (-2.5 / 20.0)) / peak
    head = int(0.04 * SR); tail = int(0.5 * SR)
    buf[:head] *= np.linspace(0, 1, head)[:, None]
    buf[-tail:] *= np.linspace(1, 0, tail)[:, None]

    arr16 = np.clip(buf * 32767.0, -32768, 32767).astype("<i2")
    with wave_open(str(OUT), "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes(arr16.tobytes())
    print(f"Wrote {OUT}  ({buf.shape[0]/SR:.2f}s)")


if __name__ == "__main__":
    main()
