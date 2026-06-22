"""
Soundtrack for "Brain Polyrhythm" (BrainPolyrhythm-IG).

The 5:7:11 polyrhythm, voiced: each ring is a pitched mallet that strikes on
its evenly-spaced onsets; the downbeat (all voices at once) gets a kick accent.
Timeline matches BrainPolyrhythm.tsx (audio is offset by TITLE via Sequence).

Run: python render_brainpoly_audio.py  →  public/audio/brain_polyrhythm.wav
"""
from __future__ import annotations

import json
from pathlib import Path
from wave import open as wave_open

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "public" / "audio" / "brain_polyrhythm.wav"
SR = 44_100
FPS = 30

INTRO = 18
BUILD = 170
CYCLE = 80
NCYC = 6
PLAY = NCYC * CYCLE
TAIL = 40
PLAY_START = INTRO + BUILD
N_FRAMES = INTRO + BUILD + PLAY + TAIL

dat = json.load(open(HERE / "public" / "brain_polyrhythm.json", encoding="utf-8"))
VOICES = dat["voices"]
PITCH = [196.0, 293.66, 440.0, 587.33]  # G3 · D4 · A4 · D5 — open stack


def mallet(freq, dur, amp, bright=2.4, decay=9.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    idx = bright * np.exp(-t * 6)
    sig = np.sin(2 * np.pi * freq * t + idx * np.sin(2 * np.pi * freq * 2 * t))
    env = np.exp(-t * decay) * (1 - np.exp(-t * 400))
    return amp * sig * env


def kick(dur, amp):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 110 * np.exp(-t * 22) + 42
    return amp * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7)


def add(buf, frame, sig):
    s = int(frame / FPS * SR)
    e = min(s + len(sig), buf.shape[0])
    buf[s:e, 0] += sig[: e - s]
    buf[s:e, 1] += sig[: e - s]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = int(N_FRAMES / FPS * SR) + int(0.6 * SR)
    buf = np.zeros((n, 2))

    # soft root drone for warmth during play
    t = np.arange(n) / SR
    drone = 0.06 * (np.sin(2 * np.pi * 98 * t) + 0.6 * np.sin(2 * np.pi * 147 * t))
    denv = np.clip((t - PLAY_START / FPS) / 0.5, 0, 1) * np.clip((N_FRAMES / FPS - t) / 1.0, 0, 1)
    buf[:, 0] += drone * denv
    buf[:, 1] += drone * denv

    for c in range(NCYC):
        for vi, v in enumerate(VOICES):
            for onset in v["onsets"]:
                fr = PLAY_START + (c + onset) * CYCLE
                amp = 0.5 if onset == 0 else 0.34
                add(buf, fr, mallet(PITCH[vi % len(PITCH)], 0.5, amp))
        # downbeat kick
        add(buf, PLAY_START + c * CYCLE, kick(0.5, 0.7))

    buf = buf[: int(N_FRAMES / FPS * SR)]
    peak = float(np.max(np.abs(buf)))
    if peak > 1e-6:
        buf *= (10 ** (-2.5 / 20.0)) / peak
    head = int(0.04 * SR)
    tail = int(0.5 * SR)
    buf[:head] *= np.linspace(0, 1, head)[:, None]
    buf[-tail:] *= np.linspace(1, 0, tail)[:, None]

    arr16 = np.clip(buf * 32767.0, -32768, 32767).astype("<i2")
    with wave_open(str(OUT), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(arr16.tobytes())
    print(f"Wrote {OUT}  ({buf.shape[0]/SR:.2f}s)")


if __name__ == "__main__":
    main()
