"""
Soundtrack for "Pitch = Rhythm" (PitchRhythm-IG).

Two sine tones a gap Δf apart; Δf sweeps 1.5 → 38 Hz (same curve as the scene).
You hear a slow tremolo (rhythm) accelerate, then fuse into a steady interval
(pitch) as the beat rate crosses ~20 Hz. A touch of pad + reverb for space.

Run: python render_pitchrhythm_audio.py  →  public/audio/pitch_rhythm.wav
"""
from __future__ import annotations

from pathlib import Path
from wave import open as wave_open

import numpy as np
from scipy.signal import fftconvolve

HERE = Path(__file__).resolve().parent
OUT = HERE / "public" / "audio" / "pitch_rhythm.wav"
SR = 44_100
FPS = 30

INTRO = 18
PLAY = 470
TAIL = 44
N_FRAMES = INTRO + PLAY + TAIL
F0 = 196.0
D_LO, D_HI = 1.5, 38.0


def delta_at(u):
    c = np.clip((u - 0.12) / 0.76, 0, 1)
    e = c * c * (3 - 2 * c)
    return D_LO * (D_HI / D_LO) ** e


def reverb(stereo, wet=0.2, decay=1.0, length=1.8):
    rng = np.random.RandomState(42)
    t = np.arange(int(length * SR)) / SR
    out = stereo.copy()
    for c in (0, 1):
        ir = np.exp(-t / decay) * rng.randn(len(t))
        ir /= np.sqrt(np.sum(ir ** 2)) or 1.0
        out[:, c] = (1 - wet) * stereo[:, c] + wet * fftconvolve(stereo[:, c], ir)[: stereo.shape[0]]
    return out


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = int(N_FRAMES / FPS * SR)
    t = np.arange(n) / SR
    frame = t * FPS
    u = np.clip((frame - INTRO) / PLAY, 0, 1)
    D = delta_at(u)
    f1 = F0 + D

    tone1 = np.sin(2 * np.pi * F0 * t)
    tone2 = np.sin(2 * np.pi * np.cumsum(f1) / SR)
    sig = 0.5 * (tone1 + tone2)

    # soft sub pad rooted at F0 for body
    pad = 0.12 * (np.sin(2 * np.pi * (F0 / 2) * t) + 0.5 * np.sin(2 * np.pi * F0 * t))

    env = np.clip((frame - INTRO) / 20, 0, 1) * np.clip((N_FRAMES - frame) / 30, 0, 1)
    mono = (sig + pad) * env
    stereo = np.stack([mono, mono], axis=1)
    stereo = reverb(stereo, wet=0.18)

    peak = float(np.max(np.abs(stereo)))
    if peak > 1e-6:
        stereo *= (10 ** (-3.0 / 20.0)) / peak
    head, tail = int(0.04 * SR), int(0.5 * SR)
    stereo[:head] *= np.linspace(0, 1, head)[:, None]
    stereo[-tail:] *= np.linspace(1, 0, tail)[:, None]

    arr16 = np.clip(stereo * 32767.0, -32768, 32767).astype("<i2")
    with wave_open(str(OUT), "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes(arr16.tobytes())
    print(f"Wrote {OUT}  ({stereo.shape[0]/SR:.2f}s)")


if __name__ == "__main__":
    main()
