"""
Soundtrack for "Fractal Rhythm" (FractalRhythm-IG) — Freesound kit.

The order-1 coincidences (the meta-rhythm) are the main groove on a deep tom; a
soft shaker shimmers the fast voice; when the order-2 wheel appears it pings a
kalimba; an ambient pad bed throughout. Timeline matches FractalRhythm.tsx.

Run: python render_fractal_audio.py  →  public/audio/fractal.wav
"""
from __future__ import annotations

import json
from pathlib import Path
from wave import open as wave_open

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

HERE = Path(__file__).resolve().parent
SAMPLES = HERE / "public" / "audio" / "samples"
OUT = HERE / "public" / "audio" / "fractal.wav"
SR = 44_100
FPS = 30

TITLE = 84
BUILD = 70
CYCLE = 96
NCYC = 5
TAIL = 44
PLAY_START = BUILD
N_FRAMES = PLAY_START + NCYC * CYCLE + TAIL

D = json.load(open(HERE / "public" / "fractal.json", encoding="utf-8"))["orders"]
O1, O2 = D[0], D[1]


def load(name):
    x, sr = sf.read(SAMPLES / f"{name}.wav")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        x = resample_poly(x, SR, sr)
    x = x.astype(float)
    pk = np.max(np.abs(x)) or 1.0
    return x[int(np.argmax(np.abs(x) > 0.04 * pk)):] / pk


def pitch(x, ratio):
    idx = np.arange(0, len(x) - 1, ratio)
    return np.interp(idx, np.arange(len(x)), x)


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
    low, kal, sha, bowl, bed = (load(x) for x in ("low", "kalimba", "shaker", "bowl", "bed"))
    n = int(N_FRAMES / FPS * SR) + int(2.0 * SR)
    buf = np.zeros((n, 2))

    def add(frame, sig, gain, pan=0.0):
        s = int(frame / FPS * SR)
        e = min(s + len(sig), n)
        if e <= max(0, s):
            return
        s = max(0, s)
        buf[s:e, 0] += sig[: e - s] * gain * (1 - max(0, pan))
        buf[s:e, 1] += sig[: e - s] * gain * (1 + min(0, pan))

    # bed
    bed_track = np.zeros(n)
    pos = int(40 / FPS * SR)
    while pos < n:
        e = min(pos + len(bed), n)
        bed_track[pos:e] += bed[: e - pos]
        pos += len(bed) - int(0.5 * SR)
    t = np.arange(n) / SR
    swell = np.clip((t - 1.0) / 2.0, 0, 1) * np.clip((N_FRAMES / FPS - t) / 1.6, 0, 1)
    buf[:, 0] += 0.26 * bed_track * swell
    buf[:, 1] += 0.26 * bed_track * swell

    fast = max(O1["voices"], key=lambda v: v["pulses"])  # shimmer the densest voice
    for c in range(NCYC):
        # order-1 coincidences = the meta groove
        for co in O1["coincidences"]:
            fr = PLAY_START + (c + co["t"]) * CYCLE
            add(fr, low, 0.5 + 0.12 * co["n"])
        # soft shaker shimmer on the fast voice
        for o in fast["onsets"]:
            add(PLAY_START + (c + o) * CYCLE, sha, 0.12, 0.3)
        # order-2 enters at cycle 2
        if c >= 2:
            if c == 2:
                add(PLAY_START + c * CYCLE, bowl, 0.45)
            for vi, v in enumerate(O2["voices"]):
                for o in v["onsets"]:
                    add(PLAY_START + (c + o) * CYCLE, pitch(kal, 0.5 + 0.08 * vi), 0.3, -0.2)

    buf = reverb(buf, wet=0.2)
    buf = buf[: int(N_FRAMES / FPS * SR)]
    peak = float(np.max(np.abs(buf)))
    if peak > 1e-6:
        buf *= (10 ** (-1.8 / 20.0)) / peak
    head, tail = int(0.04 * SR), int(0.6 * SR)
    buf[:head] *= np.linspace(0, 1, head)[:, None]
    buf[-tail:] *= np.linspace(1, 0, tail)[:, None]

    arr16 = np.clip(buf * 32767.0, -32768, 32767).astype("<i2")
    with wave_open(str(OUT), "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes(arr16.tobytes())
    print(f"Wrote {OUT}  ({buf.shape[0]/SR:.2f}s)")


if __name__ == "__main__":
    main()
