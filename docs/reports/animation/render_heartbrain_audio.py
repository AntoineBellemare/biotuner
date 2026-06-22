"""
Soundtrack for "Heart × Brain" (HeartBrainDuet-IG) — Freesound kit.

The ECG heartbeat is a real heartbeat sample on every R-peak; the brain's 5:7:11
plays a melodic kalimba/shaker polyrhythm on top; where a brain pulse lands on a
heartbeat it's accented and a singing bowl rings. An ambient pad bed underneath.
Timeline matches HeartBrainDuet.tsx (offset by TITLE via Sequence; events sound
when they cross the playhead).

Needs the sample kit:  FREESOUND_TOKEN=… python fetch_freesound.py
Run: python render_heartbrain_audio.py  →  public/audio/heart_brain.wav
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from wave import open as wave_open

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

HERE = Path(__file__).resolve().parent
SAMPLES = HERE / "public" / "audio" / "samples"
OUT = HERE / "public" / "audio" / "heart_brain.wav"
SR = 44_100
FPS = 30

INTRO = 18
LEADIN = 1.2
CYCLE = 2.0
COINC_TOL = 0.07

dat = json.load(open(HERE / "public" / "heart_brain.json", encoding="utf-8"))
R = dat["r_times"]
VOICES = dat["brain_voices"]
DUR = dat["dur_s"]
PLAYS = math.ceil((DUR + 1.0) * FPS) + INTRO
N_FRAMES = PLAYS + 36


def load(name: str) -> np.ndarray:
    p = SAMPLES / f"{name}.wav"
    if not p.exists():
        raise SystemExit(f"Missing {p} — run `FREESOUND_TOKEN=… python fetch_freesound.py` first.")
    x, sr = sf.read(p)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        x = resample_poly(x, SR, sr)
    x = x.astype(float)
    pk = np.max(np.abs(x)) or 1.0
    on = int(np.argmax(np.abs(x) > 0.04 * pk))
    return x[on:] / pk


def pitch(x, ratio):
    if abs(ratio - 1) < 1e-3:
        return x
    idx = np.arange(0, len(x) - 1, ratio)
    return np.interp(idx, np.arange(len(x)), x)


def reverb(stereo, wet=0.2, decay=0.9, length=1.8):
    rng = np.random.RandomState(42)
    n_ir = int(length * SR)
    t = np.arange(n_ir) / SR
    out = stereo.copy()
    for c in (0, 1):
        ir = np.exp(-t / decay) * rng.randn(n_ir)
        ir /= np.sqrt(np.sum(ir ** 2)) or 1.0
        out[:, c] = (1 - wet) * stereo[:, c] + wet * fftconvolve(stereo[:, c], ir)[: stereo.shape[0]]
    return out


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    heart = load("heartbeat")
    kal = load("kalimba")
    sha = load("shaker")
    bowl = load("bowl")
    bed = load("bed")
    kal_lo = pitch(kal, 0.417)   # ~A4
    kal_mid = pitch(kal, 0.556)  # ~D5

    n = int(N_FRAMES / FPS * SR) + int(2.0 * SR)
    buf = np.zeros((n, 2))

    def add(t_sec, sig, gain, pan=0.0):
        s = int((INTRO / FPS + LEADIN + t_sec) * SR)
        e = min(s + len(sig), n)
        if e <= max(0, s):
            return
        s = max(0, s)
        buf[s:e, 0] += sig[: e - s] * gain * (1 - max(0, pan))
        buf[s:e, 1] += sig[: e - s] * gain * (1 + min(0, pan))

    # ambient pad bed under the whole timeline
    bed_track = np.zeros(n)
    pos = 0
    while pos < n:
        e = min(pos + len(bed), n)
        bed_track[pos:e] += bed[: e - pos]
        pos += len(bed) - int(0.5 * SR)
    t = np.arange(n) / SR
    swell = np.clip(t / 1.5, 0, 1) * np.clip((N_FRAMES / FPS - t) / 1.6, 0, 1)
    buf[:, 0] += 0.26 * bed_track * swell
    buf[:, 1] += 0.26 * bed_track * swell

    # heart: a real heartbeat on every R-peak
    for r in R:
        add(r, heart, 0.95)

    # brain: melodic 5:7:11; accent + bowl ring on coincidences
    samp = [kal_lo, kal_mid, sha]
    pans = [-0.3, 0.0, 0.3]
    for vi, v in enumerate(VOICES):
        nC = int(DUR / CYCLE) + 2
        for c in range(nC):
            for o in v["onsets"]:
                tt = (c + o) * CYCLE
                if tt > DUR + 0.5:
                    continue
                coincide = any(abs(r - tt) < COINC_TOL for r in R)
                g = (0.62 if coincide else 0.4) * (0.7 if vi == 2 else 1.0)
                add(tt, samp[vi], g, pans[vi])
                if coincide and vi == 0:
                    add(tt, bowl, 0.34)

    buf = reverb(buf, wet=0.2)
    buf = buf[: int(N_FRAMES / FPS * SR)]
    peak = float(np.max(np.abs(buf)))
    if peak > 1e-6:
        buf *= (10 ** (-2.0 / 20.0)) / peak
    head, tail = int(0.04 * SR), int(0.5 * SR)
    buf[:head] *= np.linspace(0, 1, head)[:, None]
    buf[-tail:] *= np.linspace(1, 0, tail)[:, None]

    arr16 = np.clip(buf * 32767.0, -32768, 32767).astype("<i2")
    with wave_open(str(OUT), "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes(arr16.tobytes())
    print(f"Wrote {OUT}  ({buf.shape[0]/SR:.2f}s)")


if __name__ == "__main__":
    main()
