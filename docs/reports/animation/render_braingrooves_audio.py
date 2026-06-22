"""
Soundtrack for "Brain Grooves" (BrainGrooves-IG) — Freesound kit.

Each EEG-derived Euclidean rhythm is played as it appears in the gallery: a deep
tom on every pulse (downbeat accented), a kalimba ping + singing-bowl swell at
each new rhythm, an ambient pad bed throughout. Timeline matches BrainGrooves.tsx.

Needs the kit:  FREESOUND_TOKEN=… python fetch_freesound.py
Run: python render_braingrooves_audio.py  →  public/audio/brain_grooves.wav
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
OUT = HERE / "public" / "audio" / "brain_grooves.wav"
SR = 44_100
FPS = 30

TITLE = 84
DIDACTIC = 168
RHYTHM = 156
TAIL = 46
STEP_FRAMES = 8
PEAK_FRAMES = [70, 86, 102, 118]

RH = json.load(open(HERE / "public" / "brain_grooves.json", encoding="utf-8"))["rhythms"]
NR = len(RH)
N_FRAMES = DIDACTIC + NR * RHYTHM + TAIL


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


def reverb(stereo, wet=0.18, decay=0.9, length=1.8):
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

    # ambient pad bed
    bed_track = np.zeros(n)
    pos = int(60 / FPS * SR)
    while pos < n:
        e = min(pos + len(bed), n)
        bed_track[pos:e] += bed[: e - pos]
        pos += len(bed) - int(0.5 * SR)
    t = np.arange(n) / SR
    swell = np.clip((t - 2.0) / 2.5, 0, 1) * np.clip((N_FRAMES / FPS - t) / 1.6, 0, 1)
    buf[:, 0] += 0.28 * bed_track * swell
    buf[:, 1] += 0.28 * bed_track * swell

    # didactic pings
    for i, fr in enumerate(PEAK_FRAMES):
        add(fr, pitch(kal, 0.62 + 0.14 * i), 0.3, -0.3 + 0.2 * i)

    # gallery: play each Euclidean pattern
    for ri, R in enumerate(RH):
        pat, nstep = R["pattern"], R["steps"]
        base = DIDACTIC + ri * RHYTHM
        add(base + 2, bowl, 0.42)
        add(base + 2, pitch(kal, 0.5), 0.32)
        n_steps = (RHYTHM - 24) // STEP_FRAMES + 1
        for s in range(n_steps):
            if not pat[s % nstep]:
                continue
            fr = base + 24 + s * STEP_FRAMES
            accent = (s % nstep) == 0
            add(fr, low, 0.95 if accent else 0.6)
            add(fr, sha, 0.28 if accent else 0.16, 0.25)

    buf = reverb(buf, wet=0.18)
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
