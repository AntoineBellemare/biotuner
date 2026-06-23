"""
Soundtrack for "Brain Grooves" (BrainGrooves-IG) — Freesound world-percussion.

The EEG-derived named rhythms layer one by one onto a shared bar, each on its own
hand-drum, into a world polygroove:
  Tresillo→clave · Nawakhat→rebana · Ruchenitza→bongo · Savari→tabla · Zappa→woodblock
plus a soft tom anchor, a bowl swell at the drop, and an ambient pad bed.
Timeline matches BrainGrooves.tsx.

Run: python render_braingrooves_audio.py  →  public/audio/brain_grooves.wav
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
OUT = HERE / "public" / "audio" / "brain_grooves.wav"
SR = 44_100
FPS = 30

TITLE = 84
DIDACTIC = 130
ADD = 48
BAR = 80
HOLD = 156
OUTRO = 44
PEAK_FRAMES = [56, 72, 88, 104]
INSTR = ["clave", "rebana", "bongo", "tabla", "woodblock"]
PANS = [-0.4, -0.2, 0.0, 0.2, 0.4]

RH = json.load(open(HERE / "public" / "brain_grooves.json", encoding="utf-8"))["rhythms"]
NR = len(RH)
MIX_START = DIDACTIC
MIX_LEN = (NR - 1) * ADD + HOLD
N_FRAMES = DIDACTIC + MIX_LEN + OUTRO


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


def reverb(stereo, wet=0.16, decay=0.85, length=1.6):
    rng = np.random.RandomState(42)
    t = np.arange(int(length * SR)) / SR
    out = stereo.copy()
    for c in (0, 1):
        ir = np.exp(-t / decay) * rng.randn(len(t))
        ir /= np.sqrt(np.sum(ir ** 2)) or 1.0
        out[:, c] = (1 - wet) * stereo[:, c] + wet * fftconvolve(stereo[:, c], ir)[: stereo.shape[0]]
    return out


def main() -> None:
    kit = {k: load(k) for k in INSTR}
    low, kal, bowl, bed = (load(x) for x in ("low", "kalimba", "bowl", "bed"))
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
    pos = int(50 / FPS * SR)
    while pos < n:
        e = min(pos + len(bed), n)
        bed_track[pos:e] += bed[: e - pos]
        pos += len(bed) - int(0.5 * SR)
    t = np.arange(n) / SR
    swell = np.clip((t - 1.6) / 2.2, 0, 1) * np.clip((N_FRAMES / FPS - t) / 1.6, 0, 1)
    buf[:, 0] += 0.26 * bed_track * swell
    buf[:, 1] += 0.26 * bed_track * swell

    # didactic pings + a bowl at the drop
    for i, fr in enumerate(PEAK_FRAMES):
        add(fr, pitch(kal, 0.62 + 0.14 * i), 0.3, -0.3 + 0.2 * i)
    add(MIX_START - 4, bowl, 0.45)

    nbars = math.ceil(MIX_LEN / BAR) + 1
    for i, R in enumerate(RH):
        pat, nstep, smp = R["pattern"], R["steps"], kit[INSTR[i]]
        active = MIX_START + i * ADD
        for b in range(nbars):
            for j in range(nstep):
                if not pat[j]:
                    continue
                fr = MIX_START + b * BAR + (j / nstep) * BAR
                if fr < active - 1 or fr > N_FRAMES:
                    continue
                accent = j == 0
                add(fr, smp, (0.9 if accent else 0.6), PANS[i])
    # soft tom anchor on each bar downbeat once a couple voices are in
    for b in range(nbars):
        fr = MIX_START + b * BAR
        if fr >= MIX_START + ADD and fr <= N_FRAMES:
            add(fr, low, 0.4)

    buf = reverb(buf, wet=0.16)
    buf = buf[: int(N_FRAMES / FPS * SR)]
    peak = float(np.max(np.abs(buf)))
    if peak > 1e-6:
        buf *= (10 ** (-1.6 / 20.0)) / peak
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
