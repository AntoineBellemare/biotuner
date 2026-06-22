"""
Soundtrack for "Brain Polyrhythm" (BrainPolyrhythm-IG) — Freesound percussion.

A modern ambient kit (CC0 samples under public/audio/samples/, fetched by
fetch_freesound.py) plays the 5:7:11 polyrhythm:
  · voice 5  — deep tom          · voice 7 — kalimba (tuned to D5)
  · voice 11 — egg shaker        · downbeat — tom + kalimba accent
  · a singing bowl swells in at each movement; an ambient pad bed evolves under
    everything (volume + perc intensity ramp across the reel).
Summed through a light convolution reverb. Timeline matches BrainPolyrhythm.tsx.

Needs the sample kit first:  FREESOUND_TOKEN=… python fetch_freesound.py
Run: python render_brainpoly_audio.py  →  public/audio/brain_polyrhythm.wav
"""
from __future__ import annotations

import json
from pathlib import Path
from wave import open as wave_open

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt

HERE = Path(__file__).resolve().parent
SAMPLES = HERE / "public" / "audio" / "samples"
OUT = HERE / "public" / "audio" / "brain_polyrhythm.wav"
SR = 44_100
FPS = 30

INTRO = 18
NECK = 160
CYCLE = 84
NCYC = 6
PLAY = NCYC * CYCLE
TAIL = 44
PLAY_START = INTRO + NECK
N_FRAMES = PLAY_START + PLAY + TAIL

dat = json.load(open(HERE / "public" / "brain_polyrhythm.json", encoding="utf-8"))
VOICES = dat["voices"]


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
    on = int(np.argmax(np.abs(x) > 0.04 * pk))   # trim leading silence
    return x[on:] / pk


def pitch(x: np.ndarray, ratio: float) -> np.ndarray:
    if abs(ratio - 1) < 1e-3:
        return x
    idx = np.arange(0, len(x) - 1, ratio)
    return np.interp(idx, np.arange(len(x)), x)


def reverb(stereo, wet=0.18, decay=0.9, length=1.8):
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
    low = load("low")
    kal = load("kalimba")
    sha = load("shaker")
    bowl = load("bowl")
    bed = load("bed")
    kal_mid = pitch(kal, 0.556)   # → ~D5, warm mid pluck

    n = int(N_FRAMES / FPS * SR) + int(2.0 * SR)
    buf = np.zeros((n, 2))

    def add(frame, sig, gain, pan=0.0):
        s = int(frame / FPS * SR)
        e = min(s + len(sig), n)
        if e <= s:
            return
        buf[s:e, 0] += sig[: e - s] * gain * (1 - max(0, pan))
        buf[s:e, 1] += sig[: e - s] * gain * (1 + min(0, pan))

    # ── ambient pad bed: tiled to cover the play, swelling, slowly opening ──
    play_s, end_s = PLAY_START / FPS, N_FRAMES / FPS
    bed_track = np.zeros(n)
    pos = int((play_s - 1.0) * SR)
    while pos < int(end_s * SR):
        seg = bed
        e = min(pos + len(seg), n)
        bed_track[pos:e] += seg[: e - pos]
        pos += len(seg) - int(0.5 * SR)         # 0.5s overlap
    t = np.arange(n) / SR
    swell = np.clip((t - (play_s - 1.0)) / 2.5, 0, 1) * np.clip((end_s - t) / 1.6, 0, 1)
    swell *= 0.55 + 0.45 * np.clip((t - play_s) / (PLAY / FPS), 0, 1)   # grows over the reel
    buf[:, 0] += 0.34 * bed_track * swell
    buf[:, 1] += 0.34 * bed_track * swell

    pans = (-0.35, 0.0, 0.35)
    for c in range(NCYC):
        inten = 0.45 + 0.55 * (c / (NCYC - 1))
        for vi, v in enumerate(VOICES):
            for o in v["onsets"]:
                fr = PLAY_START + (c + o) * CYCLE
                accent = o == 0
                if vi == 0:
                    add(fr, low, (1.0 if accent else 0.66) * (0.7 + 0.3 * inten), pans[0])
                elif vi == 1:
                    add(fr, kal_mid, (0.85 if accent else 0.5) * (0.6 + 0.4 * inten), pans[1])
                else:
                    add(fr, sha, (0.7 if accent else 0.42) * inten, pans[2])
        # downbeat accent + a singing-bowl swell at each movement (cycles 0 and 3)
        fr0 = PLAY_START + c * CYCLE
        add(fr0, kal, 0.4 * inten, 0.0)
        if c in (0, 3):
            add(fr0 - 6, bowl, 0.5)

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
