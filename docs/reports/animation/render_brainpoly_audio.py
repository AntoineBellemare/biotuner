"""
Soundtrack for "Brain Polyrhythm" (BrainPolyrhythm-IG).

Modern, evolving, ambient percussion for the 5:7:11 polyrhythm:
  · voice 5  — deep membrane tom (pitch-drop body + sub + soft transient)
  · voice 7  — resonant wood/rim (band-passed noise + metallic click)
  · voice 11 — glassy shaker (high band-passed noise + shimmer), through a dub delay
  · downbeat — layered boom + bright transient
  · bed      — a slowly-opening pad drone + a riser into the ripple movement
All summed through a convolution reverb. Brightness opens over the reel so the
texture evolves. Timeline matches BrainPolyrhythm.tsx (offset by TITLE).

Run: python render_brainpoly_audio.py  →  public/audio/brain_polyrhythm.wav
"""
from __future__ import annotations

import json
from pathlib import Path
from wave import open as wave_open

import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve

HERE = Path(__file__).resolve().parent
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
CYCLE_S = CYCLE / FPS

dat = json.load(open(HERE / "public" / "brain_polyrhythm.json", encoding="utf-8"))
VOICES = dat["voices"]
RNG = np.random.RandomState(7)


def env(n, attack, decay):
    t = np.arange(n) / SR
    return np.exp(-t * decay) * (1 - np.exp(-t * attack))


def bp_noise(fc, dur, q=8.0, decay=18.0, seed=0):
    n = int(dur * SR)
    x = np.random.RandomState(seed).randn(n)
    lo = max(20.0, fc / (2 ** (1 / (2 * q))))
    hi = min(SR / 2 - 200, fc * (2 ** (1 / (2 * q))))
    sos = butter(2, [lo, hi], btype="band", fs=SR, output="sos")
    return sosfilt(sos, x) * env(n, 700, decay)


def membrane(f0, dur, drop=0.7, decay=9.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = f0 * (1 + drop * np.exp(-t * 28))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 500, decay)


def metallic(f, dur, decay=14.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    sig = sum(np.sin(2 * np.pi * f * r * t) / (1 + r) for r in (1, 2.76, 5.4, 8.93))
    return sig * np.exp(-t * decay)


def hi_click(dur=0.03):
    n = int(dur * SR)
    sos = butter(2, 3500, btype="high", fs=SR, output="sos")
    return sosfilt(sos, RNG.randn(n)) * np.exp(-np.arange(n) / SR * 130)


def fit(x, n):
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))


def voice_hit(vi, amp, bright, seed):
    if vi == 0:  # deep membrane tom + sub
        n = int(0.55 * SR)
        sub = np.sin(2 * np.pi * 47 * np.arange(n) / SR) * env(n, 300, 7)
        h = (0.95 * fit(membrane(74, 0.55, 0.75, 8), n) + 0.5 * sub
             + (0.18 + 0.22 * bright) * fit(hi_click(), n))
        return amp * h
    if vi == 1:  # resonant wood / rim
        n = int(0.3 * SR)
        h = (0.85 * fit(bp_noise(540, 0.3, q=9, decay=24, seed=seed), n)
             + 0.4 * (0.6 + 0.5 * bright) * fit(metallic(900, 0.18), n))
        return amp * h
    # vi == 2  glassy shaker
    n = int(0.16 * SR)
    h = (0.6 * (0.5 + 0.6 * bright) * fit(bp_noise(6800, 0.16, q=2.2, decay=40, seed=seed), n)
         + 0.3 * fit(metallic(5200, 0.1), n))
    return amp * h


def reverb(stereo, wet=0.3, decay=0.85, length=1.9):
    rng = np.random.RandomState(42)
    n_ir = int(length * SR)
    t = np.arange(n_ir) / SR
    out = stereo.copy()
    for c in (0, 1):
        ir = np.exp(-t / decay) * rng.randn(n_ir)
        ir /= np.sqrt(np.sum(ir ** 2)) or 1.0
        w = fftconvolve(stereo[:, c], ir)[: stereo.shape[0]]
        out[:, c] = (1 - wet) * stereo[:, c] + wet * w
    return out


def add(buf, frame, sig, pan=0.0):
    s = int(frame / FPS * SR)
    e = min(s + len(sig), buf.shape[0])
    if e <= s:
        return
    lg, rg = (1 - max(0, pan)), (1 + min(0, pan))
    buf[s:e, 0] += sig[: e - s] * lg
    buf[s:e, 1] += sig[: e - s] * rg


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = int(N_FRAMES / FPS * SR) + int(1.0 * SR)
    buf = np.zeros((n, 2))
    hi_bus = np.zeros((n, 2))   # high voice → its own bus (for dub delay)
    t = np.arange(n) / SR

    # ── evolving ambient pad: root drone whose partials open over the reel ──
    play_t0 = PLAY_START / FPS
    prog = np.clip((t - play_t0) / (PLAY / FPS), 0, 1)
    pad = np.zeros(n)
    for i, (pf, base) in enumerate([(73.42, 0.5), (110.0, 0.34), (146.83, 0.26),
                                    (220.0, 0.16), (293.66, 0.12)]):
        grow = np.clip(prog * 5 - i * 0.9, 0, 1)           # upper partials fade in over time
        pad += base * grow * np.sin(2 * np.pi * pf * t + i)
    penv = np.clip((t - (play_t0 - 1.0)) / 1.2, 0, 1) * np.clip((N_FRAMES / FPS - t) / 1.4, 0, 1)
    buf[:, 0] += 0.1 * pad * penv
    buf[:, 1] += 0.1 * pad * penv

    # ── riser into the ripple movement (cycle ~3) ──
    riser_c = 3
    rs = int((PLAY_START + (riser_c - 1.0) * CYCLE) / FPS * SR)
    rl = int(1.0 * CYCLE / FPS * SR)
    rn = np.random.RandomState(3).randn(rl)
    sos = butter(2, 2500, btype="low", fs=SR, output="sos")
    riser = sosfilt(sos, rn) * np.linspace(0, 1, rl) ** 2 * 0.25
    if rs + rl <= n:
        buf[rs:rs + rl, 0] += riser
        buf[rs:rs + rl, 1] += riser

    seed = 1
    for c in range(NCYC):
        bright = 0.35 + 0.65 * (c / (NCYC - 1))
        for vi, v in enumerate(VOICES):
            pan = (-0.4, 0.0, 0.4)[vi]
            for o in v["onsets"]:
                fr = PLAY_START + (c + o) * CYCLE
                amp = (0.95 if o == 0 else 0.6) * (0.7 + 0.3 * bright)
                hit = voice_hit(vi, amp, bright, seed)
                seed += 1
                add(hi_bus if vi == 2 else buf, fr, hit, pan)
        # downbeat accent: boom + bright transient
        fr0 = PLAY_START + c * CYCLE
        add(buf, fr0, 1.1 * membrane(58, 0.6, 0.8, 6))
        add(buf, fr0, 0.5 * np.pad(hi_click(0.05), (0, 0)))

    # dub delay on the high bus (tempo-synced: one 11-step)
    step = (CYCLE_S / 11)
    delayed = np.zeros_like(hi_bus)
    for k, g in enumerate([0.55, 0.32, 0.18], start=1):
        d = int(k * step * 3 * SR)
        if d < n:
            delayed[d:] += hi_bus[:n - d] * g
    buf += hi_bus + delayed

    buf = reverb(buf, wet=0.32)
    buf = buf[: int(N_FRAMES / FPS * SR)]
    peak = float(np.max(np.abs(buf)))
    if peak > 1e-6:
        buf *= (10 ** (-2.0 / 20.0)) / peak
    head = int(0.04 * SR); tail = int(0.6 * SR)
    buf[:head] *= np.linspace(0, 1, head)[:, None]
    buf[-tail:] *= np.linspace(1, 0, tail)[:, None]

    arr16 = np.clip(buf * 32767.0, -32768, 32767).astype("<i2")
    with wave_open(str(OUT), "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes(arr16.tobytes())
    print(f"Wrote {OUT}  ({buf.shape[0]/SR:.2f}s)")


if __name__ == "__main__":
    main()
