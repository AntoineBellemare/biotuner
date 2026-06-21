"""
Reusable intro-synth voices for the reel pipeline (the 5 audition designs).
Each returns a normalised float stereo array of length ~DUR seconds.

Used by export_reels.py (the chosen voice leads the intro) and by the
audition scripts.
"""
from __future__ import annotations

import math
from pathlib import Path
from wave import open as wave_open

import numpy as np
from scipy.signal import butter, sosfilt

SR = 44_100
DUR = 3.6
ROOT = 130.81  # C3


def _t(dur=DUR):
    return np.arange(int(dur * SR)) / SR


def _stereo(mono, pan=0.0):
    a = (pan + 1) * math.pi / 4
    return np.stack([mono * math.cos(a), mono * math.sin(a)], axis=1)


def _norm_fade(x, head=0.01, tail=0.25):
    peak = float(np.max(np.abs(x)))
    if peak > 1e-9:
        x = x * (10 ** (-1.0 / 20) / peak)
    h = int(head * SR)
    tl = int(tail * SR)
    x[:h] *= np.linspace(0, 1, h)[:, None]
    x[-tl:] *= np.linspace(1, 0, tl)[:, None]
    return x


def _reverb(x, decay=2.2, mix=0.35, seed=0):
    rng = np.random.default_rng(seed)
    n = int(decay * SR)
    ir = (rng.standard_normal(n) * np.exp(-np.arange(n) / (decay * SR / 4)))
    ir /= np.max(np.abs(ir)) + 1e-9
    wet = np.stack(
        [np.convolve(x[:, c], ir)[: x.shape[0]] for c in range(2)], axis=1
    )
    return (1 - mix) * x + mix * wet


# ───────────────────────────────────────────── 1. Singing bowl / bell ──
def singing_bowl(dur: float = DUR) -> np.ndarray:
    t = _t(dur)
    f0 = ROOT * 1.5  # G3 bell fundamental (the chord's fifth)
    partials = [
        (0.5, 0.5, 3.4), (1.0, 1.0, 3.0), (1.19, 0.6, 2.2),
        (1.56, 0.5, 1.6), (2.0, 0.45, 1.4), (2.56, 0.3, 1.0),
        (3.01, 0.25, 0.8), (4.07, 0.18, 0.6),
    ]
    out = np.zeros(len(t))
    for ratio, amp, dec in partials:
        f = f0 * ratio
        beat = 1 + 0.004 * np.sin(2 * np.pi * 1.3 * t)
        out += amp * np.exp(-t / dec) * np.sin(2 * np.pi * f * beat * t)
    return _norm_fade(_reverb(_stereo(out * 0.3), decay=2.6, mix=0.4, seed=1))


# ───────────────────────────────────────────── 2. Cinematic riser ──────
def riser(dur: float = DUR) -> np.ndarray:
    t = _t(dur)
    n = len(t)
    rng = np.random.default_rng(2)
    noise = rng.standard_normal(n)
    out = np.zeros(n)
    block = 1024
    for i in range(0, n, block):
        j = min(i + block, n)
        frac = (i + j) / 2 / n
        fc = 200 * (4000 / 200) ** frac
        lo = max(fc * 0.6, 40)
        hi = min(fc * 1.6, SR * 0.49)
        sos = butter(2, [lo, hi], btype="bandpass", fs=SR, output="sos")
        out[i:j] = sosfilt(sos, noise[i:j])
    env = (t / dur) ** 2
    out *= env
    fsweep = 100 * (8 ** (t / dur))
    out += 0.4 * env * np.sin(2 * np.pi * np.cumsum(fsweep) / SR)
    imp_t = t - (dur - 0.35)
    impact = np.where(imp_t > 0, np.exp(-imp_t / 0.12) * np.sin(2 * np.pi * ROOT * imp_t), 0)
    out += 0.8 * impact
    return _norm_fade(_stereo(out * 0.4), tail=0.1)


# ───────────────────────────────────────────── 3. Glassy harp arp ──────
def harp_arp(dur: float = DUR) -> np.ndarray:
    t = _t(dur)
    out = np.zeros(len(t))
    notes = [ROOT, ROOT * 5 / 4, ROOT * 3 / 2, ROOT * 2, ROOT * 5 / 2, ROOT * 3]
    for k, f in enumerate(notes):
        lt = t - 0.14 * k
        env = np.where(lt > 0, np.exp(-lt / 0.7), 0)
        pluck = np.zeros(len(t))
        for h in range(1, 7):
            pluck += (1 / h ** 1.5) * np.sin(2 * np.pi * f * h * np.maximum(lt, 0))
        out += 0.5 * env * pluck * (lt > 0)
    return _norm_fade(_reverb(_stereo(out * 0.18), decay=2.0, mix=0.4, seed=3))


# ───────────────────────────────────────────── 4. Deep drone + sub ─────
def deep_drone(dur: float = DUR) -> np.ndarray:
    t = _t(dur)
    swell = (1 - np.exp(-t / 1.2)) * np.exp(-np.maximum(t - dur * 0.72, 0) / 0.6)
    out = np.zeros(len(t))
    for f, a in [(ROOT / 2, 0.9), (ROOT, 0.6), (ROOT * 3 / 2, 0.4), (ROOT * 2, 0.25)]:
        vib = 1 + 0.003 * np.sin(2 * np.pi * 4.5 * t)
        for h, ah in [(1, 1.0), (2, 0.4), (3, 0.2)]:
            out += a * ah * np.sin(2 * np.pi * f * h * vib * t)
    out *= swell
    n = len(t)
    res = np.zeros(n)
    block = 2048
    for i in range(0, n, block):
        j = min(i + block, n)
        frac = (i + j) / 2 / n
        fc = 300 * (8 ** frac)
        sos = butter(2, min(fc, SR * 0.49), btype="low", fs=SR, output="sos")
        res[i:j] = sosfilt(sos, out[i:j])
    return _norm_fade(_stereo(res * 0.16))


# ───────────────────────────────── 5. Reverse swell + crystal pluck ────
def reverse_crystal(dur: float = DUR) -> np.ndarray:
    t = _t(dur)
    pad = np.zeros(len(t))
    for f, a in [(ROOT, 1.0), (ROOT * 5 / 4, 0.7), (ROOT * 3 / 2, 0.6)]:
        for h, ah in [(1, 1.0), (2, 0.5), (3, 0.3), (4, 0.18)]:
            pad += a * ah * np.sin(2 * np.pi * f * h * t)
    hit_at = dur - 0.55
    pad *= np.clip(t / hit_at, 0, 1) ** 2.2
    lt = t - hit_at
    bell = np.zeros(len(t))
    for ratio, a, dec in [(1, 1, 0.5), (2.01, 0.6, 0.4), (3.0, 0.4, 0.3),
                          (4.2, 0.25, 0.22), (5.4, 0.15, 0.16)]:
        bell += a * np.where(lt > 0, np.exp(-lt / dec) * np.sin(2 * np.pi * ROOT * 2 * ratio * lt), 0)
    return _norm_fade(_reverb(_stereo(0.12 * pad + 0.5 * bell), decay=1.8, mix=0.35, seed=5), tail=0.15)


SYNTHS = {
    "1_singing_bowl": singing_bowl,
    "2_riser": riser,
    "3_harp_arp": harp_arp,
    "4_deep_drone": deep_drone,
    "5_reverse_crystal": reverse_crystal,
}


def write_wav(stereo: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.clip(stereo * 32767, -32768, 32767).astype("<i2")
    with wave_open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(a.tobytes())
