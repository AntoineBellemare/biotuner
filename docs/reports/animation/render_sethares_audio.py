"""
Soundtrack for the "Sethares dissonance" reel (SetharesDissonance-IG).

For each timbre, a fixed lower complex tone + an upper copy gliding unison →
octave, both built from that timbre's partials — so you HEAR the roughness curve
(and how harmonic vs stretched spectra differ). Voiced as a SOFT pad: steep
partial rolloff, a sub for body, a low-pass to tame the highs, gentle envelopes
and a slow stereo detune — so even the dissonant sweeps stay easy on the ear.

Timeline matches SetharesDissonance.tsx; the glide matches the eased sweep.
Run: python render_sethares_audio.py  →  public/audio/sethares.wav
"""
from __future__ import annotations

import json
from pathlib import Path
from wave import open as wave_open

import numpy as np
from scipy.signal import butter, sosfilt

HERE = Path(__file__).resolve().parent
OUT = HERE / "public" / "audio" / "sethares.wav"
SR = 44_100
FPS = 30

INTRO = 24
SWEEP = 150
HOLD = 70
BEAT = SWEEP + HOLD
OUTRO = 34

dat = json.load(open(HERE / "public" / "sethares.json", encoding="utf-8"))
F0 = dat["f0"]
TIMBRES = dat["timbres"]
N_FRAMES = INTRO + len(TIMBRES) * BEAT + OUTRO

AUDIO_DECAY = 0.62   # steep partial rolloff → mellow, less harsh
N_AUDIO = 5          # cap audible partials (drop the clangy top ones)
LP_HZ = 1900.0       # low-pass to tame the highs
DETUNE = 0.0035


def render_beat(partials) -> np.ndarray:
    dur = BEAT / FPS + 0.7
    n = int(dur * SR)
    t = np.arange(n) / SR
    frame = t * FPS
    u = np.clip(frame / SWEEP, 0.0, 1.0)
    alpha = 1.0 + 0.5 * (1.0 - np.cos(np.pi * u))   # eased sweep, holds at 2

    parts = partials[:N_AUDIO]
    left = np.zeros(n)
    right = np.zeros(n)
    # soft sub for body/warmth
    sub = 0.5 * np.sin(2 * np.pi * (F0 / 2) * t)
    left += sub
    right += sub
    for i, p in enumerate(parts):
        a = AUDIO_DECAY ** i
        for acc, df in ((0, 1 - DETUNE), (1, 1 + DETUNE)):  # gentle stereo width
            lo = a * np.sin(2 * np.pi * p * F0 * df * t)
            f_inst = p * F0 * df * alpha
            up = 0.62 * a * np.sin(2 * np.pi * np.cumsum(f_inst) / SR)  # swept, quieter
            (left if acc == 0 else right)[:] += lo + up

    body = np.stack([left, right], axis=1)
    # low-pass to roll off the bright top
    sos = butter(3, LP_HZ, btype="low", fs=SR, output="sos")
    body[:, 0] = sosfilt(sos, body[:, 0])
    body[:, 1] = sosfilt(sos, body[:, 1])

    # soft pad envelope: slow attack, gentle release
    env = np.ones(n)
    at = int(0.18 * SR)
    rl = int(0.8 * SR)
    env[:at] = np.sin(np.linspace(0, np.pi / 2, at))
    env[-rl:] *= np.sin(np.linspace(np.pi / 2, 0, rl))
    return body * env[:, None]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    total_n = int(N_FRAMES / FPS * SR) + int(0.8 * SR)
    master = np.zeros((total_n, 2))

    for ti, T in enumerate(TIMBRES):
        beat = render_beat(T["partials"])
        start = int((INTRO + ti * BEAT) / FPS * SR)
        end = min(start + beat.shape[0], master.shape[0])
        master[start:end] += beat[: end - start]

    master = master[: int(N_FRAMES / FPS * SR)]
    peak = float(np.max(np.abs(master)))
    if peak > 1e-6:
        master *= (10 ** (-4.5 / 20.0)) / peak   # extra headroom = gentler
    head = int(0.05 * SR)
    tail = int(0.45 * SR)
    master[:head] *= np.linspace(0, 1, head)[:, None]
    master[-tail:] *= np.linspace(1, 0, tail)[:, None]

    arr16 = np.clip(master * 32767.0, -32768, 32767).astype("<i2")
    with wave_open(str(OUT), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(arr16.tobytes())
    print(f"Wrote {OUT}  ({master.shape[0]/SR:.2f}s)")


if __name__ == "__main__":
    main()
