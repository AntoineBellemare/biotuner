"""
Soundtrack for the "Sethares dissonance" reel (SetharesDissonance-IG).

The actual swept dyad: a fixed lower complex tone (6 harmonics) + an upper
complex tone whose pitch glides from unison to the octave — so you HEAR the
Plomp-Levelt roughness peak just above unison and dip at each consonance, in
sync with the curve marker. The glide matches the scene's eased sweep
(Easing.inOut(sin) == 0.5·(1−cos(π·t))).

Run: python render_sethares_audio.py  →  public/audio/sethares.wav
"""
from __future__ import annotations

from pathlib import Path
from wave import open as wave_open

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "public" / "audio" / "sethares.wav"
SR = 44_100
FPS = 30

INTRO = 30
SWEEP = 500
OUTRO = 46
N_FRAMES = INTRO + SWEEP + OUTRO
F0 = 261.63
NP = 6


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = int(N_FRAMES / FPS * SR)
    t = np.arange(n) / SR
    frame = t * FPS

    # eased sweep alpha(t): 1 → 2 over [INTRO, INTRO+SWEEP]
    u = np.clip((frame - INTRO) / SWEEP, 0.0, 1.0)
    alpha = 1.0 + 0.5 * (1.0 - np.cos(np.pi * u))

    amps = 0.88 ** np.arange(NP)
    lower = np.zeros(n)
    upper = np.zeros(n)
    for k in range(1, NP + 1):
        a = amps[k - 1]
        lower += a * np.sin(2 * np.pi * k * F0 * t)
        # instantaneous-frequency integration for the glide
        f_inst = k * F0 * alpha
        phase = 2 * np.pi * np.cumsum(f_inst) / SR
        upper += a * np.sin(phase)

    mono = 0.5 * (lower + upper)
    # gentle stereo: lower left-ish, upper right-ish
    left = 0.5 * lower + 0.5 * mono
    right = 0.5 * upper + 0.5 * mono
    out = np.stack([left, right], axis=1)

    peak = float(np.max(np.abs(out)))
    if peak > 1e-6:
        out *= (10 ** (-2.0 / 20.0)) / peak
    head = int(0.05 * SR)
    tail = int(0.5 * SR)
    out[:head] *= np.linspace(0, 1, head)[:, None]
    out[-tail:] *= np.linspace(1, 0, tail)[:, None]

    arr16 = np.clip(out * 32767.0, -32768, 32767).astype("<i2")
    with wave_open(str(OUT), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(arr16.tobytes())
    print(f"Wrote {OUT}  ({n/SR:.2f}s)")


if __name__ == "__main__":
    main()
