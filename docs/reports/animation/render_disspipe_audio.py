"""
Soundtrack for the "signal → scale" pipeline reel (DissonancePipeline-IG).

Per signal: the extracted peaks (transposed to audio) sustain as a chord while
the comb builds, then an upper copy glides unison→octave during the sweep so you
HEAR the roughness curve. Timeline matches DissonancePipeline.tsx.

Run: python render_disspipe_audio.py  →  public/audio/diss_pipeline.wav
"""
from __future__ import annotations

import json
from pathlib import Path
from wave import open as wave_open

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "public" / "audio" / "diss_pipeline.wav"
SR = 44_100
FPS = 30
AUDIO_F = 261.63

INTRO = 24
BEAT = 408
OUTRO = 30
COMB_START = 160
SWEEP_START = 235
SWEEP_END = 372

dat = json.load(open(HERE / "public" / "diss_pipeline.json", encoding="utf-8"))
SIGNALS = [dat["signal"], dat["eeg"]]
N_FRAMES = INTRO + len(SIGNALS) * BEAT + OUTRO


def partial(freq, t, amp):
    return amp * np.sin(2 * np.pi * freq * t)


def render_beat(peaks_audio, amps) -> np.ndarray:
    n = int(BEAT / FPS * SR) + int(0.6 * SR)
    t = np.arange(n) / SR
    frame = t * FPS
    # sweep alpha (eased), active SWEEP_START..SWEEP_END, then holds at 2
    u = np.clip((frame - SWEEP_START) / (SWEEP_END - SWEEP_START), 0.0, 1.0)
    alpha = 1.0 + 0.5 * (1.0 - np.cos(np.pi * u))

    lower = np.zeros(n)
    upper = np.zeros(n)
    for f, a in zip(peaks_audio, amps):
        lower += partial(f, t, a)
        upper += a * np.sin(2 * np.pi * np.cumsum(f * alpha) / SR)

    # lower chord fades in at the comb, upper fades in at the sweep
    env_l = np.clip((frame - COMB_START) / 24, 0, 1)
    env_u = np.clip((frame - SWEEP_START) / 24, 0, 1)
    rel = np.clip((frame - (BEAT - 14)) / 14, 0, 1)
    fade = (1 - rel)
    out = np.stack([(0.55 * lower * env_l + 0.3 * upper * env_u) * fade,
                    (0.3 * lower * env_l + 0.55 * upper * env_u) * fade], axis=1)
    return out


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    total_n = int(N_FRAMES / FPS * SR) + int(0.6 * SR)
    master = np.zeros((total_n, 2))

    for ti, S in enumerate(SIGNALS):
        peaks = np.array([p["f"] for p in S["peaks"]], float)
        amps = np.array([p["a"] for p in S["peaks"]], float)
        peaks_audio = peaks / peaks.min() * AUDIO_F
        beat = render_beat(peaks_audio, amps)
        start = int((INTRO + ti * BEAT) / FPS * SR)
        end = min(start + beat.shape[0], master.shape[0])
        master[start:end] += beat[: end - start]

    master = master[: int(N_FRAMES / FPS * SR)]
    peak = float(np.max(np.abs(master)))
    if peak > 1e-6:
        master *= (10 ** (-2.0 / 20.0)) / peak
    head = int(0.05 * SR)
    tail = int(0.4 * SR)
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
