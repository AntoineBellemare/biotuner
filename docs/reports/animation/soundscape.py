"""
Natural-soundscape bed builder for the reel pipeline.

Decodes the public-domain nature recordings in assets/soundscapes/ and mixes
a gentle, loopable bed (stream + distant forest) of any length. Used by
export_reels.py (quiet bed under a reel) and the audition scripts.

Sources (both public domain) — see assets/soundscapes/SOURCES.md.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from wave import open as wave_open

import numpy as np
from scipy.signal import butter, sosfilt

SR = 44_100
HERE = Path(__file__).resolve().parent
SOUNDS = HERE / "assets" / "soundscapes"


def decode_ogg(path: Path) -> np.ndarray:
    """ffmpeg-decode an ogg/vorbis file to float stereo at SR → (n, 2)."""
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "x.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
             "-ar", str(SR), "-ac", "2", "-f", "wav", str(wav)],
            check=True,
        )
        with wave_open(str(wav), "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        a = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
        return a.reshape(-1, 2)


def loop_to(x: np.ndarray, n: int) -> np.ndarray:
    if x.shape[0] >= n:
        return x[:n]
    reps = int(np.ceil(n / x.shape[0]))
    return np.tile(x, (reps, 1))[:n]


def highpass(x: np.ndarray, fc: float = 80.0) -> np.ndarray:
    sos = butter(2, fc, btype="high", fs=SR, output="sos")
    return np.stack([sosfilt(sos, x[:, c]) for c in range(2)], axis=1)


def calmest_slice(x: np.ndarray, dur: float) -> np.ndarray:
    """Lowest-energy `dur`-second window (skips loud transients like crow caws)."""
    n = int(dur * SR)
    if x.shape[0] <= n:
        return loop_to(x, n)
    energy = (x.mean(axis=1)) ** 2
    hop = int(0.25 * SR)
    best_i, best_e = 0, np.inf
    for i in range(0, x.shape[0] - n, hop):
        e = float(energy[i:i + n].mean())
        if e < best_e:
            best_e, best_i = e, i
    return x[best_i:best_i + n]


def build_bed(
    dur: float,
    *,
    water_level: float = 0.6,
    forest_level: float = 0.28,
    peak: float = 0.5,
    fade_s: float = 0.3,
) -> np.ndarray:
    """A gentle, loopable nature bed of `dur` seconds, normalised to `peak`.

    Water (continuous stream) is the soothing base; a calm slice of forest
    ambience adds distant life. Returns (n, 2)."""
    n = int(dur * SR)
    water = loop_to(decode_ogg(SOUNDS / "hemlock_stream.ogg"), n)
    forest = calmest_slice(decode_ogg(SOUNDS / "forest_ambience.ogg"), dur)
    bed = highpass(water_level * water + forest_level * forest)
    h = int(fade_s * SR)
    bed[:h] *= np.linspace(0, 1, h)[:, None]
    bed[-h:] *= np.linspace(1, 0, h)[:, None]
    pk = float(np.max(np.abs(bed)))
    if pk > 1e-9:
        bed *= peak / pk
    return bed
