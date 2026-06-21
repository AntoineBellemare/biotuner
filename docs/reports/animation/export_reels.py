"""
Build per-reel data + soundtracks for the Instagram reels pipeline.

Each reel is declared once here as a small spec (chord sequence + timing +
audio config). This script writes:

  * public/reels/<id>.json    — the chord sequence + timing the React scene
                                reads (tiny; the cymatics field itself is
                                computed live in-canvas per frame, so no
                                heavy field data is exported).
  * public/audio/<id>.wav     — a chord-synced soundtrack rendered with the
                                same additive-synth voice as the flagship
                                render_audio.py, so audio and visuals stay
                                locked.

Run (needs the biotuner env — imports nothing heavy, just numpy/scipy):

    python docs/reports/animation/export_reels.py
    python docs/reports/animation/export_reels.py --only Reel02-Cymatics
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from wave import open as wave_open

import numpy as np

# Reuse the flagship synth voice so every reel shares the sonic brand.
from render_audio import SR, voice_chord, chladni_morph_filter
from intro_synths import singing_bowl

HERE = Path(__file__).resolve().parent
PUB = HERE / "public"
REELS_DIR = PUB / "reels"
AUDIO_DIR = PUB / "audio"


# ======================================================================
# Reel registry — each entry is one spec dict.
# ======================================================================
#
# A reel spec the EXPORTER understands:
#   id                 composition id (must match src/reels/specs.ts)
#   fps                frames per second
#   frames_per_segment frames per chord→chord morph
#   chords             [{name, label, ratios:[...]}]   (ratios = integer
#                      plate wavenumbers; the cymatics scene reads these)
#   symmetry           'd4_max' | 'd4_sum' | 'none'   (visual D4 hint)
#   root_hz            audio: frequency assigned to each chord's LOWEST voice
#   loop               whether the last chord morphs back to the first
#
# The TOTAL frame count = len(chords) * frames_per_segment when looping
# (one morph per chord, the last wrapping to the first).


REEL02_CYMATICS = {
    "id": "Reel02-Cymatics",
    "fps": 30,
    "frames_per_segment": 60,  # 2.0 s per chord
    "intro_frames": 90,        # 3.0 s brand opening
    "symmetry": "d4_max",
    "root_hz": 130.81,  # C3 assigned to each chord's lowest voice
    "loop": True,
    "hold_fraction": 0.0,      # continuous morph
    "hook": "every chord has a <b>shape</b>",
    "intro": {
        "title": "BIOTUNER",
        "tagline": "Visualizing and sonifying biological signals",
        "topic": "Harmonic Geometry",
        "motif": "flower_of_life",
        "accent": "#7ad6c1",
    },
    "chords": [
        {"name": "Major",      "label": "major",      "ratios": [4, 5, 6]},
        {"name": "Diminished", "label": "diminished", "ratios": [5, 6, 7]},
        {"name": "Suspended",  "label": "sus 4",      "ratios": [6, 8, 9]},
        {"name": "Minor",      "label": "minor",      "ratios": [10, 12, 15]},
        {"name": "Augmented",  "label": "augmented",  "ratios": [16, 20, 25]},
    ],
}

# Consonance palette for the interval labels (calm teal → tense red).
_TEAL = "#7ad6c1"
_GOLD = "#e8d68a"
_AMBER = "#e8a26b"
_RED = "#e87a7a"

REEL03_INTERVALS = {
    "id": "Reel03-Intervals",
    "fps": 30,
    "frames_per_segment": 66,  # 2.2 s per interval (held, then morph)
    "intro_frames": 90,
    "symmetry": "d4_max",
    "root_hz": 130.81,
    "loop": True,
    "hold_fraction": 0.58,     # show/hear each interval clearly, then morph
    "hook": "can you <b>see</b> dissonance?",
    "intro": {
        "title": "BIOTUNER",
        "tagline": "Visualizing and sonifying biological signals",
        "topic": "Consonance & Dissonance",
        "motif": "flower_of_life",
        "accent": "#9bb1e8",
    },
    # Ordered most-consonant → most-dissonant. ratios = integer plate
    # wavenumbers; ratio_str = the musical interval; tag = perceptual label.
    "chords": [
        {"name": "Octave",        "label": "octave",  "ratios": [4, 8],
         "ratio_str": "1 : 2",  "tag": "perfectly consonant", "accent": _TEAL},
        {"name": "Perfect Fifth", "label": "fifth",   "ratios": [4, 6],
         "ratio_str": "2 : 3",  "tag": "consonant",           "accent": _TEAL},
        {"name": "Perfect Fourth","label": "fourth",  "ratios": [6, 8],
         "ratio_str": "3 : 4",  "tag": "consonant",           "accent": _TEAL},
        {"name": "Major Third",   "label": "third",   "ratios": [4, 5],
         "ratio_str": "4 : 5",  "tag": "sweet",               "accent": _GOLD},
        {"name": "Minor Third",   "label": "minorthird", "ratios": [5, 6],
         "ratio_str": "5 : 6",  "tag": "sweet",               "accent": _GOLD},
        {"name": "Major Second",  "label": "second",  "ratios": [8, 9],
         "ratio_str": "8 : 9",  "tag": "mild tension",        "accent": _AMBER},
        {"name": "Tritone",       "label": "tritone", "ratios": [5, 7],
         "ratio_str": "5 : 7",  "tag": "restless",            "accent": _AMBER},
        {"name": "Minor Second",  "label": "minorsecond", "ratios": [15, 16],
         "ratio_str": "15 : 16", "tag": "dissonant",          "accent": _RED},
    ],
}

# ── Famous-song reels (Reel 04 + variants) — synthesised chord progressions ──
from song_chords import SONGS, song_chords  # noqa: E402


def _song_reel(reel_id: str, song_id: str) -> dict:
    s = SONGS[song_id]
    return {
        "id": reel_id,
        "fps": 30,
        "frames_per_segment": 60,   # 2.0 s per chord
        "intro_frames": 90,
        "symmetry": "d4_max",
        "root_hz": 130.81,          # unused: chords carry explicit freqs
        "loop": True,
        "hold_fraction": 0.55,      # settle each chord, then morph
        "portamento_s": 0.04,       # crisp chord changes (not gliding)
        "hook": f"<b>{s['title']}</b> &middot; the chords",
        "intro": {
            "title": "BIOTUNER",
            "tagline": "Visualizing and sonifying biological signals",
            "topic": s["title"],
            "motif": "flower_of_life",
            "accent": s["accent"],
        },
        "chords": song_chords(song_id),
    }


REEL04_HEYJUDE = _song_reel("Reel04-HeyJude", "HeyJude")
REEL05_LETITBE = _song_reel("Reel05-LetItBe", "LetItBe")
REEL06_CANON = _song_reel("Reel06-Canon", "CanonInD")

REELS = {
    r["id"]: r
    for r in [
        REEL02_CYMATICS,
        REEL03_INTERVALS,
        REEL04_HEYJUDE,
        REEL05_LETITBE,
        REEL06_CANON,
    ]
}


# ======================================================================
# Audio
# ======================================================================
def _chord_freqs(ratios: list[float], root_hz: float) -> list[float]:
    """Map integer plate-wavenumber ratios to audio frequencies, the
    chord's lowest voice landing on ``root_hz``."""
    m = min(ratios)
    return [root_hz * (r / m) for r in ratios]


def render_chord_morph(spec: dict) -> np.ndarray:
    """The looping chord-pad morph (no intro, no bed). Returns (morph_n, 2).

    Each chord's audio frequencies are taken from its explicit ``freqs``
    (song reels carry real chord voicings) or derived from its ``ratios``
    (abstract-chord reels). ``portamento_s`` controls how much pitch glides
    between chords — small for song reels so chords change crisply."""
    fps = spec["fps"]
    seg_dur = spec["frames_per_segment"] / fps
    root_hz = spec["root_hz"]
    chords = spec["chords"]
    n_seg = len(chords)
    porta_s = spec.get("portamento_s", 0.5)

    morph_n = int(n_seg * seg_dur * SR)
    tail = int(1.0 * SR)
    morph = np.zeros((morph_n + tail, 2))
    prev_freqs: list[float] | None = None
    for i in range(n_seg):
        ch = chords[i]
        freqs = ch["freqs"] if "freqs" in ch else _chord_freqs(ch["ratios"], root_hz)
        porta = prev_freqs
        if porta is not None and len(porta) < len(freqs):
            porta = porta + [porta[-1]] * (len(freqs) - len(porta))
        seg = voice_chord(
            freqs, seg_dur + 1.0, sr=SR,
            attack=0.35, release=1.2, shimmer=0.5, root_amp=0.17,
            portamento_from=porta, portamento_s=porta_s,
        )
        start = int(i * seg_dur * SR)
        end = min(start + seg.shape[0], morph.shape[0])
        morph[start:end] += seg[: end - start]
        prev_freqs = freqs
    morph = morph[:morph_n]
    return chladni_morph_filter(morph, sweep_period_s=seg_dur * 2, depth=0.25)


def render_reel_audio(spec: dict) -> np.ndarray:
    """Full soundtrack:
        intro  = singing-bowl strike (rings into the morph)
        morph  = looping chord-pad cymatics morph
    Returns float (n, 2) of exactly total_frames length."""
    fps = spec["fps"]
    intro_frames = spec.get("intro_frames", 0)
    intro_n = int(intro_frames / fps * SR)
    intro_dur = intro_frames / fps

    morph = render_chord_morph(spec)
    morph_n = morph.shape[0]
    total_n = intro_n + morph_n

    # Bowl leads the intro and rings (rendered long) into the first chord;
    # then the chord morph carries the rest. No ambience bed.
    out = np.zeros((total_n, 2))
    if intro_n > 0:
        bowl = singing_bowl(intro_dur + 2.0)
        bn = min(bowl.shape[0], total_n)
        out[:bn] += bowl[:bn] * 0.95
    out[intro_n:intro_n + morph_n] += morph

    peak = float(np.max(np.abs(out)))
    if peak > 1e-6:
        out *= (10 ** (-1.0 / 20.0)) / peak
    head = int(0.04 * SR)
    tail_f = int(0.25 * SR)
    out[:head] *= np.linspace(0, 1, head)[:, None]
    out[-tail_f:] *= np.linspace(1, 0, tail_f)[:, None]
    return out


def write_wav(stereo: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr16 = np.clip(stereo * 32767.0, -32768, 32767).astype("<i2")
    with wave_open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(arr16.tobytes())


# ======================================================================
# Per-reel build
# ======================================================================
def build_reel(spec: dict) -> None:
    rid = spec["id"]
    fps = spec["fps"]
    seg = spec["frames_per_segment"]
    intro_frames = spec.get("intro_frames", 0)
    morph_frames = len(spec["chords"]) * seg  # looping
    total_frames = intro_frames + morph_frames
    print(f"[{rid}] intro {intro_frames}f + {len(spec['chords'])} chords x "
          f"{seg}f = {total_frames}f ({total_frames / fps:.1f}s)")

    # 1. Data JSON the React scene reads.
    np.random.seed(7)  # deterministic audio phases
    REELS_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "id": rid,
        "fps": fps,
        "frames_per_segment": seg,
        "intro_frames": intro_frames,
        "morph_frames": morph_frames,
        "total_frames": total_frames,
        "symmetry": spec["symmetry"],
        "loop": spec["loop"],
        "hold_fraction": spec.get("hold_fraction", 0),
        "hook": spec.get("hook"),
        "intro": spec.get("intro"),
        "chords": spec["chords"],
        "audio": f"audio/{rid}.wav",
    }
    out_json = REELS_DIR / f"{rid}.json"
    out_json.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"  wrote {out_json.relative_to(HERE)} "
          f"({out_json.stat().st_size / 1024:.1f} KB)")

    # 2. Soundtrack.
    stereo = render_reel_audio(spec)
    out_wav = AUDIO_DIR / f"{rid}.wav"
    write_wav(stereo, out_wav)
    print(f"  wrote {out_wav.relative_to(HERE)} "
          f"({stereo.shape[0] / SR:.1f}s, {out_wav.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="Build just this reel id.")
    args = ap.parse_args()

    ids = [args.only] if args.only else list(REELS)
    for rid in ids:
        if rid not in REELS:
            raise SystemExit(f"unknown reel id {rid!r}; have {list(REELS)}")
        build_reel(REELS[rid])


if __name__ == "__main__":
    main()
