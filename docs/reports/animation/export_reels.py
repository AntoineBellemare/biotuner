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
    # Ordered most-consonant → most-dissonant by Tenney height (so the
    # cymatics complexity rises monotonically and the tritone — the most
    # dissonant — is the complex climax). ratios = integer plate wavenumbers.
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
        {"name": "Minor Second",  "label": "minorsecond", "ratios": [15, 16],
         "ratio_str": "15 : 16", "tag": "harsh",              "accent": _RED},
        # Tritone (23:16) — the most dissonant; a genuinely complex, bold
        # nodal lattice. (The simple septimal 5:7 looked far too clean; the
        # 45:32 was so high-wavenumber it rendered as a faint speckle.)
        {"name": "Tritone",       "label": "tritone", "ratios": [16, 23],
         "ratio_str": "16 : 23", "tag": "the devil's interval", "accent": _RED},
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

# ── Reel 07 — Brain vs Heart, as GALLERIES (a wall of each) ──────────────────
from biosignal_chords import brain_gallery, heart_gallery_real  # noqa: E402

_r07_brains = brain_gallery()
_r07_hearts = heart_gallery_real()  # real-ECG-precision hearts (decimal Hz)
REEL07_BRAINHEART = {
    "id": "Reel07-BrainHeart",
    "fps": 30,
    "frames_per_segment": 165,  # 5.5 s per wall
    "intro_frames": 90,
    "symmetry": "d4_max",
    "root_hz": 130.81,
    "loop": True,
    "hold_fraction": 0.0,
    "audio_style": "brainheart",  # brain = shimmer pad, heart = heartbeat pulse
    "scene": "gallery",
    "hook": "nine <b>minds</b>, nine <b>hearts</b>",
    "gallery_phases": [
        {"title": "BRAINS", "subtitle": "EEG — every mind, intricate",
         "accent": "#8a9be8", "cells": _r07_brains},
        {"title": "HEARTS", "subtitle": "ECG — every beat, ordered",
         "accent": "#e87a8a", "cells": _r07_hearts},
    ],
    "intro": {
        "title": "BIOTUNER",
        "tagline": "Visualizing and sonifying biological signals",
        "topic": "Brain vs Heart",
        "motif": "flower_of_life",
        "accent": "#b89be8",
    },
    # Audio: one brain sound, one heart sound (2 walls = 2 sounds).
    "chords": [
        {**_r07_brains[0], "name": "BRAIN", "label": "brain"},
        {**_r07_hearts[0], "name": "HEART", "label": "heart"},
    ],
}

# ── Reel 08 — Four geometries at once, many chords (2×2 quadrant grid) ───────
_MANY_CHORDS = [
    {"name": "Major",     "label": "major",  "ratios": [4, 5, 6]},
    {"name": "Minor",     "label": "minor",  "ratios": [10, 12, 15]},
    {"name": "Sus 4",     "label": "sus4",   "ratios": [6, 8, 9]},
    {"name": "Dom 7",     "label": "dom7",   "ratios": [4, 5, 6, 7]},
    {"name": "Maj 7",     "label": "maj7",   "ratios": [8, 10, 12, 15]},
    {"name": "Dim 7",     "label": "dim7",   "ratios": [5, 6, 7, 9]},
    {"name": "Augmented", "label": "aug",    "ratios": [16, 20, 25]},
    {"name": "Min 7",     "label": "minor",  "ratios": [10, 12, 15, 18]},
]
REEL08_MANYSHAPES = {
    "id": "Reel08-ManyShapes",
    "fps": 30,
    "frames_per_segment": 66,   # 2.2 s per chord
    "intro_frames": 90,
    "symmetry": "d4_max",
    "root_hz": 130.81,
    "loop": True,
    "hold_fraction": 0.5,       # settle each chord, then morph (all 4 quadrants)
    "portamento_s": 0.4,
    "scene": "quad",
    "hook": "one chord, <b>four cymatics</b>",
    "intro": {
        "title": "BIOTUNER",
        "tagline": "Visualizing and sonifying biological signals",
        "topic": "Four Cymatics",
        "motif": "flower_of_life",
        "accent": "#7ad6c1",
    },
    "chords": _MANY_CHORDS,
}

# ── Reel 09 — Canon in D as a harmonograph (same music, different geometry) ──
_canon_chords = song_chords("CanonInD")
REEL09_CANON_HARMONO = {
    "id": "Reel09-CanonHarmonograph",
    "fps": 30,
    "frames_per_segment": 60,
    "intro_frames": 90,
    "symmetry": "d4_max",
    "root_hz": 130.81,
    "loop": True,
    "hold_fraction": 0.5,
    "portamento_s": 0.04,
    "scene": "multi",
    "geometries": ["harmonograph"] * len(_canon_chords),
    "hook": "<b>Canon in D</b> &middot; the same chords, drawn",
    "intro": {
        "title": "BIOTUNER",
        "tagline": "Visualizing and sonifying biological signals",
        "topic": "Canon in D — drawn",
        "motif": "flower_of_life",
        "accent": "#e8c98a",
    },
    "chords": _canon_chords,
}

# ── Reel 10 — Let It Be, every shape (each chord a different geometry) ───────
_litb = song_chords("LetItBe")
_geo_cycle = ["cymatics", "lissajous", "harmonograph", "interference"]
REEL10_LETITBE_SHAPES = {
    "id": "Reel10-LetItBeShapes",
    "fps": 30,
    "frames_per_segment": 66,
    "intro_frames": 90,
    "symmetry": "d4_max",
    "root_hz": 130.81,
    "loop": True,
    "hold_fraction": 0.5,
    "portamento_s": 0.04,
    "scene": "multi",
    "geometries": [_geo_cycle[i % len(_geo_cycle)] for i in range(len(_litb))],
    "hook": "<b>Let It Be</b> &middot; every shape",
    "intro": {
        "title": "BIOTUNER",
        "tagline": "Visualizing and sonifying biological signals",
        "topic": "Let It Be — every shape",
        "motif": "flower_of_life",
        "accent": "#7ad6c1",
    },
    "chords": _litb,
}

# ── Reel 12 — Meditative: 1-min EEG cymatics morph + palette morph ───────────
from biosignal_chords import meditative_eeg_sequence  # noqa: E402

REEL12_MEDITATIVE = {
    "id": "Reel12-Meditative",
    "fps": 30,
    "frames_per_segment": 150,  # 5 s per EEG state, very slow
    "intro_frames": 0,          # no brand intro — pure meditation
    "symmetry": "d4_max",
    "root_hz": 130.81,
    "loop": True,
    "hold_fraction": 0.0,       # continuous smooth morph
    "audio_style": "meditative",
    "scene": "meditative",
    "hook": None,
    "intro": None,
    "chords": meditative_eeg_sequence(),  # 12 EEG chords × 5 s = 60 s
}

REELS = {
    r["id"]: r
    for r in [
        REEL02_CYMATICS,
        REEL03_INTERVALS,
        REEL04_HEYJUDE,
        REEL05_LETITBE,
        REEL06_CANON,
        REEL07_BRAINHEART,
        REEL08_MANYSHAPES,
        REEL09_CANON_HARMONO,
        REEL10_LETITBE_SHAPES,
        REEL12_MEDITATIVE,
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


def _heartbeat_segment(dur: float, freqs: list[float], hr: float = 1.05) -> np.ndarray:
    """A 'lub-dub' heartbeat at ``hr`` Hz + a soft harmonic drone. (n, 2)."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    period = 1.0 / hr
    bt = 0.0
    while bt < dur:
        for off, f, amp, dec in [(0.0, 54.0, 1.0, 0.10), (0.17, 44.0, 0.62, 0.13)]:
            i0 = int((bt + off) * SR)
            lt = (np.arange(n) - i0) / SR
            env = np.where(lt >= 0, np.exp(-np.maximum(lt, 0) / dec), 0.0)
            out += amp * env * np.sin(2 * np.pi * f * np.maximum(lt, 0))
        bt += period
    # soft tonal drone from the heart's harmonic peaks
    drone = np.zeros(n)
    for f in freqs:
        drone += 0.05 * np.sin(2 * np.pi * f * t)
    mono = 0.55 * out + drone
    env_in = np.minimum(t / 0.05, 1.0)
    mono *= env_in
    return np.stack([mono, mono], axis=1)


def _brain_segment(dur: float, freqs: list[float],
                   porta: list[float] | None) -> np.ndarray:
    """Shimmery, inharmonic, ethereal pad — the brain's peak cluster."""
    return voice_chord(
        freqs, dur, sr=SR,
        attack=0.9, release=1.4, shimmer=0.95, root_amp=0.16,
        detune_cents=(0.0, 8.0, -8.0, 4.0), vibrato_hz=6.5, vibrato_cents=5.0,
        portamento_from=porta, portamento_s=0.6,
    )


def render_brainheart_morph(spec: dict) -> np.ndarray:
    """Per-segment audio that DIFFERS by signal: brain = shimmer pad,
    heart = heartbeat pulse. Driven by each chord's ``label``."""
    fps = spec["fps"]
    seg_dur = spec["frames_per_segment"] / fps
    chords = spec["chords"]
    n_seg = len(chords)
    morph_n = int(n_seg * seg_dur * SR)
    tail = int(1.2 * SR)
    out = np.zeros((morph_n + tail, 2))
    prev: list[float] | None = None
    for i, ch in enumerate(chords):
        freqs = ch["freqs"]
        if ch.get("label") == "heart":
            seg = _heartbeat_segment(seg_dur + 1.0, freqs)
        else:
            seg = _brain_segment(seg_dur + 1.0, freqs, prev)
        start = int(i * seg_dur * SR)
        end = min(start + seg.shape[0], out.shape[0])
        out[start:end] += seg[: end - start]
        prev = freqs
    return out[:morph_n]


def render_meditative_morph(spec: dict) -> np.ndarray:
    """A soft, continuous, evolving drone-pad through the EEG chords — long
    attacks/releases, heavy overlap, a sustained low drone, lots of air. (n,2)."""
    fps = spec["fps"]
    seg_dur = spec["frames_per_segment"] / fps
    chords = spec["chords"]
    n_seg = len(chords)
    total_n = int(n_seg * seg_dur * SR)
    tail = int(4.0 * SR)
    out = np.zeros((total_n + tail, 2))

    # Sustained low drone (C2 + C3 + G3), very soft, slow tremolo.
    t = np.arange(total_n + tail) / SR
    trem = 0.85 + 0.15 * np.sin(2 * np.pi * 0.07 * t)
    drone = np.zeros(total_n + tail)
    for f, a in [(65.41, 0.5), (130.81, 0.35), (196.0, 0.18)]:
        drone += a * np.sin(2 * np.pi * f * t)
    drone *= 0.10 * trem
    out[:, 0] += drone
    out[:, 1] += drone

    # Slowly-morphing pad: each EEG chord, long attack/release, big overlap.
    prev = None
    for i, ch in enumerate(chords):
        freqs = ch["freqs"]
        pad = voice_chord(
            freqs, seg_dur * 2.2, sr=SR,
            attack=seg_dur * 0.55, release=seg_dur * 1.1,
            shimmer=0.5, root_amp=0.14,
            detune_cents=(0.0, 6.0, -6.0), vibrato_hz=5.0, vibrato_cents=4.0,
            portamento_from=prev, portamento_s=seg_dur * 0.5,
        )
        start = int(i * seg_dur * SR)
        end = min(start + pad.shape[0], out.shape[0])
        out[start:end] += pad[: end - start]
        prev = freqs

    out = out[:total_n]
    # airy shimmer sweep
    out = chladni_morph_filter(out, sweep_period_s=18.0, depth=0.18)
    peak = float(np.max(np.abs(out)))
    if peak > 1e-6:
        out *= (10 ** (-2.0 / 20.0)) / peak  # leave gentle headroom
    head = int(1.5 * SR)
    tl = int(2.5 * SR)
    out[:head] *= np.linspace(0, 1, head)[:, None]
    out[-tl:] *= np.linspace(1, 0, tl)[:, None]
    return out


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

    style = spec.get("audio_style")
    if style == "brainheart":
        morph = render_brainheart_morph(spec)
    elif style == "meditative":
        morph = render_meditative_morph(spec)
    else:
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
        "scene": spec.get("scene", "cymatics"),
        "geometries": spec.get("geometries"),
        "gallery_phases": spec.get("gallery_phases"),
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
