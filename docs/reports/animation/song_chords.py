"""
Famous-song chord progressions → cymatics wavenumbers + audio frequencies.

A song is a list of (scale-degree, quality, symbol, roman) chords in a key.
For each chord this produces:

  * ratios  — small-integer plate wavenumbers for the cymatics visual,
              scaled GLOBALLY across the song so the patterns stay legible
              while still differing per chord (harmonic motion shows).
  * freqs   — audio frequencies, octave-voiced into a consistent register
              with a bass root, so the synthesized progression sounds like
              the song's chords (not an ever-climbing stack).
  * label   — the chord QUALITY (drives the on-screen label colour).
  * name / ratio_str — chord symbol + roman numeral for display.

Only the harmony (chord progression) is synthesised — no copyrighted
recording is used. The reel is branded with the song name so the
progression is recognisable; the user can overlay the real track in-app.
"""
from __future__ import annotations

from fractions import Fraction

# Chord qualities as just-intonation ratios over the chord root.
QUALITIES: dict[str, list[Fraction]] = {
    "maj":  [Fraction(1), Fraction(5, 4), Fraction(3, 2)],
    "min":  [Fraction(1), Fraction(6, 5), Fraction(3, 2)],
    "dom7": [Fraction(1), Fraction(5, 4), Fraction(3, 2), Fraction(7, 4)],
    "maj7": [Fraction(1), Fraction(5, 4), Fraction(3, 2), Fraction(15, 8)],
    "min7": [Fraction(1), Fraction(6, 5), Fraction(3, 2), Fraction(9, 5)],
    "sus4": [Fraction(1), Fraction(4, 3), Fraction(3, 2)],
    "dim":  [Fraction(1), Fraction(6, 5), Fraction(7, 5)],
}

# Scale-degree root → JI ratio above the tonic.
DEGREE: dict[str, Fraction] = {
    "I":   Fraction(1),    "II":  Fraction(9, 8),  "III": Fraction(5, 4),
    "IV":  Fraction(4, 3), "V":   Fraction(3, 2),  "VI":  Fraction(5, 3),
    "VII": Fraction(15, 8),
    "bII": Fraction(16, 15), "bIII": Fraction(6, 5), "bVI": Fraction(8, 5),
    "bVII": Fraction(16, 9),
}

QUALITY_LABEL = {
    "maj": "major", "min": "minor", "dom7": "dom7", "maj7": "maj7",
    "min7": "minor", "sus4": "sus4", "dim": "diminished",
}


def _octave_reduce(x: float) -> float:
    while x >= 2.0:
        x /= 2.0
    while x < 1.0:
        x *= 2.0
    return x


def build_song(
    prog: list[tuple[str, str, str, str]],
    *,
    tonic_hz: float = 130.81,   # tonic of the key (C3 by default)
    base_hz: float = 196.0,     # register the chord voicings sit in (~G3)
    wn_lo: float = 4.0,
    wn_hi: float = 11.0,
) -> list[dict]:
    """Turn ``prog`` = [(degree, quality, symbol, roman), …] into chord dicts.

    Visual wavenumbers are scaled globally across the song; audio freqs are
    octave-voiced with a bass root."""
    # Absolute JI ratios (to tonic) for every chord.
    abs_chords = []
    for degree, quality, symbol, roman in prog:
        root = DEGREE[degree]
        ratios = [root * q for q in QUALITIES[quality]]
        abs_chords.append((symbol, roman, quality, root, ratios))

    allr = [float(r) for *_, ratios in abs_chords for r in ratios]
    rmin, rmax = min(allr), max(allr)
    span = (rmax - rmin) or 1.0

    def to_wn(r: Fraction) -> int:
        wn = wn_lo + (float(r) - rmin) / span * (wn_hi - wn_lo)
        return max(2, round(wn))

    out: list[dict] = []
    for symbol, roman, quality, root, ratios in abs_chords:
        # Visual: globally-scaled small-int wavenumbers (distinct per chord).
        wn = [to_wn(r) for r in ratios]

        # Audio: octave-voice each tone into [base, 2·base), + a bass root.
        voiced = sorted({round(base_hz * _octave_reduce(float(r)), 3)
                         for r in ratios})
        bass = round(base_hz * 0.5 * _octave_reduce(float(root)), 3)
        freqs = [bass] + voiced

        out.append({
            "name": symbol,
            "label": QUALITY_LABEL.get(quality, "major"),
            "ratio_str": roman,
            "ratios": wn,
            "freqs": freqs,
        })
    return out


# ── Song library — recognisable progressions, just the chords ─────────────
# Each entry: key tonic (Hz) + progression as (degree, quality, symbol, roman).

SONGS: dict[str, dict] = {
    "HeyJude": {
        "title": "Hey Jude",
        "tonic_hz": 174.61,  # F3
        "accent": "#e8d68a",
        "prog": [
            ("I", "maj", "F", "I"),
            ("V", "maj", "C", "V"),
            ("V", "dom7", "C7", "V7"),
            ("I", "maj", "F", "I"),
            ("IV", "maj", "B♭", "IV"),
            ("I", "maj", "F", "I"),
            ("V", "dom7", "C7", "V7"),
            ("I", "maj", "F", "I"),
        ],
    },
    "LetItBe": {
        "title": "Let It Be",
        "tonic_hz": 130.81,  # C3
        "accent": "#7ad6c1",
        "prog": [
            ("I", "maj", "C", "I"),
            ("V", "maj", "G", "V"),
            ("VI", "min", "Am", "vi"),
            ("IV", "maj", "F", "IV"),
            ("I", "maj", "C", "I"),
            ("V", "maj", "G", "V"),
            ("IV", "maj", "F", "IV"),
            ("I", "maj", "C", "I"),
        ],
    },
    "CanonInD": {
        "title": "Canon in D",
        "tonic_hz": 146.83,  # D3
        "accent": "#9bb1e8",
        "prog": [
            ("I", "maj", "D", "I"),
            ("V", "maj", "A", "V"),
            ("VI", "min", "Bm", "vi"),
            ("III", "min", "F♯m", "iii"),
            ("IV", "maj", "G", "IV"),
            ("I", "maj", "D", "I"),
            ("IV", "maj", "G", "IV"),
            ("V", "maj", "A", "V"),
        ],
    },
}


def song_chords(song_id: str) -> list[dict]:
    s = SONGS[song_id]
    return build_song(s["prog"], tonic_hz=s["tonic_hz"])


if __name__ == "__main__":
    for sid, s in SONGS.items():
        print(f"\n{sid} — {s['title']}")
        for c in song_chords(sid):
            print(f"  {c['name']:4s} {c['ratio_str']:4s}  wn={c['ratios']}  "
                  f"freqs={[round(f,1) for f in c['freqs']]}")
