"""
render_audio.py — synced soundtrack for GeometryV2 / GeometryV2-IG.

Outputs:
    public/audio/score.wav   (44.1 kHz, 16-bit PCM, stereo)

Design
------
* One continuous master matching MainV2.tsx's 30-fps timeline:
    title 3.0 s, harmonograph 16 s, knots 20 s, chladni 17.5 s,
    trees 20 s, clouds 15 s, outro 3 s   (= 94.5 s)
* Chord events are pulled from public/geometry.json so the audio is
  locked to the visuals.
* Voice = 3-voice detuned unison * 8-partial additive stack
  (1/n^1.3 amplitude, random phases) + shimmer octave & twelfth,
  5.5 Hz vibrato (~3 cents), ADSR, light low-pass, stereo spread.
  Warm and evolving — no raw oscillators.
* Chord-to-chord morphing: voice frequencies glide log-linearly from
  the previous chord's ratios to the new chord's ratios over ~0.45 s,
  and the previous chord's release tail bleeds into the next attack.
* Chladni scene gets an extra slow resonant band-pass sweep
  (1 cycle / 8 s, 180-900 Hz) layered in parallel, mirroring the
  plate-geometry visual morph with a watery, plate-like timbre.

Run from this directory:
    python render_audio.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from wave import open as wave_open

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

HERE = Path(__file__).resolve().parent
PUB = HERE / "public"
OUT = PUB / "audio" / "score.wav"

SR = 44_100
FPS = 30

# Must match docs/reports/animation/src/MainV2.tsx
SCENE_FRAMES = {
    "title":         90,
    "harmonograph": 4 * 120,
    "knots":        4 * 150,
    "chladni":      7 * 75,
    "trees":        4 * 150,
    "clouds":       3 * 150,
    "outro":         90,
}
ORDER = ["title", "harmonograph", "knots", "chladni", "trees", "clouds", "outro"]


# ----------------------------------------------------------------------
# Chord data — pulled from geometry.json where possible.
# ----------------------------------------------------------------------
def parse_ratios(s: str) -> list[float]:
    """Parse '4 : 5 : 6' or '1 : 5/4 : 8/5' to a list of floats normalised so min == 1."""
    parts = [p.strip() for p in s.split(":")]
    vals: list[float] = []
    for p in parts:
        if "/" in p:
            a, b = p.split("/")
            vals.append(float(a) / float(b))
        else:
            vals.append(float(p))
    m = min(vals)
    return [v / m for v in vals]


def _norm(ratios: list[float]) -> list[float]:
    """Normalise a ratio list so the lowest voice == 1.0 (treats `root_hz`
    as the actual bottom note in render_chord_sequence)."""
    m = min(ratios)
    return [r / m for r in ratios]


def load_chord_lists() -> dict[str, list[list[float]]]:
    geom = json.load(open(PUB / "geometry.json", encoding="utf-8"))
    s = geom["scenes"]

    # Harmonograph variants — peak ratios advertised in the scene captions
    harmonograph = [
        _norm([2, 3, 5, 7]),     # harmonic drift
        _norm([4, 5, 6]),        # major chord
        _norm([4, 5, 6, 7]),     # dom 7th
        _norm([10, 12, 15]),     # minor spiral
    ]
    knots = [
        parse_ratios("3:4:5"),
        parse_ratios("2:3:5"),
        parse_ratios("3:5:7"),
        parse_ratios("4:5:7"),
    ]
    chladni = [parse_ratios(it["ratios_str"]) for it in s["chladni_expanded"]["items"]]
    trees = [
        _norm([4, 5, 6]),        # Major
        _norm([6, 8, 9]),        # Sus4
        _norm([4, 5, 6, 7]),     # Dom7
        _norm([5, 6, 7, 9]),     # Dim7
    ]
    clouds = [
        _norm([4, 5, 6]),
        _norm([4, 5, 6, 7]),
        _norm([5, 6, 7, 9]),
    ]
    return {
        "harmonograph": harmonograph,
        "knots": knots,
        "chladni": chladni,
        "trees": trees,
        "clouds": clouds,
    }


# ----------------------------------------------------------------------
# Synth
# ----------------------------------------------------------------------
def adsr(n: int, sr: int, A=0.4, D=0.5, S=0.85, R=1.0) -> np.ndarray:
    a = max(int(A * sr), 1)
    d = max(int(D * sr), 1)
    r = max(int(R * sr), 1)
    sustain_n = max(n - a - d - r, 0)
    env = np.concatenate([
        np.linspace(0, 1, a, endpoint=False),
        np.linspace(1, S, d, endpoint=False),
        np.full(sustain_n, S),
        np.linspace(S, 0, r),
    ])
    if len(env) < n:
        env = np.concatenate([env, np.zeros(n - len(env))])
    return env[:n]


def voice_chord(
    freqs: list[float],
    dur_s: float,
    sr: int = SR,
    root_amp: float = 0.18,
    attack: float = 0.4,
    release: float = 1.0,
    shimmer: float = 0.35,
    n_partials: int = 8,
    detune_cents: tuple[float, ...] = (0.0, 5.5, -5.5),
    vibrato_hz: float = 5.5,
    vibrato_cents: float = 3.0,
    pan_spread: float = 0.6,
    portamento_from: list[float] | None = None,
    portamento_s: float = 0.45,
) -> np.ndarray:
    """Render a polyphonic pad. Returns float (n, 2) stereo."""
    n = int(dur_s * sr)
    t = np.arange(n) / sr
    env = adsr(n, sr, A=attack, D=0.5, S=0.85, R=release)

    out = np.zeros((n, 2), dtype=np.float64)
    nfreq = max(len(freqs), 1)
    pans = np.linspace(-pan_spread, pan_spread, nfreq) if nfreq > 1 else np.array([0.0])

    for vi, (f_to, pan) in enumerate(zip(freqs, pans)):
        # Portamento: log-interpolate from previous chord's matching voice
        if portamento_from is not None and vi < len(portamento_from):
            f_from = portamento_from[vi]
            np_porta = min(int(portamento_s * sr), n)
            f_inst = np.empty(n)
            if np_porta > 0:
                f_inst[:np_porta] = np.exp(
                    np.linspace(math.log(f_from), math.log(f_to), np_porta)
                )
                f_inst[np_porta:] = f_to
            else:
                f_inst[:] = f_to
        else:
            f_inst = np.full(n, f_to)

        # Vibrato in cents
        phi0 = np.random.uniform(0, 2 * np.pi)
        vib = vibrato_cents * np.sin(2 * np.pi * vibrato_hz * t + phi0)
        f_vib = f_inst * (2.0 ** (vib / 1200.0))

        voice_sig = np.zeros(n)
        for d_cents in detune_cents:
            f_det = f_vib * (2.0 ** (d_cents / 1200.0))
            phase = 2 * np.pi * np.cumsum(f_det) / sr + np.random.uniform(0, 2 * np.pi)
            for h in range(1, n_partials + 1):
                if (f_to * h) > 0.45 * sr:
                    continue
                amp = 1.0 / (h ** 1.3)
                voice_sig += amp * np.sin(h * phase) / len(detune_cents)

        # Shimmer (octave + twelfth, single sine, low amplitude)
        if shimmer > 0:
            for h, sh_amp in ((2, shimmer * 0.7), (3, shimmer * 0.3)):
                if (f_to * h) > 0.45 * sr:
                    continue
                ph = 2 * np.pi * np.cumsum(f_vib * h) / sr + np.random.uniform(0, 2 * np.pi)
                voice_sig += sh_amp * np.sin(ph) * 0.15

        voice_sig *= env

        # Equal-power pan
        a = (pan + 1.0) * math.pi / 4.0
        L, R = math.cos(a), math.sin(a)
        out[:, 0] += voice_sig * L
        out[:, 1] += voice_sig * R

    # Gentle low-pass to take the edge off high partials
    sos = butter(2, 5500, btype="low", fs=sr, output="sos")
    out[:, 0] = sosfilt(sos, out[:, 0])
    out[:, 1] = sosfilt(sos, out[:, 1])

    return out * root_amp


def render_chord_sequence(
    chord_list: list[list[float]],
    total_dur_s: float,
    root_hz: float = 130.81,
    release_overlap_s: float = 1.0,
    sr: int = SR,
    **voice_kwargs,
) -> np.ndarray:
    """Equal-duration chords with portamento glide between them and overlapping tails."""
    if not chord_list:
        return np.zeros((int(total_dur_s * sr), 2))
    n_chord = len(chord_list)
    dur_each = total_dur_s / n_chord
    total_n = int(total_dur_s * sr)
    tail = int(release_overlap_s * sr) + 4
    out = np.zeros((total_n + tail, 2))

    prev_freqs: list[float] | None = None
    for i, ratios in enumerate(chord_list):
        freqs = [root_hz * r for r in ratios]
        seg_dur = dur_each + release_overlap_s
        # Pad the previous-freq vector if this chord has more voices
        porta_from = prev_freqs
        if porta_from is not None and len(porta_from) < len(freqs):
            last = porta_from[-1]
            porta_from = porta_from + [last] * (len(freqs) - len(porta_from))
        seg = voice_chord(
            freqs, seg_dur, sr=sr,
            portamento_from=porta_from,
            portamento_s=0.45,
            **voice_kwargs,
        )
        start = int(i * dur_each * sr)
        end = start + seg.shape[0]
        end_clip = min(end, out.shape[0])
        out[start:end_clip] += seg[: end_clip - start]
        prev_freqs = freqs

    # Return the FULL buffer including the last chord's release tail (which
    # fades to zero at the buffer end). main() overlap-adds it so each scene's
    # tail crossfades into the next; truncating to total_n here would chop the
    # final release mid-amplitude → an audible jump at every scene boundary.
    return out


def chladni_morph_filter(stereo: np.ndarray, sr: int = SR,
                         lo_hz: float = 180.0, hi_hz: float = 900.0,
                         sweep_period_s: float = 8.0,
                         depth: float = 0.5) -> np.ndarray:
    """Layer a slow resonant band-pass sweep — the audio counterpart of the
    plate-geometry morph.

    Implemented block-wise with continuous filter state (sosfilt zi) carried
    between blocks. Without state propagation each new band-pass starts cold,
    which produces an audible click at every block boundary (~23 ms). Small
    blocks (128 samples) keep the cutoff trajectory smooth as well.
    """
    n = stereo.shape[0]
    t = np.arange(n) / sr
    sweep = 0.5 * (1.0 + np.sin(2 * np.pi * t / sweep_period_s - np.pi / 2))
    f_peak = lo_hz * (hi_hz / lo_hz) ** sweep
    out = stereo.copy()

    block = 128
    # Per-channel filter state, lazily initialised to the steady-state
    # response for the very first sample so the band-pass doesn't ring up
    # from zero.
    zi: list[np.ndarray | None] = [None, None]
    last_band: list[float] = [stereo[0, 0], stereo[0, 1]]

    for i in range(0, n, block):
        j = min(i + block, n)
        fc = float(f_peak[(i + j) // 2])
        bw = 0.30
        lo = max(fc * (1.0 - bw), 40.0)
        hi = min(fc * (1.0 + bw), sr * 0.49)
        if hi <= lo:
            continue
        sos = butter(2, [lo, hi], btype="bandpass", fs=sr, output="sos")
        for c in (0, 1):
            if zi[c] is None:
                zi[c] = sosfilt_zi(sos) * last_band[c]
            y, zi[c] = sosfilt(sos, stereo[i:j, c], zi=zi[c])
            out[i:j, c] += depth * y
            last_band[c] = float(y[-1])
    return out


# ----------------------------------------------------------------------
# Scene renderers
# ----------------------------------------------------------------------
def render_title(dur_s: float) -> np.ndarray:
    # Open with a sparse perfect-fifth + major-third pad, low register,
    # long attack, lots of shimmer in the upper partials.
    # Ratios normalised so the bottom voice == root_hz (A2 = 110 Hz).
    return voice_chord(
        [110.0 * r for r in _norm([2, 3, 5])], dur_s,
        attack=0.9, release=0.6, root_amp=0.16, shimmer=0.55,
    )


def render_outro(dur_s: float) -> np.ndarray:
    # Resolve on a major triad — bottom voice at C3.
    return voice_chord(
        [130.81 * r for r in _norm([4, 5, 6])], dur_s,
        attack=0.4, release=1.6, root_amp=0.16, shimmer=0.6,
    )


# ----------------------------------------------------------------------
# Master assembly
# ----------------------------------------------------------------------
def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(7)
    chords = load_chord_lists()

    pieces: dict[str, np.ndarray] = {}
    pieces["title"] = render_title(SCENE_FRAMES["title"] / FPS)

    pieces["harmonograph"] = render_chord_sequence(
        chords["harmonograph"], SCENE_FRAMES["harmonograph"] / FPS,
        root_hz=130.81,  # C3
        attack=0.6, release=1.4, shimmer=0.35,
    )

    pieces["knots"] = render_chord_sequence(
        chords["knots"], SCENE_FRAMES["knots"] / FPS,
        root_hz=146.83,  # D3
        attack=0.25, release=1.0, shimmer=0.55,
    )

    chladni_pad = render_chord_sequence(
        chords["chladni"], SCENE_FRAMES["chladni"] / FPS,
        root_hz=164.81,  # E3
        attack=0.15, release=1.2, shimmer=0.65,
    )
    pieces["chladni"] = chladni_morph_filter(chladni_pad)

    pieces["trees"] = render_chord_sequence(
        chords["trees"], SCENE_FRAMES["trees"] / FPS,
        root_hz=130.81,
        attack=0.35, release=1.2, shimmer=0.45,
    )

    pieces["clouds"] = render_chord_sequence(
        chords["clouds"], SCENE_FRAMES["clouds"] / FPS,
        root_hz=110.0,  # A2
        attack=0.7, release=1.6, shimmer=0.6,
    )

    pieces["outro"] = render_outro(SCENE_FRAMES["outro"] / FPS)

    # Lay scenes end-to-end, with each scene's tail spilling into the next
    total_frames = sum(SCENE_FRAMES[s] for s in ORDER)
    n_total = total_frames * SR // FPS
    tail_extra = SR  # 1 s of headroom for trailing release
    master = np.zeros((n_total + tail_extra, 2))
    cursor = 0
    for s in ORDER:
        seg = pieces[s]
        nseg = SCENE_FRAMES[s] * SR // FPS
        end = min(cursor + seg.shape[0], master.shape[0])
        master[cursor:end] += seg[: end - cursor]
        cursor += nseg

    master = master[:n_total]

    # Normalise to -1 dB peak
    peak = float(np.max(np.abs(master)))
    if peak > 1e-6:
        master *= (10 ** (-1.0 / 20.0)) / peak

    # Soft 50 ms head fade-in and 250 ms tail fade-out
    head = int(0.05 * SR)
    tail = int(0.25 * SR)
    master[:head] *= np.linspace(0.0, 1.0, head)[:, None]
    master[-tail:] *= np.linspace(1.0, 0.0, tail)[:, None]

    arr16 = np.clip(master * 32767.0, -32768, 32767).astype("<i2")
    with wave_open(str(OUT), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(arr16.tobytes())
    print(f"Wrote {OUT}  ({n_total / SR:.2f} s, {OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
