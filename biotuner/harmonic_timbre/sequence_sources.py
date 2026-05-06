"""biotuner.harmonic_timbre.sequence_sources — TimbreSequence constructors.

Module type: Functions

A ``TimbreSequence`` is a time-resolved chain of Timbres. Phase 1 shipped
the dataclass + ``synthesize`` with cross-fading; this module ships the
*constructors* — ways to build a sequence from time-evolving biosignal
data or from generative models.

    timbre_sequence_from_ratio_frames(frames, ...)
        Most general: takes a list of ratio lists; produces one Timbre
        per frame and bundles them into a TimbreSequence.

    timbre_sequence_from_signal(signal, sf, ...)
        Chunks a 1D biosignal, runs the Biotuner pipeline per chunk,
        and builds a TimbreSequence from the per-chunk peaks_ratios.
        The "biosignal over time → time-varying instrument" entry point.

    timbre_sequence_from_transitional_harmony(th, ...)
        Convenience wrapper around ``timbre_sequence_from_signal`` for a
        fitted ``transitional_harmony`` instance — uses its ``data`` and
        ``sf`` attributes plus matching window / overlap conventions.

    timbre_sequence_from_markov_walk(transition_matrix, state_ratios, ...)
        Walks a discrete Markov chain for ``n_steps``, decoding each
        visited state to a ratio set via the ``state_ratios`` mapping.
        Pairs naturally with :class:`biotuner.harmonic_sequence.HarmonicMarkov`,
        but takes the matrix + state→ratios mapping as plain inputs so it
        also works with hand-authored chains.
"""

from __future__ import annotations

import logging
from typing import Iterable, Mapping, Sequence

import numpy as np

from biotuner.harmonic_timbre.matching import match_timbre
from biotuner.harmonic_timbre.timbre import Timbre, TimbreSequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. From an explicit list of ratio frames
# ---------------------------------------------------------------------------

def timbre_sequence_from_ratio_frames(
    frames: Sequence[Sequence[float]],
    *,
    matching_method: str = "consonance_weighted",
    base_freq: float = 220.0,
    times: np.ndarray | None = None,
    **matching_kwargs,
) -> TimbreSequence:
    """Build a TimbreSequence with one Timbre per ratio-list frame.

    Parameters
    ----------
    frames : sequence of (sequence of float)
        Each entry is a ratio list; each becomes one frame of the sequence.
    matching_method : str, default='consonance_weighted'
        Forwarded to :func:`~biotuner.harmonic_timbre.match_timbre`.
    base_freq : float, default=220.0
    times : array, optional
        Per-frame timestamps. If None, frames are uniformly spaced.
    **matching_kwargs
        Forwarded to the matching function.

    Returns
    -------
    TimbreSequence
    """
    frame_list = [list(f) for f in frames if len(f) > 0]
    if not frame_list:
        raise ValueError("timbre_sequence_from_ratio_frames: no non-empty frames")

    timbres: list[Timbre] = []
    for ratios in frame_list:
        t = match_timbre(
            ratios, method=matching_method,
            base_freq=base_freq, **matching_kwargs,
        )
        t.metadata.setdefault("scale_source", "ratio_frame")
        timbres.append(t)

    return TimbreSequence(frames=timbres, times=times)


# ---------------------------------------------------------------------------
# 2. From a raw biosignal — chunks → per-chunk Biotuner → ratios
# ---------------------------------------------------------------------------

def timbre_sequence_from_signal(
    signal,
    sf: float,
    *,
    window_size: float = 2.0,
    overlap: float = 0.5,
    matching_method: str = "consonance_weighted",
    base_freq: float = 220.0,
    peaks_function: str = "FOOOF",
    n_peaks: int = 5,
    min_freq: float = 1.0,
    max_freq: float = 60.0,
    precision: float = 0.1,
    **matching_kwargs,
) -> TimbreSequence:
    """Chunk a 1D biosignal, run Biotuner per chunk, build a TimbreSequence.

    Each window produces a fitted ``compute_biotuner`` whose
    ``peaks_ratios_cons`` becomes one frame of the sequence. Frames where
    peak extraction failed are skipped silently (logged at WARNING).

    Parameters
    ----------
    signal : array-like
        1D biosignal samples.
    sf : float
        Source samplerate (Hz).
    window_size : float, default=2.0
        Window length (seconds).
    overlap : float, default=0.5
        Fractional overlap between adjacent windows (0..<1).
    peaks_function, n_peaks, min_freq, max_freq, precision
        Forwarded to ``compute_biotuner.peaks_extraction``.
    matching_method, base_freq, matching_kwargs
        Forwarded to ``match_timbre``.

    Returns
    -------
    TimbreSequence
        With ``times`` set to per-window center times (seconds).
    """
    from biotuner import compute_biotuner

    sig = np.asarray(signal, dtype=np.float64).flatten()
    if sig.size < int(window_size * sf):
        raise ValueError(
            f"timbre_sequence_from_signal: signal too short "
            f"({sig.size} samples) for window_size={window_size}s @ sf={sf}"
        )
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0, 1)")

    win_n = int(round(window_size * sf))
    step_n = max(1, int(round(win_n * (1.0 - overlap))))

    frames: list[list[float]] = []
    times: list[float] = []

    start = 0
    while start + win_n <= sig.size:
        chunk = sig[start : start + win_n]
        try:
            bt = compute_biotuner(
                sf=int(sf), peaks_function=peaks_function, precision=precision,
            )
            bt.peaks_extraction(
                chunk, min_freq=min_freq, max_freq=max_freq,
                n_peaks=n_peaks,
            )
            ratios_attr = getattr(bt, "peaks_ratios_cons", None)
            if ratios_attr is None or len(ratios_attr) == 0:
                ratios_attr = getattr(bt, "peaks_ratios", None)
            ratios = (
                [float(r) for r in ratios_attr if r > 0]
                if ratios_attr is not None
                else []
            )
        except Exception as exc:
            logger.warning("biotuner pipeline failed on window @ %.2fs: %s",
                           start / sf, exc)
            ratios = []

        if ratios:
            frames.append(ratios)
            times.append((start + win_n / 2) / sf)

        start += step_n

    if not frames:
        raise ValueError(
            "timbre_sequence_from_signal: no windows produced ratios "
            "(check sf, peaks_function, signal content)"
        )

    return timbre_sequence_from_ratio_frames(
        frames,
        matching_method=matching_method,
        base_freq=base_freq,
        times=np.asarray(times),
        **matching_kwargs,
    )


# ---------------------------------------------------------------------------
# 3. From a fitted transitional_harmony instance
# ---------------------------------------------------------------------------

def timbre_sequence_from_transitional_harmony(
    th,
    *,
    matching_method: str = "consonance_weighted",
    base_freq: float = 220.0,
    **matching_kwargs,
) -> TimbreSequence:
    """Build a TimbreSequence from a fitted ``transitional_harmony`` instance.

    The fitted object only stores per-window subharmonic tensions, not
    the underlying ratios — so this helper re-runs Biotuner on the same
    data using the instance's window and overlap settings, then builds
    a sequence per chunk.

    Parameters
    ----------
    th : transitional_harmony
        A fitted instance of :class:`biotuner.transitional_harmony.transitional_harmony`.

    Returns
    -------
    TimbreSequence
    """
    if not hasattr(th, "data") or th.data is None:
        raise ValueError(
            "timbre_sequence_from_transitional_harmony: th.data not available"
        )
    sf = float(getattr(th, "sf"))
    overlap_pct = float(getattr(th, "overlap", 10) or 10)
    overlap = float(np.clip(overlap_pct / 100.0, 0.0, 0.95)) if overlap_pct > 1 else float(overlap_pct)

    return timbre_sequence_from_signal(
        th.data, sf,
        window_size=2.0,
        overlap=overlap,
        matching_method=matching_method,
        base_freq=base_freq,
        peaks_function=getattr(th, "peaks_function", "FOOOF"),
        n_peaks=int(getattr(th, "n_peaks", 5) or 5),
        min_freq=float(getattr(th, "min_freq", 2.0) or 2.0),
        max_freq=float(getattr(th, "max_freq", 80.0) or 80.0),
        precision=float(getattr(th, "precision", 0.1) or 0.1),
        **matching_kwargs,
    )


# ---------------------------------------------------------------------------
# 4. From a Markov chain walk over harmonic states
# ---------------------------------------------------------------------------

def timbre_sequence_from_markov_walk(
    transition_matrix,
    state_ratios: Mapping[int, Sequence[float]],
    *,
    n_steps: int = 16,
    start: int = 0,
    matching_method: str = "consonance_weighted",
    base_freq: float = 220.0,
    seed: int = 42,
    **matching_kwargs,
) -> TimbreSequence:
    """Walk a discrete Markov chain and build a TimbreSequence by decoding
    each visited state to a ratio set.

    Pairs with :class:`biotuner.harmonic_sequence.HarmonicMarkov` — its
    ``transition_matrix_`` property gives the matrix; the user supplies a
    ``state_ratios`` mapping (state index → ratio list) to decode states.
    Decoupled from any specific clustering implementation so it works
    with hand-authored chains too.

    Parameters
    ----------
    transition_matrix : array, shape (K, K)
        Row-stochastic transition matrix.
    state_ratios : mapping {int → sequence of float}
        Ratio list to use when the walk visits state ``k``.
    n_steps : int, default=16
        Length of the walk (number of frames in the sequence).
    start : int, default=0
        Starting state index.
    matching_method, base_freq, matching_kwargs
        Forwarded to ``match_timbre``.
    seed : int, default=42
        RNG seed for the stochastic walk.

    Returns
    -------
    TimbreSequence
    """
    P = np.asarray(transition_matrix, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("transition_matrix must be a square 2D array")
    K = P.shape[0]
    if not (0 <= start < K):
        raise ValueError(f"start state {start} out of range [0, {K})")
    if n_steps < 1:
        raise ValueError("n_steps must be ≥ 1")
    # Validate state_ratios coverage
    missing = [k for k in range(K) if k not in state_ratios]
    if missing and any(k == start for k in missing):
        raise ValueError(
            f"state_ratios is missing the starting state {start}"
        )

    rng = np.random.default_rng(seed)
    visited: list[int] = [start]
    cur = start
    for _ in range(n_steps - 1):
        row = P[cur]
        s = float(row.sum())
        if s <= 0:
            # Dead-end state: stay put
            visited.append(cur)
            continue
        probs = row / s
        cur = int(rng.choice(K, p=probs))
        visited.append(cur)

    # Decode each visited state to ratios; skip states without a mapping
    frames: list[list[float]] = []
    times: list[float] = []
    for i, state in enumerate(visited):
        ratios = state_ratios.get(state)
        if not ratios:
            logger.debug("Markov walk visited state %d with no ratios; skipping", state)
            continue
        frames.append([float(r) for r in ratios])
        times.append(float(i))

    if not frames:
        raise ValueError(
            "timbre_sequence_from_markov_walk: walk produced no decodable states"
        )

    seq = timbre_sequence_from_ratio_frames(
        frames,
        matching_method=matching_method,
        base_freq=base_freq,
        times=np.asarray(times),
        **matching_kwargs,
    )
    # tag the sequence with walk provenance
    seq.frames[0].metadata.setdefault("markov_walk", {
        "n_steps_requested": int(n_steps),
        "n_frames_decoded": len(frames),
        "start_state": int(start),
        "states_visited": visited,
    })
    return seq
