"""biotuner.resonance.nm_detect — sound cross-signal n:m phase-coupling detection.

The validated instrument from resonance_paper studies 38-43. Measures n:m phase coupling between two
signals the way the literature (Tass 1998) and our soundness battery establish as correct:

  * each component is band-passed in ITS OWN band and Hilbert-transformed; the ratio enters as INTEGER
    PHASE MULTIPLIERS ``n*phi_a - m*phi_b`` with the correct (Tass) convention (``n*f_a = m*f_b``),
    NOT as a filter band;
  * a PANEL of complementary techniques is reported, because no single one wins everywhere (Study 41):
    ``nm_plv`` is most sensitive on clean unimodal locks and at low SNR, while ``nm_rho_entropy`` and
    ``nm_phase_mi`` read the whole relative-phase distribution and recover multimodal / multistable
    locks where PLV anti-detects;
  * significance is a surrogate z / rank-p against an IAAFT-of-channel-B null (preserves B's spectrum
    and amplitude distribution, destroys the genuine A-B relationship);
  * SCOPE: this is sound only CROSS-signal (two independent sources). Within-signal or shared-source
    n:m at harmonically related frequencies is confounded with waveform shape (Aru 2015; Studies 42/43:
    every technique false-positives and no surrogate fully fixes it). The detector flags that case.

Example
-------
>>> from biotuner.resonance.nm_detect import detect_nm_coupling
>>> out = detect_nm_coupling(a, b, sf=500, freq_pairs=[(10.0, 15.0)])   # test a 2:3 lock
>>> out['results'][0]['metrics']['nm_rho_entropy']['z']
"""
from __future__ import annotations

import warnings
from fractions import Fraction

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert

from biotuner.resonance.coupling import nm_plv, nm_rho_entropy, nm_phase_mi, nm_conditional_prob
from biotuner.resonance.nulls import iaaft_surrogate

# default panel: sensitivity (plv) + all-moment generality (rho, mi)
PANEL = {
    "nm_plv": nm_plv,
    "nm_rho_entropy": nm_rho_entropy,
    "nm_phase_mi": nm_phase_mi,
    "nm_conditional_prob": nm_conditional_prob,
}
DEFAULT_PANEL = ("nm_plv", "nm_rho_entropy", "nm_phase_mi")


def _bandpass_hilbert_phase(x, sf, f, bandwidth):
    nyq = sf / 2.0
    lo = max(f - bandwidth / 2.0, 1e-6)
    hi = min(f + bandwidth / 2.0, nyq - 1e-6)
    sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    return np.angle(hilbert(sosfiltfilt(sos, np.asarray(x, dtype=np.float64))))


def nm_multipliers(f_a, f_b, max_denom=16):
    """Correct (Tass) integer multipliers (n, m) for frequencies f_a, f_b: n*f_a = m*f_b.

    ``n`` multiplies phi_a, ``m`` multiplies phi_b. For f_b/f_a = p/q (lowest terms) this returns
    (n=p, m=q) so that ``n*phi_a - m*phi_b`` is stationary for a genuine lock.
    """
    frac = Fraction(float(f_b) / float(f_a)).limit_denominator(max_denom)
    return frac.numerator, frac.denominator


def detect_nm_coupling(a, b, sf, freq_pairs, *, metrics=DEFAULT_PANEL, bandwidth=3.0,
                       max_denom=16, n_surrogates=99, seed=0, scope_guard=True):
    """Detect n:m phase coupling between signals ``a`` and ``b`` at the given frequency pairs.

    Parameters
    ----------
    a, b : 1-D arrays — the two signals (ideally INDEPENDENT sources; see scope note).
    sf : sampling frequency (Hz).
    freq_pairs : list of (f_a, f_b) — the component frequencies to test (e.g. (10, 15) for 2:3).
    metrics : panel of technique names (subset of PANEL). Default = plv + rho_entropy + phase_mi.
    bandwidth : Hz width of the band-pass around each frequency (>=3 keeps Hilbert phase stable).
    n_surrogates : number of IAAFT-of-b surrogates for the null.
    scope_guard : if True, warn when a and b are highly correlated (within-signal / shared source).

    Returns
    -------
    dict with ``results`` (one entry per freq pair: resolved n:m, and per-metric value / surrogate z /
    rank-p) and ``warning`` (the scope-guard message, or None).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    for mname in metrics:
        if mname not in PANEL:
            raise ValueError(f"unknown metric {mname!r}; choose from {sorted(PANEL)}")

    warning = None
    if scope_guard and a.shape == b.shape:
        r = float(np.abs(np.corrcoef(a, b)[0, 1]))
        if r > 0.9:
            warning = (
                f"channels are highly correlated (|r|={r:.2f}): within-signal / shared-source n:m at "
                "harmonically related frequencies is confounded with waveform shape (Aru 2015); the "
                "IAAFT surrogate z is NOT a valid null here. Treat any 'coupling' as a lower bound."
            )
            warnings.warn(warning, stacklevel=2)

    rng = np.random.default_rng(seed)
    results = []
    for (f_a, f_b) in freq_pairs:
        n, m = nm_multipliers(f_a, f_b, max_denom)
        pa = _bandpass_hilbert_phase(a, sf, f_a, bandwidth)
        pb = _bandpass_hilbert_phase(b, sf, f_b, bandwidth)
        # surrogate B phases computed ONCE per pair, reused across the panel
        seeds = rng.integers(0, 2 ** 31 - 1, n_surrogates)
        surr_pb = [_bandpass_hilbert_phase(iaaft_surrogate(b, np.random.default_rng(int(s))), sf, f_b, bandwidth)
                   for s in seeds]
        entry = {"f_a": float(f_a), "f_b": float(f_b), "n": int(n), "m": int(m),
                 "ratio": f"{n}:{m}", "metrics": {}}
        for mname in metrics:
            fn = PANEL[mname]
            obs = float(fn(pa, pb, n, m))
            sv = np.array([float(fn(pa, spb, n, m)) for spb in surr_pb])
            z = float((obs - sv.mean()) / (sv.std() + 1e-12))
            rank_p = float((1 + np.sum(sv >= obs)) / (len(sv) + 1))
            entry["metrics"][mname] = {"value": obs, "z": z, "rank_p": rank_p}
        results.append(entry)
    return {"results": results, "warning": warning}
