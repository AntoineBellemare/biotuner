"""Tests for the mathematical properties of the probability-weighted
harmonicity formulation (the post-May-2026 refactor of compute_global_harmonicity,
preserved by the resonance package).

Two key properties motivated the user-side refactor:

1. **Scale invariance.** Multiplying the input signal by a constant must not
   change the harmonicity spectrum. The probability normalization
   ``psd_prob = psd_clean / np.sum(psd_clean)`` ensures any global amplitude
   factor cancels (and the min-max pre-step ``(psd - psd_min)/(psd_max - psd_min)``
   also cancels k algebraically).

2. **Joint-presence weighting.** The reducer
   ``H[i] = p[i] * Σⱼ (S[i,j] * p[j])`` requires BOTH p[i] AND p[j] to be
   nonzero for a contribution. Adding power at a harmonic partner of an
   existing peak should multiply H[i] up; introducing power at a non-resonant
   frequency leaves H[i] essentially unchanged. This was the conceptual fix
   over the pre-2026 raw-PSD weighting.

These tests guard against any future regression of those properties.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _signals import SIGNALS  # noqa: E402

from biotuner.resonance import compute_resonance, ResonanceConfig  # noqa: E402


SF = 1000
CFG = ResonanceConfig(precision_hz=0.5, fmin=2, fmax=30)


# ---------------------------------------------------------------------------
# Property 1 — scale invariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [0.01, 0.1, 10.0, 100.0])
@pytest.mark.parametrize("signal_name", ["harmonic_5_10_20_40", "pink_noise"])
def test_scale_invariance(signal_name, scale):
    """Multiplying the input by a constant should not change H(f) within
    numerical precision (drift below 0.1% across 4 orders of magnitude)."""
    sig = SIGNALS[signal_name](sf=SF)
    H_ref = compute_resonance(sig, sf=SF, config=CFG).factors["H"]
    H_scaled = compute_resonance(sig * scale, sf=SF, config=CFG).factors["H"]

    # Relative tolerance: 1e-3 absorbs FOOOF-fit / float-subtraction noise
    # at extreme scales; tight enough to catch any real algorithmic violation.
    np.testing.assert_allclose(
        H_scaled, H_ref, rtol=1e-3, atol=1e-5,
        err_msg=f"Scale invariance violated for {signal_name} × {scale}: "
                 f"max |ΔH| = {np.max(np.abs(H_scaled - H_ref)):.3e}",
    )


# ---------------------------------------------------------------------------
# Property 2 — joint-presence weighting
# ---------------------------------------------------------------------------


def _make_tone_signal(freqs_hz, duration=8.0, noise=0.001, seed=0):
    """Build a sum of pure tones at the given frequencies (with very low noise)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(SF * duration)) / SF
    sig = sum(np.sin(2 * np.pi * f * t) for f in freqs_hz)
    sig += noise * rng.standard_normal(len(t))
    return sig


def _H_at_freq(result, f_target):
    idx = np.argmin(np.abs(result.freqs - f_target))
    return float(result.factors["H"][idx])


def test_adding_harmonic_partner_increases_H_at_both_freqs():
    """Joint-presence: when only 10 Hz has power, H[20 Hz] is low (no power
    at 20). Add a 20 Hz tone (1:2 harmonic ratio) and H at BOTH 10 Hz and
    20 Hz must increase substantially — each now has a power-carrying
    harmonic partner."""
    sig_A = _make_tone_signal([10.0])
    sig_B = _make_tone_signal([10.0, 20.0])
    r_A = compute_resonance(sig_A, sf=SF, config=CFG)
    r_B = compute_resonance(sig_B, sf=SF, config=CFG)

    H_A_10 = _H_at_freq(r_A, 10.0)
    H_A_20 = _H_at_freq(r_A, 20.0)
    H_B_10 = _H_at_freq(r_B, 10.0)
    H_B_20 = _H_at_freq(r_B, 20.0)

    # H at 10 Hz must rise (a harmonic partner appeared)
    assert H_B_10 > H_A_10 * 1.2, (
        f"H[10 Hz] did not rise enough: A={H_A_10:.3e}, B={H_B_10:.3e}"
    )
    # H at 20 Hz must rise dramatically (was tiny — only noise; now matches a
    # power-carrying harmonic partner)
    assert H_B_20 > H_A_20 * 5.0, (
        f"H[20 Hz] did not rise enough: A={H_A_20:.3e}, B={H_B_20:.3e}"
    )


def test_silent_freq_yields_low_H():
    """A frequency with no PSD power must have low H — even if the rest of
    the spectrum has harmonic content. Property: p[i] gates H[i]."""
    sig = _make_tone_signal([10.0, 20.0])  # only 10 and 20 Hz carry power
    r = compute_resonance(sig, sf=SF, config=CFG)
    H_at_powered = max(_H_at_freq(r, 10.0), _H_at_freq(r, 20.0))
    H_at_silent_25 = _H_at_freq(r, 25.0)  # no power here
    H_at_silent_28 = _H_at_freq(r, 28.0)

    # The silent freqs should have substantially lower H than the powered ones
    assert H_at_silent_25 < H_at_powered * 0.2, (
        f"H at silent 25 Hz too high: {H_at_silent_25:.3e} vs powered {H_at_powered:.3e}"
    )
    assert H_at_silent_28 < H_at_powered * 0.2, (
        f"H at silent 28 Hz too high: {H_at_silent_28:.3e} vs powered {H_at_powered:.3e}"
    )


def test_zero_psd_yields_zero_H():
    """Edge case: a frequency bin with literally zero probability mass must
    contribute zero H. This is the limit case of p[i] = 0 → H[i] = 0."""
    # Note: with minmax_prob normalization, the minimum bin gets set to exactly
    # zero after the (psd - psd_min)/(psd_max - psd_min) step. The H at that
    # bin should therefore be exactly zero (p[i] = 0 gates the whole sum).
    sig = _make_tone_signal([10.0, 20.0])
    r = compute_resonance(sig, sf=SF, config=ResonanceConfig(
        precision_hz=0.5, fmin=2, fmax=30, return_intermediates=True,
    ))
    psd_prob = r.intermediates["psd_prob"]
    H = r.factors["H"]

    # Find the bin with the smallest psd_prob (will be 0 due to minmax)
    silent_idx = int(np.argmin(psd_prob))
    assert psd_prob[silent_idx] == 0.0, "expected minmax_prob to zero out the lowest bin"
    # H at that bin should be exactly 0 because p[i] = 0 multiplies the whole sum
    # (only the gaussian_smooth_sigma blur might bleed a tiny amount from neighbors;
    # without smoothing it would be exactly 0)
    assert H[silent_idx] < 1e-3 * H.max(), (
        f"H at silent bin should be tiny: H[{silent_idx}]={H[silent_idx]:.3e}, "
        f"H_max={H.max():.3e}"
    )
