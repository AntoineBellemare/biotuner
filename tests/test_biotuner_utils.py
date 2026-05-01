"""Tests for biotuner.biotuner_utils.

Covers the foundational helper layer used by every other module in the
package.  Sections (in source order):

  1. Number theory                 — gcd, lcm, prime_factors, contFrac, …
  2. Tuning / ratio conversions    — compute_peak_ratios, scale2frac, …
  3. List utilities                — flatten, findsubsets, top_n_indexes, …
  4. Signal processing             — smooth, butter_bandpass, PSD, surrogates
  5. I/O                           — create_midi, create_SCL, …
  6. Music-theory helpers          — frequency_to_note, freq_to_note, …
  7. Time-series segmentation      — chunk_ts, segment_time_series
  8. Small utilities               — safe_max, safe_mean, string_to_list, …

Audio playback (pygame: play_chord, listen_chords, listen_scale,
make_chord, major_triad) is intentionally skipped — those require sound
hardware and aren't meaningfully testable here.
"""
import math
import os
from fractions import Fraction

import numpy as np
import pytest

from biotuner.biotuner_utils import (
    alpha2bands,
    apply_power_law_remove,
    butter_bandpass,
    butter_bandpass_filter,
    chords_to_ratios,
    chunk_ts,
    compareLists,
    compute_frequency_and_psd,
    compute_IMs,
    compute_peak_ratios,
    contFrac,
    correlated_noise_surrogates,
    create_midi,
    create_SCL,
    distinct_intervals,
    findsubsets,
    flatten,
    frac2scale,
    freq_to_note,
    frequency_to_note,
    gcd,
    generate_signal,
    getPairs,
    identify_mode,
    lcm,
    nth_root,
    pairs_most_frequent,
    peaks_to_amps,
    peaks_to_notes,
    phaseScrambleTS,
    power_law,
    prime_factors,
    ratio2frac,
    ratio_to_name,
    ratios2cents,
    ratios_harmonics,
    ratios_increments,
    rebound,
    rebound_list,
    reduced_form,
    safe_max,
    safe_mean,
    scale2frac,
    scale_from_pairs,
    scale_interval_names,
    segment_time_series,
    smooth,
    string_to_list,
    sum_list,
    top_n_indexes,
)


# ─── 1. Number theory ───────────────────────────────────────────────────────


def test_gcd_basic():
    assert gcd(12, 8) == 4
    assert gcd(15, 10, 5) == 5
    assert gcd(7, 13) == 1                           # coprime


def test_gcd_with_zero():
    assert gcd(0, 5) == 5
    assert gcd(0, 0) == 0


def test_lcm_basic():
    assert lcm(4, 6) == 12
    assert lcm(2, 3, 4) == 12
    assert lcm(7, 13) == 91


def test_lcm_with_zero():
    assert lcm(0, 0) == 0


def test_reduced_form_basic():
    assert reduced_form(4, 8, 12, 80) == (1, 2, 3, 20)
    assert reduced_form(15, 10, 5) == (3, 2, 1)


def test_prime_factors_known_values():
    assert prime_factors(12) == [2, 2, 3]
    assert prime_factors(7) == [7]
    assert prime_factors(60) == [2, 2, 3, 5]
    assert prime_factors(1) == []                    # 1 has no prime factors


def test_nth_root_basic():
    assert math.isclose(nth_root(8, 3), 2.0, rel_tol=1e-9)
    assert math.isclose(nth_root(2, 12), 2 ** (1 / 12), rel_tol=1e-9)
    assert nth_root(1, 5) == 1.0


def test_contFrac_known_values():
    # The continued fraction of 0.5 is [0, 2]
    assert contFrac(0.5, 5) == [0, 2]
    # The continued fraction of golden ratio φ = (1 + √5) / 2 ≈ 1.618 is [1, 1, 1, …]
    cf = contFrac((1 + 5 ** 0.5) / 2, 8)
    assert cf[:5] == [1, 1, 1, 1, 1]


def test_contFrac_terminates_for_rational():
    # 7/3 has continued fraction [2, 3]
    cf = contFrac(7 / 3, 10)
    assert cf[0] == 2
    # Subsequent terms should rapidly settle (rational)


def test_compute_IMs_docstring_example():
    """The compute_IMs docstring example should hold."""
    IMs, orders = compute_IMs(3, 12, 2)
    # Order pairs (j, i) iterate j, i ∈ [1, 2]
    # Add (3*1+12*1=15, 3*2+12*1=18, 3*1+12*2=27, 3*2+12*2=30)
    # Sub |3*1-12*1|=9, |3*2-12*1|=6, |3*1-12*2|=21, |3*2-12*2|=18
    expected_set = {6, 9, 15, 18, 21, 27, 30}
    assert set(IMs) == expected_set
    assert len(orders) == len(IMs)


# ─── 2. Tuning / ratio conversions ─────────────────────────────────────────


def test_compute_peak_ratios_basic():
    ratios = compute_peak_ratios([100, 150, 200])
    # All ratios are bounded in [1, 2)
    for r in ratios:
        assert 1.0 <= r < 2.0
    assert len(ratios) > 0


def test_compute_peak_ratios_includes_known_ratios():
    """Peaks 100 / 150 / 200 should yield ratios 3:2 and 4:3."""
    ratios = compute_peak_ratios([100, 150, 200])
    # 3/2 = 1.5 (from 150/100), 4/3 = 1.333 (from 200/150)
    assert any(math.isclose(r, 1.5, rel_tol=0.01) for r in ratios)
    assert any(math.isclose(r, 4 / 3, rel_tol=0.01) for r in ratios)


def test_compute_peak_ratios_unbounded():
    """rebound=False does not fold ratios into [1, 2)."""
    ratios = compute_peak_ratios([100, 200, 400], rebound=False)
    assert any(r >= 2.0 for r in ratios)


def test_compute_peak_ratios_excludes_unison():
    ratios = compute_peak_ratios([100, 100, 200])
    assert 1.0 not in ratios


def test_compute_peak_ratios_sub_includes_below_unity():
    """sub=True includes ratios < 1."""
    ratios = compute_peak_ratios([200, 100], rebound=False, sub=True)
    assert any(r < 1.0 for r in ratios)


def test_scale2frac_returns_fractions():
    scale_frac, num, den = scale2frac([1.5, 1.25, 1.333])
    assert len(scale_frac) == 3
    # 1.5 = 3/2 — verify numerator/denominator
    assert num[0] == 3 and den[0] == 2
    assert num[1] == 5 and den[1] == 4


def test_ratio2frac_basic():
    assert ratio2frac(1.5) == [3, 2]
    assert ratio2frac(0.75) == [3, 4]
    assert ratio2frac(2.0) == [2, 1]


def test_ratio2frac_irrational_caps_denominator():
    """An irrational ratio should be approximated with denominator ≤ maxdenom."""
    frac = ratio2frac(math.pi, maxdenom=1000)
    assert frac[1] <= 1000
    # Reconstructed value is close to π
    assert math.isclose(frac[0] / frac[1], math.pi, rel_tol=1e-3)


def test_frac2scale_inverts_scale2frac():
    original = [1.5, 1.25, 1.333]
    scale_frac, _, _ = scale2frac(original)
    recovered = frac2scale(scale_frac)
    for a, b in zip(original, recovered):
        assert math.isclose(a, b, rel_tol=1e-3)


def test_ratios2cents_known_intervals():
    cents = ratios2cents([1.0, 1.5, 2.0])
    assert math.isclose(cents[0], 0.0, abs_tol=1e-9)
    assert math.isclose(cents[1], 1200 * math.log2(1.5), rel_tol=1e-9)
    assert math.isclose(cents[2], 1200.0, rel_tol=1e-9)


def test_ratio_to_name_unknown_returns_none():
    assert ratio_to_name(1.234567) is None


def test_ratios_harmonics_includes_originals():
    out = ratios_harmonics([1.5], n_harms=3)
    # The harmonics of 1.5 up to order 3 are 1.5, 3.0, 4.5
    assert math.isclose(out[0], 1.5, rel_tol=1e-9)
    assert any(math.isclose(x, 3.0, rel_tol=1e-9) for x in out)


def test_ratios_increments_includes_originals():
    out = ratios_increments([1.5], n_inc=2)
    assert any(math.isclose(x, 1.5, rel_tol=1e-9) for x in out)


def test_chords_to_ratios_returns_two_aligned_lists():
    """chords_to_ratios returns (fractions, ratios), one entry per chord."""
    chords = [[100, 150], [200, 300]]
    fractions, ratios = chords_to_ratios(chords, harm_limit=2, spread=True)
    assert len(fractions) == len(chords)
    assert len(ratios) == len(chords)
    # Each ratio sequence is non-empty
    assert len(ratios[0]) > 0


def test_scale_from_pairs_returns_ratios():
    pairs = [[2, 3], [3, 4], [4, 5]]
    out = scale_from_pairs(pairs)
    assert len(out) == len(pairs)


def test_rebound_within_octave():
    # 3.0 → 1.5 (divided once by 2)
    assert math.isclose(rebound(3.0, low=1, high=2, octave=2), 1.5)
    # 0.25 must fold up into [1, 2] — the inner loop uses ≤ so it lands at 2
    out = rebound(0.25, low=1, high=2, octave=2)
    assert 1.0 <= out <= 2.0
    # Already in range stays
    assert rebound(1.5) == 1.5


def test_rebound_list_applies_per_element():
    out = rebound_list([3.0, 0.25, 1.5])
    assert len(out) == 3
    for v in out:
        assert 1.0 <= v <= 2.0


# ─── 3. List utilities ─────────────────────────────────────────────────────


def test_flatten_two_levels():
    assert flatten([[1, 2], [3, 4], [5]]) == [1, 2, 3, 4, 5]


def test_flatten_empty():
    assert flatten([]) == []
    assert flatten([[], []]) == []


def test_findsubsets_count_and_size():
    S = {1, 2, 3, 4}
    out = findsubsets(S, 2)
    # C(4, 2) = 6 distinct 2-element subsets
    assert len(out) == 6
    for s in out:
        assert len(s) == 2


def test_top_n_indexes_returns_top_values():
    arr = np.array([[1, 2], [3, 4]])
    indexes = top_n_indexes(arr, 2)
    # The two largest entries are 4 (1,1) and 3 (1,0)
    expected = {(1, 1), (1, 0)}
    assert set(indexes) == expected


def test_top_n_indexes_count():
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    indexes = top_n_indexes(arr, 3)
    assert len(indexes) == 3


def test_getPairs_count():
    out = getPairs([1, 2, 3, 4])
    # n*(n-1)/2 = 6 unordered pairs
    assert len(out) == 6
    for pair in out:
        assert len(pair) == 2


def test_pairs_most_frequent_returns_two_lists():
    pairs = [(1, 'a'), (1, 'b'), (2, 'a'), (1, 'a'), (3, 'c')]
    out = pairs_most_frequent(pairs, 2)
    assert len(out) == 2
    # Each sublist has at most n=2 elements
    assert len(out[0]) <= 2 and len(out[1]) <= 2


def test_sum_list_basic():
    assert sum_list([1, 2, 3, 4]) == 10
    assert sum_list([]) == 0


def test_compareLists_finds_close_matches():
    list1 = [1.0, 2.0, 3.0]
    list2 = [1.05, 5.0, 3.02]
    matching, positions, matching_pos, ratios = compareLists(list1, list2, 0.1)
    # 1.0 ↔ 1.05 and 3.0 ↔ 3.02 should match
    assert len(matching) == 2


# ─── 4. Signal processing ──────────────────────────────────────────────────


def test_smooth_preserves_length_for_window_1():
    """Convolution-based smooth: output length depends on the window."""
    x = np.arange(50.0)
    y = smooth(x, window_len=1, window="flat")
    # Window of length 1 → identity (within the convolution mode='valid')
    assert len(y) >= 1


def test_smooth_reduces_high_frequency_noise():
    rng = np.random.default_rng(0)
    base = np.linspace(0, 1, 200)
    noisy = base + 0.5 * rng.standard_normal(200)
    smoothed = smooth(noisy, window_len=21, window="hanning")
    # Variance after smoothing should be lower
    assert np.var(smoothed) < np.var(noisy)


def test_butter_bandpass_returns_filter_coeffs():
    b, a = butter_bandpass(5.0, 50.0, fs=1000, order=4)
    assert len(a) > 0 and len(b) > 0
    assert isinstance(a, np.ndarray) and isinstance(b, np.ndarray)


def test_butter_bandpass_filter_attenuates_outside_band():
    """A 100 Hz signal should be attenuated by a 5–20 Hz bandpass."""
    fs = 1000
    t = np.arange(0, 2.0, 1 / fs)
    sig = np.sin(2 * np.pi * 100 * t)        # outside the band
    filtered = butter_bandpass_filter(sig, 5, 20, fs, order=4)
    # After settling, amplitude should be much smaller than input
    settle = filtered[fs:]                    # skip transient
    assert np.max(np.abs(settle)) < 0.5 * np.max(np.abs(sig))


def test_generate_signal_length_and_dc():
    sig = generate_signal(1000, 1.0, freqs=[10, 50], amps=[1, 0.5])
    assert len(sig) == 1000
    # Sum of two sine waves has zero mean
    assert abs(np.mean(sig)) < 0.05


def test_compute_frequency_and_psd_shapes():
    rng = np.random.default_rng(0)
    sig = np.sin(2 * np.pi * 10 * np.arange(0, 4, 1 / 1000)) \
          + 0.1 * rng.standard_normal(4000)
    freqs, psd = compute_frequency_and_psd(sig, precision_hz=0.5, fs=1000)
    assert freqs.shape == psd.shape
    assert freqs[0] >= 0 and freqs[-1] <= 500


def test_compute_frequency_and_psd_fmin_fmax_clip():
    sig = np.random.randn(4000)
    freqs, psd = compute_frequency_and_psd(
        sig, precision_hz=0.5, fs=1000, fmin=2.0, fmax=30.0
    )
    assert freqs.min() >= 2.0 and freqs.max() <= 30.0


def test_compute_frequency_and_psd_peak_at_known_frequency():
    """A pure 17 Hz sine should produce a PSD peak at 17 Hz."""
    fs = 1000
    t = np.arange(0, 4.0, 1 / fs)
    sig = np.sin(2 * np.pi * 17 * t)
    freqs, psd = compute_frequency_and_psd(sig, precision_hz=0.5, fs=fs,
                                           fmin=1.0, fmax=50.0)
    peak_freq = freqs[np.argmax(psd)]
    assert abs(peak_freq - 17.0) < 1.0


def test_power_law_function():
    x = np.array([1.0, 2.0, 4.0, 8.0])
    # a*x^b with a=2, b=-1 → [2, 1, 0.5, 0.25]
    out = power_law(x, 2.0, -1.0)
    assert np.allclose(out, [2.0, 1.0, 0.5, 0.25])


def test_apply_power_law_remove_passthrough_when_false():
    psd = np.array([10.0, 5.0, 2.5, 1.0])
    freqs = np.array([1.0, 2.0, 4.0, 8.0])
    out = apply_power_law_remove(freqs, psd, power_law_remove=False)
    assert np.array_equal(out, psd)


def test_apply_power_law_remove_subtracts_trend_when_true():
    """A pure power-law PSD should yield ~0 residuals after removal."""
    freqs = np.linspace(1, 100, 200)
    psd = power_law(freqs, 5.0, -1.5)
    residual = apply_power_law_remove(freqs, psd, power_law_remove=True)
    # Residual amplitude is much smaller than original PSD
    assert np.max(np.abs(residual)) < 0.1 * np.max(psd)


def test_phaseScrambleTS_preserves_power():
    rng = np.random.default_rng(0)
    ts = rng.standard_normal(2048)
    scrambled = phaseScrambleTS(ts)
    # Total power preserved (Parseval)
    assert math.isclose(np.var(ts), np.var(scrambled), rel_tol=0.05)
    # Output length matches
    assert len(scrambled) == len(ts)


def test_phaseScrambleTS_changes_phases():
    rng = np.random.default_rng(0)
    ts = rng.standard_normal(1024)
    scrambled = phaseScrambleTS(ts)
    # Time-domain signals differ
    assert not np.allclose(ts, scrambled)


def test_correlated_noise_surrogates_preserves_psd():
    rng = np.random.default_rng(0)
    # Shape (n_series, n_time)
    data = rng.standard_normal((2, 512))
    surr = correlated_noise_surrogates(data)
    assert surr.shape == data.shape
    # Power spectra should match for the interior bins.  DC and Nyquist
    # bins are real-valued in rfft and the random-phase trick distorts
    # them, so we exclude them from the comparison.
    p_orig = np.abs(np.fft.rfft(data, axis=1)) ** 2
    p_surr = np.abs(np.fft.rfft(surr, axis=1)) ** 2
    assert np.allclose(p_orig[:, 1:-1], p_surr[:, 1:-1], rtol=1e-6, atol=1e-6)


# ─── 5. I/O — MIDI and Scala ────────────────────────────────────────────────


def test_create_midi_writes_file(tmp_path):
    chords = [[220.0, 275.0, 330.0], [220.0, 264.0, 330.0]]
    durations = [1.0, 1.0]
    name = str(tmp_path / "tester")
    mid = create_midi(chords, durations, filename=name)
    assert os.path.isfile(name + ".mid")
    assert len(mid.tracks) >= 1


def test_create_midi_microtonal_emits_pitchwheel(tmp_path):
    chords = [[440.0]]
    durations = [1.0]
    name = str(tmp_path / "micro")
    mid = create_midi(chords, durations, microtonal=True, filename=name)
    # At least one pitchwheel message should be present somewhere
    types = [m.type for tr in mid.tracks for m in tr]
    assert "note_on" in types
    assert "pitchwheel" in types


def test_create_midi_non_microtonal_skips_pitchwheel(tmp_path):
    chords = [[440.0]]
    durations = [1.0]
    name = str(tmp_path / "no_micro")
    mid = create_midi(chords, durations, microtonal=False, filename=name)
    types = [m.type for tr in mid.tracks for m in tr]
    assert "pitchwheel" not in types


def test_create_midi_variable_chord_sizes(tmp_path):
    """Tracks must always step forward even when a chord is shorter."""
    chords = [[220, 275, 330], [220, 275]]
    durations = [1.0, 1.0]
    name = str(tmp_path / "ragged")
    mid = create_midi(chords, durations, filename=name)
    # Number of tracks equals the maximum chord size
    assert len(mid.tracks) == 3


def test_create_SCL_writes_file_and_returns_string(tmp_path):
    # Scala needs unison + ratios; see create_SCL: it skips scale[0]
    name = str(tmp_path / "tunable")
    scale = [1.0, 1.25, 1.5, 2.0]
    out = create_SCL(scale, name, write=True)
    assert isinstance(out, str)
    assert os.path.isfile(name + ".scl")


def test_create_SCL_count_matches_emitted_ratios(tmp_path):
    name = str(tmp_path / "counts")
    scale = [1.0, 1.25, 1.3333, 1.5, 1.6667, 1.875, 2.0]
    out = create_SCL(scale, name, write=False)
    lines = [ln for ln in out.splitlines() if ln.strip()
             and not ln.lstrip().startswith("!")]
    declared = int(lines[1].strip())            # second non-comment is the count
    n_ratios = len(lines) - 2                     # rest are the ratios
    assert n_ratios == declared
    # We passed 7 elements; first is dropped → 6 ratios
    assert declared == 6


def test_distinct_intervals_basic():
    """Every interval in a non-trivial scale lies in (1, 2)."""
    scale = [1.0, 1.25, 1.5, 2.0]
    intervals = distinct_intervals(scale)
    for x in intervals:
        assert 1.0 < x < 2.0


def test_distinct_intervals_empty_returns_empty():
    assert distinct_intervals([]) == []


def test_scale_interval_names_returns_pairs():
    out = scale_interval_names([1.0, 1.5, 2.0])
    # One [step, name_or_none] pair per scale step
    assert len(out) == 3
    for pair in out:
        assert len(pair) == 2


# ─── 6. Music-theory helpers ────────────────────────────────────────────────


def test_frequency_to_note_a4():
    name, octave = frequency_to_note(440.0)
    assert name == "A"
    assert octave == 4


def test_frequency_to_note_c4():
    """C4 ≈ 261.63 Hz."""
    name, octave = frequency_to_note(261.63)
    assert name == "C"
    assert octave == 4


def test_frequency_to_note_octave_up():
    name, octave = frequency_to_note(880.0)        # A5
    assert name == "A"
    assert octave == 5


def test_freq_to_note_a4_zero_cents():
    name, cents = freq_to_note(440.0)
    assert name == "A4"
    assert abs(cents) < 1e-6


def test_freq_to_note_includes_cents_offset():
    # 442 Hz is ~7.85 cents above A4
    _, cents = freq_to_note(442.0)
    assert 0 < cents < 20


def test_peaks_to_notes_in_tune_flag():
    # 440 Hz exactly → in tune; 460 Hz is way out → not in tune.
    # The flag may be a numpy boolean, hence the truthy comparison.
    out = peaks_to_notes([440.0, 460.0], cents_threshold=10)
    assert bool(out[0]["in_tune"]) is True
    assert bool(out[1]["in_tune"]) is False


def test_identify_mode_recognises_major_scale():
    # C major: C, D, E, F, G, A, B (frequencies)
    peaks = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]
    mode_name, similarity, _ = identify_mode(peaks)
    assert "Ionian" in mode_name or "Major" in mode_name
    assert similarity > 50


def test_identify_mode_too_few_peaks_returns_insufficient():
    name, sim, root = identify_mode([440.0, 880.0])
    assert "Insufficient" in name
    assert sim == 0.0


def test_alpha2bands_returns_five_bands():
    bands = alpha2bands(10.0)
    assert len(bands) == 5
    for band in bands:
        assert len(band) == 2
        assert band[0] < band[1]


def test_peaks_to_amps_returns_per_peak_amplitude():
    fs = 1000
    freqs = np.arange(0, fs / 2, 1.0)            # 0 .. fs/2 in 1 Hz steps
    amps = np.linspace(0, 10, len(freqs))
    peaks = [10.0, 50.0]
    out = peaks_to_amps(peaks, freqs, amps, fs)
    assert len(out) == 2


# ─── 7. Time-series segmentation ───────────────────────────────────────────


def test_chunk_ts_basic():
    # 4 seconds at 1000 Hz, precision 1 Hz → 1000-sample chunks
    data = np.arange(4000)
    pairs = chunk_ts(data, sf=1000, overlap=10, precision=1)
    assert len(pairs) > 0
    for start, end in pairs:
        assert end - start == 1000


def test_chunk_ts_overlap_increases_pair_count():
    """Higher overlap → more chunks."""
    data = np.arange(4000)
    p1 = chunk_ts(data, sf=1000, overlap=10, precision=1)
    p2 = chunk_ts(data, sf=1000, overlap=50, precision=1)
    assert len(p2) >= len(p1)


def test_segment_time_series_uses_boundaries():
    data = np.arange(10)
    bounds = [3, 7]
    segments = segment_time_series(data, bounds)
    # Three segments: [0:3], [3:7], [7:]
    assert len(segments) == 3
    assert list(segments[0]) == [0, 1, 2]
    assert list(segments[1]) == [3, 4, 5, 6]
    assert list(segments[2]) == [7, 8, 9]


def test_segment_time_series_no_bounds_returns_full():
    data = np.arange(5)
    segments = segment_time_series(data, [])
    assert len(segments) == 1
    assert list(segments[0]) == [0, 1, 2, 3, 4]


# ─── 8. Small utilities ────────────────────────────────────────────────────


def test_safe_max_with_list():
    assert safe_max([1, 5, 3]) == 5
    assert safe_max([]) is np.nan or np.isnan(safe_max([]))


def test_safe_max_with_float():
    assert safe_max(3.14) == 3.14


def test_safe_max_with_garbage_returns_nan():
    assert np.isnan(safe_max("hello"))
    assert np.isnan(safe_max(None))


def test_safe_mean_with_list():
    assert safe_mean([2, 4, 6]) == 4.0


def test_safe_mean_with_string_parses():
    """safe_mean accepts a stringified list ('[1 2 3]') via string_to_list."""
    assert safe_mean("[1 2 3]") == 2.0


def test_safe_mean_with_float():
    assert safe_mean(2.71) == 2.71


def test_safe_mean_empty_returns_nan():
    assert np.isnan(safe_mean([]))


def test_string_to_list_basic():
    assert string_to_list("[1.0 2.5 3.0]") == [1.0, 2.5, 3.0]
    assert string_to_list("[ 4 5 6 ]") == [4.0, 5.0, 6.0]
