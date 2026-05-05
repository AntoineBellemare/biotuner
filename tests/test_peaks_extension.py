import numpy as np
import pytest
from biotuner.peaks_extension import (
    EEG_harmonics_mult,
    EEG_harmonics_div,
    harmonic_fit,
    intermodulation_spectrum,
    multi_consonance,
)

def test_harmonic_multiplication():
    """
    Test the EEG_harmonics_mult function.
    """
    peaks = [10.0, 20.0, 30.0]
    n_harmonics = 3
    mult_harmonics = EEG_harmonics_mult(peaks, n_harmonics, n_oct_up=1)
    
    assert mult_harmonics.shape == (len(peaks), n_harmonics + 1), \
        "EEG_harmonics_mult returned incorrect dimensions"
    assert np.all(mult_harmonics > 0), \
        "EEG_harmonics_mult returned non-positive values"

def test_harmonic_division():
    """
    Test the EEG_harmonics_div function.
    """
    peaks = [10.0, 20.0, 30.0]
    n_harmonics = 3
    div_harmonics, div_harm_bound = EEG_harmonics_div(peaks, n_harmonics, n_oct_up=1, mode="div")
    
    assert div_harmonics.shape == (len(peaks), n_harmonics + 1), \
        "EEG_harmonics_div returned incorrect dimensions"
    assert np.all(div_harm_bound >= 1) and np.all(div_harm_bound <= 2), \
        "EEG_harmonics_div bounded harmonics are not within [1, 2]"

def test_harmonic_fit():
    """
    Test the harmonic_fit function.
    """
    peaks = [10.0, 20.0, 30.0]
    n_harmonics = 10
    bounds = 0.1
    harm_fit, harmonics_pos, most_common_harmonics, matching_positions = harmonic_fit(
        peaks, n_harm=n_harmonics, bounds=bounds, function="mult"
    )
    
    assert len(harm_fit) > 0, \
        "harmonic_fit returned no common harmonics"
    assert all(isinstance(h, float) for h in harm_fit), \
        "harmonic_fit returned non-float harmonics"

def test_multi_consonance():
    """
    Test the multi_consonance function.
    """
    cons_pairs = [[10.0, 20.0], [20.0, 30.0], [30.0, 10.0]]
    top_consonant_freqs = multi_consonance(cons_pairs, n_freqs=2)
    
    assert len(top_consonant_freqs) == 2, \
        "multi_consonance did not return the correct number of consonant frequencies"
    assert all(f in [10.0, 20.0, 30.0] for f in top_consonant_freqs), \
        "multi_consonance returned frequencies outside the original set"


# ============================================================ intermodulation_spectrum


class TestIntermodulationSpectrum:
    def test_order2_sum_diff_two_peaks(self):
        peaks, amps, _ = intermodulation_spectrum(
            [100.0, 150.0], amplitudes=[1.0, 1.0], max_order=2,
        )
        # Originals: 100, 150 ; sum: 250 ; diff (positive): 50
        assert sorted(peaks.tolist()) == [50.0, 100.0, 150.0, 250.0]

    def test_drop_originals(self):
        peaks, _, _ = intermodulation_spectrum(
            [100.0, 150.0], amplitudes=[1.0, 1.0],
            max_order=2, drop_originals=True,
        )
        assert sorted(peaks.tolist()) == [50.0, 250.0]

    def test_sum_only(self):
        peaks, _, _ = intermodulation_spectrum(
            [100.0, 150.0], max_order=2,
            sum_diff=("sum",), drop_originals=True,
        )
        assert sorted(peaks.tolist()) == [250.0]

    def test_diff_only(self):
        peaks, _, _ = intermodulation_spectrum(
            [100.0, 150.0], max_order=2,
            sum_diff=("diff",), drop_originals=True,
        )
        assert sorted(peaks.tolist()) == [50.0]

    def test_order3_includes_2f1_pm_f2(self):
        peaks, _, _ = intermodulation_spectrum(
            [100.0, 150.0], max_order=3, drop_originals=True,
        )
        ps = sorted(peaks.tolist())
        # Expected: 50 (1*100-1*150 → -50, dropped), 250 (1+1 sum),
        # 50 (2*100-1*150), 200 (1*150*2-1*100), 350 (2*100+1*150),
        # 400 (1*100+2*150), 100 (2*150-1*100*2 = 100; 2*150-2*100=100 also via order-4),
        # ... a lot. Just assert that 350 (2*100+150) and 400 (100+2*150) appear.
        assert any(abs(p - 350.0) < 1e-9 for p in ps)
        assert any(abs(p - 400.0) < 1e-9 for p in ps)

    def test_dedupe_merges_amplitudes(self):
        # 100 and 200 have IMD: 200-100=100 (same as the original 100).
        # With dedupe=True, the original 100 and the IMD-derived 100 merge.
        # With dedupe=False, both appear.
        peaks_d, amps_d, sources_d = intermodulation_spectrum(
            [100.0, 200.0], amplitudes=[1.0, 1.0],
            max_order=2, dedupe=True,
        )
        peaks_nd, _, _ = intermodulation_spectrum(
            [100.0, 200.0], amplitudes=[1.0, 1.0],
            max_order=2, dedupe=False,
        )
        # Dedupe collapses by ~1 element; the merged source list should
        # contain MULTIPLE source records for the merged peak.
        assert len(peaks_nd) > len(peaks_d)
        # The 100 Hz peak should have two source records after dedupe
        # (original + 200-100 IMD).
        idx = int(np.argmin(np.abs(peaks_d - 100.0)))
        assert len(sources_d[idx]) >= 2

    def test_amplitude_law(self):
        # IMD amplitude = a_i^m * a_j^n. For order 2 (m=n=1), each
        # ordered pair (i, j) and (j, i) contributes a_i*a_j = 0.5*0.4
        # = 0.2 at the same frequency 250 Hz. With dedupe=True (default)
        # they merge and amplitudes sum to 0.4. With dedupe=False each
        # appears separately.
        peaks_nd, amps_nd, _ = intermodulation_spectrum(
            [100.0, 150.0], amplitudes=[0.5, 0.4],
            max_order=2, drop_originals=True, sum_diff=("sum",),
            dedupe=False,
        )
        # Two entries, both at 250 Hz with amp 0.2.
        assert peaks_nd.shape == (2,)
        for p in peaks_nd.tolist():
            assert p == pytest.approx(250.0)
        for a in amps_nd.tolist():
            assert a == pytest.approx(0.2)
        # With dedupe (default), one entry with summed amplitude 0.4.
        peaks_d, amps_d, _ = intermodulation_spectrum(
            [100.0, 150.0], amplitudes=[0.5, 0.4],
            max_order=2, drop_originals=True, sum_diff=("sum",),
        )
        assert peaks_d.shape == (1,)
        assert amps_d[0] == pytest.approx(0.4)

    def test_min_freq_filter(self):
        # min_freq=120 should drop the 50 Hz diff product.
        peaks, _, _ = intermodulation_spectrum(
            [100.0, 150.0], max_order=2,
            min_freq=120.0, drop_originals=True,
        )
        assert all(p >= 120.0 for p in peaks.tolist())

    def test_negative_dropped_by_default(self):
        # f_i - f_j with f_i < f_j gives negative; should be dropped.
        peaks, _, _ = intermodulation_spectrum(
            [100.0, 150.0], max_order=2, drop_originals=True,
        )
        assert (peaks >= 0).all()

    def test_single_peak_no_imd(self):
        peaks, _, _ = intermodulation_spectrum(
            [100.0], max_order=2, drop_originals=True,
        )
        assert peaks.size == 0

    def test_empty_input(self):
        peaks, amps, sources = intermodulation_spectrum([], max_order=2)
        assert peaks.size == 0
        assert amps.size == 0
        assert sources == []

    def test_invalid_max_order(self):
        with pytest.raises(ValueError):
            intermodulation_spectrum([100, 150], max_order=1)

    def test_invalid_sum_diff(self):
        with pytest.raises(ValueError):
            intermodulation_spectrum(
                [100, 150], max_order=2, sum_diff=("bogus",),
            )

    def test_amplitudes_length_mismatch(self):
        with pytest.raises(ValueError):
            intermodulation_spectrum(
                [100, 150], amplitudes=[1.0], max_order=2,
            )

    def test_composability_with_harmonic_geometry(self):
        """Smoke test: feed IMD output into a harmonic_geometry function."""
        from biotuner.harmonic_geometry import (
            HarmonicInput,
            quasicrystal_field_2d,
        )
        peaks_in = [100.0, 125.0, 150.0]  # Major chord (4 : 5 : 6)
        peaks_imd, amps_imd, _ = intermodulation_spectrum(
            peaks_in, max_order=2,
        )
        rich = HarmonicInput(
            peaks=peaks_imd.tolist(), amplitudes=amps_imd.tolist(),
        )
        g = quasicrystal_field_2d(rich, n_fold=7, resolution=64)
        assert g.coordinates.shape == (64, 64)

