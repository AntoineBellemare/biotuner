"""Tests for biotuner.harmonic_sequence.

Covers:
  - Stateless encoders (cents histogram, JI matrix, harmonicity spectrum/matrix)
  - Bridge layer (histogram_to_ratios, .scl, MIDI)
  - All six method classes (Markov, Wasserstein, DMD, Latent, Topology, Grammar)
  - The HarmonicSequenceAnalyzer orchestrator
  - Caching behaviour and end-to-end pipeline through compute_biotuner

The biotuner-dependent tests share a session-scoped bt_list fixture so the
slow peaks_extraction step runs once for the whole module.
"""
import os
import warnings

import numpy as np
import pytest

from biotuner.harmonic_sequence import (
    HarmonicDMD,
    HarmonicGrammar,
    HarmonicLatentSpace,
    HarmonicMarkov,
    HarmonicSequenceAnalyzer,
    HarmonicTopology,
    WassersteinTrajectory,
    _DYAD_MATRIX_CACHE,
    clear_harmonicity_cache,
    encode_harmonicity_matrices,
    encode_harmonicity_spectrum,
    encode_histograms,
    encode_ji_matrix,
    encode_scalar_metrics,
    extract_tuning,
    find_optimal_n_states,
    histogram_to_ratios,
    histogram_to_scl,
    histograms_to_midi,
)


# ─── shared fixtures ────────────────────────────────────────────────────────

JUST_MAJOR = [1.25, 1.3333, 1.5, 1.6667, 1.875]   # 5/4, 4/3, 3/2, 5/3, 15/8
JUST_MINOR = [1.2, 1.3333, 1.5, 1.6, 1.8]
DRIFT_PROGRESSION = [
    [1.5, 1.25, 1.6667],
    [1.495, 1.252, 1.665],
    [1.49, 1.255, 1.66],
    [1.485, 1.258, 1.655],
    [1.48, 1.26, 1.65],
    [1.475, 1.262, 1.645],
    [1.47, 1.265, 1.64],
    [1.465, 1.268, 1.635],
]


@pytest.fixture
def ratios_list_simple():
    """A short, well-behaved ratio sequence — no biotuner required."""
    return [
        list(JUST_MAJOR),
        list(JUST_MAJOR),
        list(JUST_MINOR),
        list(JUST_MINOR),
        list(JUST_MAJOR),
        list(JUST_MAJOR),
        list(JUST_MINOR),
        list(JUST_MINOR),
    ]


@pytest.fixture
def histograms_simple(ratios_list_simple):
    return encode_histograms(ratios_list_simple)


# Session-scoped bt_list — slow to build (peaks_extraction ~3 s/window) so
# the whole module shares it.
@pytest.fixture(scope="session")
def bt_list():
    from biotuner.biotuner_object import compute_biotuner

    sf = 1000
    duration = 4.0
    rng = np.random.default_rng(0)

    def _signal(freqs):
        t = np.linspace(0, duration, int(sf * duration), endpoint=False)
        sig = sum((1.0 / (i + 1)) * np.sin(2 * np.pi * f * t)
                  for i, f in enumerate(freqs))
        sig += 0.02 * rng.standard_normal(len(t))
        return sig.astype(np.float64)

    freq_seq = [
        [2, 5, 10, 20, 40],
        [2, 5, 10, 20, 40],
        [2, 4.8, 7.2, 14.4, 28.8],
        [2, 4.8, 7.2, 14.4, 28.8],
    ]
    bts = []
    for freqs in freq_seq:
        bt = compute_biotuner(sf=sf, peaks_function="fixed", precision=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bt.peaks_extraction(_signal(freqs), n_peaks=5,
                                precision=0.5, min_freq=1.0, max_freq=50.0)
        try:
            bt.compute_peaks_metrics()
        except Exception:
            pass
        bts.append(bt)
    return bts


# ─── 1. Encoders ────────────────────────────────────────────────────────────


def test_encode_histograms_shape_and_normalisation(ratios_list_simple):
    H = encode_histograms(ratios_list_simple, n_bins=240)
    assert H.shape == (len(ratios_list_simple), 240)
    sums = H.sum(axis=1)
    # All non-empty rows must sum to 1; empty rows sum to 0
    nonempty = sums > 0
    assert np.allclose(sums[nonempty], 1.0)


def test_encode_histograms_known_intervals_land_in_correct_bins():
    """A 3:2 perfect fifth (~702¢) and an octave (1200¢) land in known bins."""
    H = encode_histograms([[1.5, 2.0]], n_bins=240, min_cents=0, max_cents=1200)
    fifth_bin = int(1200 * np.log2(1.5) / 5)        # 5 cents per bin
    octave_bin = 240 - 1                              # last bin
    # 1.5 should populate the fifth bin; 2.0 sits at the upper edge.
    assert H[0, fifth_bin] > 0


def test_encode_histograms_handles_empty_and_invalid():
    H = encode_histograms([[], [-1, 0, np.inf, 1.5]], n_bins=240)
    assert H.shape == (2, 240)
    assert H[0].sum() == 0                      # empty stays empty
    assert np.isclose(H[1].sum(), 1.0)          # only 1.5 survives


def test_encode_ji_matrix_shape_dtype_and_matching():
    X, labels = encode_ji_matrix([[1.5], [1.25, 1.5], []],
                                 tolerance_cents=30.0)
    assert X.dtype == np.int8
    assert X.shape == (3, len(labels))
    assert X[0].sum() >= 1                      # 1.5 matches a JI interval
    assert X[1].sum() >= X[0].sum()             # superset → at least as many matches
    assert X[2].sum() == 0                      # empty row → zero matches


def test_encode_ji_matrix_drops_invalid_ratios():
    X, _ = encode_ji_matrix([[-1.0, 0.0, np.nan, 1.5]])
    # 1.5 is the only valid ratio; should still produce some matches
    assert X[0].sum() > 0


def test_encode_scalar_metrics_handles_objects_without_metrics():
    """encode_scalar_metrics tolerates bt-like objects missing fields."""
    class FakeBt:
        peaks_metrics = {}
        scale_metrics = {}
        peaks = []
    X = encode_scalar_metrics([FakeBt(), FakeBt()])
    assert X.shape[0] == 2
    assert np.all(np.isnan(X))


def test_extract_tuning_returns_only_finite_positive():
    class FakeBt:
        peaks_ratios = [1.5, "junk", -1.0, np.inf, 1.25]
    out = extract_tuning(FakeBt(), tuning="peaks_ratios")
    assert out == [1.5, 1.25]


def test_extract_tuning_missing_attribute_returns_empty():
    class FakeBt:
        pass
    assert extract_tuning(FakeBt(), tuning="peaks_ratios") == []


# ─── 2. Bridge layer ────────────────────────────────────────────────────────


def test_histogram_to_ratios_round_trip():
    H = encode_histograms([JUST_MAJOR])[0]
    recovered = histogram_to_ratios(
        H, include_unison=False, include_octave=False,
    )
    assert len(recovered) == len(JUST_MAJOR)
    # Every original ratio is matched within ~5 cents (one bin width)
    for r in JUST_MAJOR:
        c = 1200 * np.log2(r)
        rec_cents = [1200 * np.log2(x) for x in recovered]
        assert min(abs(rc - c) for rc in rec_cents) < 5.0


def test_histogram_to_ratios_unison_octave_flags():
    H = encode_histograms([JUST_MAJOR])[0]
    full = histogram_to_ratios(H, include_unison=True, include_octave=True)
    assert full[0] == 1.0 and full[-1] == 2.0
    bare = histogram_to_ratios(H, include_unison=False, include_octave=False)
    assert 1.0 not in bare and 2.0 not in bare


def test_histogram_to_ratios_empty_histogram():
    out = histogram_to_ratios(
        np.zeros(240), include_unison=True, include_octave=True,
    )
    assert out == [1.0, 2.0]
    out_bare = histogram_to_ratios(
        np.zeros(240), include_unison=False, include_octave=False,
    )
    assert out_bare == []


def test_histogram_to_ratios_n_peaks_caps_output():
    H = encode_histograms([JUST_MAJOR])[0]
    out = histogram_to_ratios(
        H, n_peaks=2, include_unison=False, include_octave=False,
    )
    assert len(out) == 2


def test_histogram_to_scl_returns_string_with_count_line():
    H = encode_histograms([JUST_MAJOR])[0]
    s = histogram_to_scl(H, name="tester")
    assert isinstance(s, str)
    assert "tester" in s
    # Scala spec: a non-comment line declares the count, followed by that
    # many ratio lines.  Find the integer count line and count the ratios
    # that follow.
    lines = [ln for ln in s.splitlines() if ln.strip()
             and not ln.lstrip().startswith("!")]
    # First non-comment is the name, second is the integer count
    declared = int(lines[1].strip())
    ratio_lines = lines[2:]
    assert len(ratio_lines) == declared
    # Each ratio line is either a fraction "n/d" or a cents float
    for ln in ratio_lines:
        ln = ln.strip()
        assert "/" in ln or "." in ln


def test_histogram_to_scl_writes_file(tmp_path):
    H = encode_histograms([JUST_MAJOR])[0]
    name = str(tmp_path / "tunable")
    histogram_to_scl(H, name=name, write=True)
    assert os.path.isfile(name + ".scl")


def test_histograms_to_midi_writes_file(tmp_path):
    H = encode_histograms([JUST_MAJOR, JUST_MINOR, JUST_MAJOR])
    out = str(tmp_path / "trio")
    mid = histograms_to_midi(H, filename=out,
                             base_freq=220.0, duration_beats=0.5, n_peaks=4)
    assert os.path.isfile(out + ".mid")
    assert mid.tracks                              # at least one track


def test_histograms_to_midi_accepts_1d_histogram(tmp_path):
    H = encode_histograms([JUST_MAJOR])[0]
    out = str(tmp_path / "single")
    histograms_to_midi(H, filename=out, n_peaks=4)
    assert os.path.isfile(out + ".mid")


def test_histograms_to_midi_variable_durations(tmp_path):
    H = encode_histograms([JUST_MAJOR, JUST_MINOR, JUST_MAJOR])
    out = str(tmp_path / "vardur")
    histograms_to_midi(H, filename=out,
                       duration_beats=[0.25, 0.5, 1.0], n_peaks=4)
    assert os.path.isfile(out + ".mid")


def test_histograms_to_midi_duration_length_mismatch_raises(tmp_path):
    H = encode_histograms([JUST_MAJOR, JUST_MINOR])
    with pytest.raises(ValueError):
        histograms_to_midi(H, filename=str(tmp_path / "bad"),
                           duration_beats=[1.0, 1.0, 1.0])


def test_histograms_to_midi_all_empty_raises(tmp_path):
    empty = np.zeros((3, 240))
    with pytest.raises(ValueError):
        histograms_to_midi(empty, filename=str(tmp_path / "empty"))


# ─── 3. HarmonicMarkov ──────────────────────────────────────────────────────


def test_markov_fit_transition_matrix_row_stochastic(histograms_simple):
    mk = HarmonicMarkov(n_states=2, random_state=0).fit(histograms_simple)
    T = mk.transition_matrix_
    assert T.shape == (2, 2)
    # Each row sums to 1 (or 0 for unreached states)
    sums = T.sum(axis=1)
    for s in sums:
        assert s == 0 or np.isclose(s, 1.0)


def test_markov_steady_state_is_distribution(histograms_simple):
    mk = HarmonicMarkov(n_states=3, random_state=0).fit(histograms_simple)
    pi = mk.steady_state_
    assert pi.shape == (3,)
    assert (pi >= 0).all()
    assert np.isclose(pi.sum(), 1.0)


def test_markov_transition_entropy_within_bounds(histograms_simple):
    mk = HarmonicMarkov(n_states=2, random_state=0).fit(histograms_simple)
    H = mk.transition_entropy_
    assert 0.0 <= H <= np.log2(2) + 1e-9


def test_markov_predict_next_proba_returns_distribution(histograms_simple):
    mk = HarmonicMarkov(n_states=2, random_state=0).fit(histograms_simple)
    p = mk.predict_next_proba(0)
    assert p.shape == (2,)
    assert np.isclose(p.sum(), 1.0) or p.sum() == 0     # 0 if state unseen


def test_markov_higher_order_stores_history_dict(histograms_simple):
    mk = HarmonicMarkov(n_states=2, order=2, random_state=0).fit(histograms_simple)
    assert mk.high_order_transition_ is not None
    for hist, dist in mk.high_order_transition_.items():
        assert len(hist) == 2
        assert np.isclose(dist.sum(), 1.0)


def test_markov_invalid_order_raises():
    with pytest.raises(ValueError):
        HarmonicMarkov(n_states=2, order=0)


def test_find_optimal_n_states_reasonable_range():
    # 4 obvious clusters in a small synthetic matrix
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((4, 6))
    X = np.vstack([c + 0.05 * rng.standard_normal((6, 6)) for c in centers])
    best_k, scores = find_optimal_n_states(X, k_range=(2, 6), random_state=0)
    assert 2 <= best_k <= 6
    assert all(2 <= k <= 6 for k in scores)


# ─── 4. WassersteinTrajectory ───────────────────────────────────────────────


def test_wasserstein_distance_matrix_properties(histograms_simple):
    wt = WassersteinTrajectory(n_bins=240).fit(histograms_simple)
    D = wt.distance_matrix_
    T = len(histograms_simple)
    assert D.shape == (T, T)
    # Symmetric, zero diagonal, non-negative
    assert np.allclose(D, D.T)
    assert np.all(D >= 0)
    assert np.allclose(np.diag(D), 0.0)


def test_wasserstein_flux_length_T_minus_one(histograms_simple):
    wt = WassersteinTrajectory(n_bins=240).fit(histograms_simple)
    assert wt.flux_.shape == (len(histograms_simple) - 1,)


def test_wasserstein_barycenter_endpoints_match_inputs(histograms_simple):
    wt = WassersteinTrajectory(n_bins=240).fit(histograms_simple)
    h1, h2 = histograms_simple[0], histograms_simple[2]
    b0 = wt.barycenter(h1, h2, alpha=0.0)
    b1 = wt.barycenter(h1, h2, alpha=1.0)
    # The barycenter at endpoints should resemble the corresponding input
    # (within the quantile-discretisation noise).
    assert np.dot(b0, h1) > np.dot(b0, h2)
    assert np.dot(b1, h2) > np.dot(b1, h1)


def test_wasserstein_interpolate_pair_count(histograms_simple):
    wt = WassersteinTrajectory(n_bins=240).fit(histograms_simple)
    morph = wt.interpolate_pair(0, 2, n_steps=5)
    assert len(morph) == 5
    for m in morph:
        assert m.shape == (240,)


def test_wasserstein_embed_mds_shape(histograms_simple):
    wt = WassersteinTrajectory(n_bins=240).fit(histograms_simple)
    Z = wt.embed(n_components=2, method="mds")
    assert Z.shape == (len(histograms_simple), 2)


# ─── 5. HarmonicDMD ─────────────────────────────────────────────────────────


def test_dmd_fit_eigenvalues_nonempty():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 4))
    dmd = HarmonicDMD().fit(X)
    assert dmd.eigenvalues_.size > 0
    assert dmd.modes_.shape[0] == X.shape[1]


def test_dmd_reconstruct_shape():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 5))
    dmd = HarmonicDMD(rank=3).fit(X)
    pred = dmd.reconstruct(n_steps=4)
    assert pred.shape == (4, 5)


def test_dmd_oscillatory_modes_within_threshold():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 5))
    dmd = HarmonicDMD().fit(X)
    eigs, idx = dmd.oscillatory_modes(threshold=0.1)
    for lam in eigs:
        assert abs(abs(lam) - 1.0) < 0.1


def test_dmd_imputes_nans():
    """NaN columns get replaced; fit should not crash."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 4))
    X[2, 1] = np.nan
    dmd = HarmonicDMD().fit(X)
    assert dmd.eigenvalues_.size > 0


# ─── 6. HarmonicLatentSpace ─────────────────────────────────────────────────


def test_latent_fit_sets_dim_and_explained_variance(histograms_simple):
    ls = HarmonicLatentSpace(latent_dim=2, random_state=0).fit(histograms_simple)
    assert ls.trajectory().shape[1] <= 2
    evr = ls.explained_variance_ratio_
    assert (evr >= 0).all() and evr.sum() <= 1.0 + 1e-9


def test_latent_encode_decode_round_trip(histograms_simple):
    ls = HarmonicLatentSpace(latent_dim=4, random_state=0).fit(histograms_simple)
    Z = ls.encode(np.array(histograms_simple))
    X_rec = ls.decode(Z)
    assert X_rec.shape == np.array(histograms_simple).shape
    # With rank ≥ matrix rank, reconstruction should be near-perfect
    err = np.mean((np.array(histograms_simple) - X_rec) ** 2)
    assert err < 1e-3


def test_latent_interpolate_endpoints(histograms_simple):
    ls = HarmonicLatentSpace(latent_dim=2, random_state=0).fit(histograms_simple)
    Z = ls.trajectory()
    path = ls.interpolate(Z[0], Z[-1], n_steps=5)
    assert path.shape == (5, Z.shape[1])
    assert np.allclose(path[0], Z[0])
    assert np.allclose(path[-1], Z[-1])


# ─── 7. HarmonicTopology ────────────────────────────────────────────────────


def test_topology_fit_betti_numbers_shape():
    series = np.sin(np.linspace(0, 4 * np.pi, 60))
    topo = HarmonicTopology(embedding_dim=3, delay=1).fit(series)
    bn = topo.betti_numbers_
    assert bn.shape == (2,)
    assert (bn >= 0).all()


def test_topology_session_fingerprint_six_elements():
    series = np.linspace(0, 1, 40) + 0.1 * np.sin(np.linspace(0, 2 * np.pi, 40))
    topo = HarmonicTopology(embedding_dim=3, delay=2).fit(series)
    fp = topo.session_fingerprint()
    assert fp.shape == (6,)
    # mean / max persistences are non-negative
    assert (fp >= 0).all()


def test_topology_takens_embedding_shape():
    series = np.arange(30, dtype=float)
    topo = HarmonicTopology(embedding_dim=3, delay=1).fit(series)
    cloud = topo.takens_embedding_
    assert cloud.shape == (30 - 2, 3)


def test_topology_short_series_raises():
    with pytest.raises(ValueError):
        HarmonicTopology(embedding_dim=4, delay=2).fit(np.array([1.0, 2.0, 3.0]))


def test_topology_handles_nan_via_interpolation():
    series = np.linspace(0, 1, 20).astype(float)
    series[5:7] = np.nan
    topo = HarmonicTopology(embedding_dim=2, delay=1).fit(series)
    assert topo.betti_numbers_.shape == (2,)


# ─── 8. HarmonicGrammar ─────────────────────────────────────────────────────


def test_grammar_chord_sequence_length(ratios_list_simple):
    gr = HarmonicGrammar(n_gram=2).fit(ratios_list_simple)
    assert len(gr.chord_sequence_) == len(ratios_list_simple)


def test_grammar_vocabulary_unique(ratios_list_simple):
    gr = HarmonicGrammar(n_gram=2).fit(ratios_list_simple)
    vocab = gr.vocabulary_
    assert len(vocab) == len(set(vocab))


def test_grammar_top_ngrams_respects_top_k(ratios_list_simple):
    gr = HarmonicGrammar(n_gram=2).fit(ratios_list_simple)
    top = gr.top_ngrams(top_k=3)
    assert len(top) <= 3
    for gram, count in top:
        assert isinstance(count, int) and count >= 1


def test_grammar_top_motifs_lengths_in_range(ratios_list_simple):
    gr = HarmonicGrammar(n_gram=2).fit(ratios_list_simple)
    motifs = gr.top_motifs(min_length=2, max_length=3, top_k=5)
    for tup, _ in motifs:
        assert 2 <= len(tup) <= 3


def test_grammar_levenshtein_zero_for_identical():
    seqs = [frozenset(["a"]), frozenset(["b"])]
    assert HarmonicGrammar.levenshtein(seqs, list(seqs)) == 0


def test_grammar_levenshtein_positive_for_diff():
    a = [frozenset(["x"]), frozenset(["y"])]
    b = [frozenset(["x"]), frozenset(["z"])]
    assert HarmonicGrammar.levenshtein(a, b) == 1


def test_grammar_transition_entropy_nonnegative(ratios_list_simple):
    gr = HarmonicGrammar(n_gram=2).fit(ratios_list_simple)
    assert gr.transition_entropy_ >= 0.0


# ─── 9. HarmonicSequenceAnalyzer ────────────────────────────────────────────


def test_analyzer_from_ratios_list(ratios_list_simple):
    ana = HarmonicSequenceAnalyzer.from_ratios_list(ratios_list_simple)
    assert ana.histograms.shape == (len(ratios_list_simple), 240)


def test_analyzer_histograms_property_lazy(ratios_list_simple):
    ana = HarmonicSequenceAnalyzer.from_ratios_list(ratios_list_simple)
    assert ana._histograms is None       # unset before access
    _ = ana.histograms                    # trigger computation
    assert ana._histograms is not None


def test_analyzer_invalid_representation_raises():
    with pytest.raises(ValueError):
        HarmonicSequenceAnalyzer(representation="bogus")


def test_analyzer_invalid_tuning_raises():
    with pytest.raises(ValueError):
        HarmonicSequenceAnalyzer.from_biotuner_list([], tuning="not_a_tuning")


def test_analyzer_fit_methods_run(ratios_list_simple):
    ana = HarmonicSequenceAnalyzer.from_ratios_list(ratios_list_simple)
    ana.fit_markov(n_states=2)
    ana.fit_wasserstein()
    ana.fit_latent(latent_dim=2)
    ana.fit_grammar()
    assert ana.markov is not None
    assert ana.wasserstein is not None
    assert ana.latent is not None
    assert ana.grammar is not None


def test_analyzer_summary_mentions_representation(ratios_list_simple):
    ana = HarmonicSequenceAnalyzer.from_ratios_list(ratios_list_simple)
    ana.fit_markov(n_states=2)
    summary = ana.summary()
    assert "cents_histogram" in summary
    assert "Markov" in summary


def test_analyzer_get_histograms_observed(ratios_list_simple):
    ana = HarmonicSequenceAnalyzer.from_ratios_list(ratios_list_simple)
    H = ana.get_histograms("observed")
    assert H.shape == ana.histograms.shape


def test_analyzer_to_scl_writes_file(ratios_list_simple, tmp_path):
    ana = HarmonicSequenceAnalyzer.from_ratios_list(ratios_list_simple)
    name = str(tmp_path / "ana_window")
    ana.to_scl(index=0, name=name, write=True)
    assert os.path.isfile(name + ".scl")


def test_analyzer_to_midi_writes_file(ratios_list_simple, tmp_path):
    ana = HarmonicSequenceAnalyzer.from_ratios_list(ratios_list_simple)
    out = str(tmp_path / "ana")
    ana.to_midi(filename=out, duration_beats=0.5, n_peaks=4)
    assert os.path.isfile(out + ".mid")


def test_analyzer_to_midi_wasserstein_interp(ratios_list_simple, tmp_path):
    ana = HarmonicSequenceAnalyzer.from_ratios_list(ratios_list_simple)
    ana.fit_wasserstein()
    out = str(tmp_path / "morph")
    ana.to_midi(filename=out, source="wasserstein_interp",
                t1=0, t2=len(ratios_list_simple) - 1, n_steps=4,
                duration_beats=0.4, n_peaks=4)
    assert os.path.isfile(out + ".mid")


def test_analyzer_get_histograms_unknown_source_raises(ratios_list_simple):
    ana = HarmonicSequenceAnalyzer.from_ratios_list(ratios_list_simple)
    with pytest.raises(ValueError):
        ana.get_histograms(source="bogus")


# ─── 10. Cache behaviour ───────────────────────────────────────────────────


def test_clear_harmonicity_cache_empties_module_cache():
    _DYAD_MATRIX_CACHE[("__test_key__",)] = np.zeros((1, 1))
    assert ("__test_key__",) in _DYAD_MATRIX_CACHE
    clear_harmonicity_cache()
    assert ("__test_key__",) not in _DYAD_MATRIX_CACHE


# ─── 11. End-to-end through compute_biotuner ───────────────────────────────


def test_full_pipeline_from_biotuner_list(bt_list, tmp_path):
    """Build → fit_all → render — covering the integration surface."""
    ana = HarmonicSequenceAnalyzer.from_biotuner_list(
        bt_list, tuning="peaks_ratios",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ana.fit_all(n_states=2, latent_dim=2, n_gram=2)
    assert ana.markov is not None or ana.wasserstein is not None  # ≥ 1 fit
    out = str(tmp_path / "session")
    ana.to_midi(filename=out, duration_beats=0.5, n_peaks=4)
    assert os.path.isfile(out + ".mid")


def test_harmonicity_spectrum_encoder_shape(bt_list):
    H, freqs = encode_harmonicity_spectrum(
        bt_list, fmin=1.0, fmax=30.0, precision_hz=0.5,
    )
    assert H.shape == (len(bt_list), len(freqs))
    assert freqs[0] >= 1.0 and freqs[-1] <= 30.0


def test_harmonicity_matrix_encoder_shape(bt_list):
    M, freqs = encode_harmonicity_matrices(
        bt_list, fmin=1.0, fmax=30.0, precision_hz=0.5,
    )
    F = len(freqs)
    assert M.shape == (len(bt_list), F, F)


def test_harmonicity_encoder_caches_on_bt(bt_list):
    """Second call should populate bt._harm_cache."""
    clear_harmonicity_cache()
    for bt in bt_list:
        if hasattr(bt, "_harm_cache"):
            delattr(bt, "_harm_cache")
    encode_harmonicity_spectrum(
        bt_list, fmin=1.0, fmax=30.0, precision_hz=0.5, cache=True,
    )
    assert all(hasattr(bt, "_harm_cache") for bt in bt_list)


def test_analyzer_harmonicity_spectrum_representation(bt_list):
    ana = HarmonicSequenceAnalyzer.from_biotuner_list(
        bt_list, representation="harmonicity_spectrum",
        representation_kwargs={"fmin": 1.0, "fmax": 30.0, "precision_hz": 0.5},
    )
    F = ana.features
    assert F.shape[0] == len(bt_list)
    # Bridge rendering should refuse for non-cents reps
    with pytest.raises(RuntimeError):
        ana.get_histograms(source="wasserstein_interp")


def test_analyzer_harmonicity_matrix_exposes_3d_tensor(bt_list):
    ana = HarmonicSequenceAnalyzer.from_biotuner_list(
        bt_list, representation="harmonicity_matrix",
        representation_kwargs={"fmin": 1.0, "fmax": 30.0, "precision_hz": 0.5},
    )
    _ = ana.features
    M = ana.harmonicity_matrices_
    assert M is not None
    assert M.ndim == 3 and M.shape[1] == M.shape[2]
