"""
test_harmonic_sequence_capabilities.py
======================================
End-to-end capability check for ``biotuner.harmonic_sequence``.

Synthesises a controlled sequence of biosignals whose harmonic content
drifts between a "just major" and "just minor" regime, runs the full
HarmonicSequenceAnalyzer pipeline, and prints what each of the six
approaches reports — together with a pass / fail verdict against a
hand-derived expectation.

Run:
    python scripts/test_harmonic_sequence_capabilities.py

Optional flags:
    --skip-bt        Use synthetic ratio sequences only (no compute_biotuner).
    --no-midi        Skip MIDI/SCL export.
    --outdir <path>  Where to write generated artifacts (default: ./out_hseq).
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import warnings
from pathlib import Path
from typing import List

import numpy as np

# Make sure the repo root is importable when run from anywhere
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biotuner.harmonic_sequence import (
    HarmonicDMD,
    HarmonicGrammar,
    HarmonicLatentSpace,
    HarmonicMarkov,
    HarmonicSequenceAnalyzer,
    HarmonicTopology,
    WassersteinTrajectory,
    encode_histograms,
    encode_ji_matrix,
    histogram_to_ratios,
    histogram_to_scl,
    histograms_to_midi,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────────────────────

# Force UTF-8 stdout on Windows so any unicode in metric labels doesn't crash.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ANSI colours work on Win10+ terminals; harmless if not rendered.
GREEN, RED, YELLOW, BLUE, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[0m"
RESULTS: list[tuple[str, str, str]] = []  # (section, status, message)


def section(title: str) -> None:
    bar = "=" * 72
    print(f"\n{BLUE}{bar}\n  {title}\n{bar}{RESET}")


def report(name: str, ok: bool, msg: str = "") -> None:
    tag = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  [{tag}] {name}" + (f"  -- {msg}" if msg else ""))
    RESULTS.append((name, "PASS" if ok else "FAIL", msg))


def info(msg: str) -> None:
    print(f"  {YELLOW}*{RESET} {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data
# ─────────────────────────────────────────────────────────────────────────────

JUST_MAJOR = [1.0, 9 / 8, 5 / 4, 4 / 3, 3 / 2, 5 / 3, 15 / 8]   # 7-note JI major
JUST_MINOR = [1.0, 9 / 8, 6 / 5, 4 / 3, 3 / 2, 8 / 5, 9 / 5]    # 7-note JI minor


def synthetic_ratio_sequence(repeats: int = 4) -> List[List[float]]:
    """Alternating major/minor regimes with small drift inside each block."""
    seq: List[List[float]] = []
    rng = np.random.default_rng(0)
    for r in range(repeats):
        # 3 major frames with a tiny cents-drift
        for k in range(3):
            jitter = 1.0 + 0.002 * (k - 1) + 0.0005 * rng.standard_normal(len(JUST_MAJOR))
            seq.append([float(x * j) for x, j in zip(JUST_MAJOR, jitter)])
        # 3 minor frames with a tiny cents-drift
        for k in range(3):
            jitter = 1.0 + 0.002 * (k - 1) + 0.0005 * rng.standard_normal(len(JUST_MINOR))
            seq.append([float(x * j) for x, j in zip(JUST_MINOR, jitter)])
    return seq


def build_bt_list(n_blocks: int = 3, sf: int = 1000, duration: float = 4.0):
    """Build a small list of fitted compute_biotuner objects.

    Each window's signal is a harmonic stack whose fundamental drifts so the
    sequence has measurable harmonic motion across windows.
    """
    from biotuner.biotuner_object import compute_biotuner
    rng = np.random.default_rng(42)

    def _signal(freqs):
        t = np.linspace(0, duration, int(sf * duration), endpoint=False)
        sig = sum((1.0 / (i + 1)) * np.sin(2 * np.pi * f * t)
                  for i, f in enumerate(freqs))
        sig += 0.02 * rng.standard_normal(len(t))
        return sig.astype(np.float64)

    # Two regimes: "alpha-stack" and "shifted alpha-stack"
    freq_set_A = [2, 5, 10, 20, 40]
    freq_set_B = [2, 4.8, 7.2, 14.4, 28.8]
    freq_seq = []
    for _ in range(n_blocks):
        freq_seq += [freq_set_A, freq_set_A, freq_set_B, freq_set_B]

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


# ─────────────────────────────────────────────────────────────────────────────
# Capability checks
# ─────────────────────────────────────────────────────────────────────────────

def check_encoders(ratios_list):
    section("1 — Stateless encoders")

    H = encode_histograms(ratios_list, n_bins=240)
    info(f"cents histogram shape: {H.shape}")
    row_sums = H.sum(axis=1)
    report(
        "histograms normalised per row",
        np.allclose(row_sums[row_sums > 0], 1.0),
        f"max |Σ-1| = {abs(row_sums - 1).max():.3g}",
    )

    # Known landmarks: a 3:2 (~702¢) ratio must light a bin near 702/5 = 140
    H1 = encode_histograms([[1.5]], n_bins=240)
    fifth_bin = int(round(1200 * np.log2(1.5) / 5))
    report(
        "perfect fifth lands in expected bin",
        H1[0, fifth_bin] > 0,
        f"bin {fifth_bin} has weight {H1[0, fifth_bin]:.2f}",
    )

    # JI matrix should mark known intervals
    X, labels = encode_ji_matrix([[5 / 4, 3 / 2]], tolerance_cents=20.0)
    n_hits = int(X.sum())
    report(
        "JI matrix labels known intervals",
        n_hits >= 2,
        f"{n_hits} of {len(labels)} interval slots matched",
    )

    return H


def check_bridge(H, outdir: Path, want_midi: bool):
    section("2 — Decoding bridge (histogram → ratios / .scl / MIDI)")

    ratios = histogram_to_ratios(H[0], include_unison=False, include_octave=False)
    info(f"decoded {len(ratios)} ratios from frame 0: " +
         ", ".join(f"{r:.4f}" for r in ratios[:6]) + (" …" if len(ratios) > 6 else ""))
    report(
        "round-trip recovers ≥ 1 ratio",
        len(ratios) >= 1,
        "fails only if histogram peak detection misses all peaks",
    )

    # Round-trip distance check on JUST_MAJOR (the canonical input)
    H_one = encode_histograms([JUST_MAJOR])[0]
    rec = histogram_to_ratios(H_one, include_unison=False, include_octave=False)
    cents_orig = sorted(1200 * np.log2(r) for r in JUST_MAJOR if r > 0)
    cents_rec = sorted(1200 * np.log2(r) for r in rec)
    # cents_orig has 7 entries (unison at 0¢); decoded has ≤7; compare overlap
    if len(cents_rec) >= len(cents_orig) - 2:
        max_err = max(min(abs(co - cr) for cr in cents_rec) for co in cents_orig if co > 0)
        report(
            "round-trip cents error ≤ 5¢ (one bin)",
            max_err <= 5.0,
            f"worst cents error = {max_err:.2f}¢",
        )
    else:
        report("round-trip cents error ≤ 5¢ (one bin)", False,
               f"decoded only {len(cents_rec)} of {len(cents_orig)} intervals")

    # SCL export (string, no disk)
    scl = histogram_to_scl(H[0], name="hseq_test", write=False)
    has_header = scl.lstrip().startswith("!")
    report("SCL export produces valid header", has_header,
           f"first line: {scl.splitlines()[0]!r}")

    if not want_midi:
        info("MIDI export skipped (--no-midi).")
        return

    try:
        outdir.mkdir(parents=True, exist_ok=True)
        cwd = os.getcwd()
        os.chdir(outdir)
        try:
            mid = histograms_to_midi(
                H[:8], filename="hseq_test", base_freq=220.0,
                duration_beats=1.0, n_peaks=4, microtonal=True,
            )
        finally:
            os.chdir(cwd)
        n_tracks = len(mid.tracks)
        report("MIDI export wrote a multi-track file", n_tracks >= 1,
               f"{n_tracks} tracks, file at {outdir/'hseq_test.mid'}")
    except Exception as exc:
        report("MIDI export wrote a multi-track file", False, f"exception: {exc}")


def check_markov(analyzer: HarmonicSequenceAnalyzer):
    section("3 — Approach 1: HarmonicMarkov")

    mk = analyzer.fit_markov(n_states="auto", auto_k_range=(2, 6))
    info(f"selected K = {mk.n_states} (silhouette scores: {mk.silhouette_scores_})")

    T = mk.transition_matrix_
    rows_normalised = np.allclose(T.sum(axis=1), 1.0)
    report("transition rows sum to 1", rows_normalised,
           f"row sums = {T.sum(axis=1)}")

    pi = mk.steady_state_
    report("stationary distribution is a probability vector",
           np.isclose(pi.sum(), 1.0) and (pi >= -1e-9).all(),
           f"π = {np.round(pi, 3).tolist()}")

    H_trans = mk.transition_entropy_
    report("transition entropy in [0, log2(K)]",
           0.0 <= H_trans <= np.log2(mk.n_states) + 1e-9,
           f"H = {H_trans:.3f} bits  (max = {np.log2(mk.n_states):.3f})")

    # Sanity: the labels alternate when the source data alternates
    labels = mk.state_labels_
    n_unique = len(set(labels.tolist()))
    report("at least 2 distinct states discovered on alternating data",
           n_unique >= 2, f"{n_unique} unique states in {len(labels)} frames")

    # Predictive interface
    proba = mk.predict_next_proba(int(labels[0]))
    report("predict_next_proba returns valid distribution",
           proba.shape == (mk.n_states,) and np.isclose(proba.sum(), 1.0),
           f"P(·|s={labels[0]}) = {np.round(proba, 2).tolist()}")


def check_wasserstein(analyzer: HarmonicSequenceAnalyzer):
    section("4 — Approach 2: WassersteinTrajectory")

    wt = analyzer.fit_wasserstein()
    D = wt.distance_matrix_
    flux = wt.flux_
    info(f"distance matrix shape {D.shape}; flux length {len(flux)}")

    report("distance matrix is symmetric, zero-diag",
           np.allclose(D, D.T) and np.allclose(np.diag(D), 0.0),
           f"max asym = {np.abs(D - D.T).max():.3g}")

    report("flux is non-negative",
           (flux >= -1e-12).all(),
           f"min flux = {flux.min():.4g}")

    # Barycenter: α=0 → h1, α=1 → h2
    X = analyzer.features
    if X.shape[0] >= 2:
        b0 = wt.barycenter(X[0], X[1], alpha=0.0)
        report("barycenter(α=0) ≈ h1",
               np.allclose(b0, X[0] / max(X[0].sum(), 1e-9), atol=0.05),
               "tolerance is loose: barycenter samples 1000 quantile pts")

    # Embedding
    try:
        Z = wt.embed(n_components=2)
        report("MDS embedding returns (T, 2)",
               Z.shape == (D.shape[0], 2),
               f"Z range x:[{Z[:,0].min():.2f},{Z[:,0].max():.2f}] "
               f"y:[{Z[:,1].min():.2f},{Z[:,1].max():.2f}]")
    except Exception as exc:
        report("MDS embedding returns (T, 2)", False, f"exception: {exc}")


def check_dmd(analyzer: HarmonicSequenceAnalyzer):
    section("5 — Approach 3: HarmonicDMD")

    if analyzer._bt_list is None:
        # Without bt objects scalar_metrics is empty; fall back to histogram PCA
        info("no bt_list → using use_histograms=True path")
        dmd = analyzer.fit_dmd(use_histograms=True)
    else:
        dmd = analyzer.fit_dmd()

    eigs = dmd.eigenvalues_
    modes = dmd.modes_
    info(f"rank = {len(eigs)}; mode matrix {modes.shape}")
    report("eigenvalues finite",
           np.all(np.isfinite(eigs)),
           f"|λ| range = [{np.abs(eigs).min():.3g}, {np.abs(eigs).max():.3g}]")

    growth = dmd.growth_rates_
    freqs = dmd.frequencies_
    report("growth_rates_ and frequencies_ are real",
           np.isrealobj(growth) and np.isrealobj(freqs),
           f"growth ∈ [{growth.min():.3g}, {growth.max():.3g}]")

    osc, idx = dmd.oscillatory_modes(threshold=0.1)
    info(f"{len(osc)} oscillatory modes (|λ|≈1 within 0.1)")

    pred = dmd.reconstruct(n_steps=4)
    report("reconstruct returns (n_steps, D)",
           pred.ndim == 2 and pred.shape[0] == 4,
           f"shape = {pred.shape}")


def check_latent(analyzer: HarmonicSequenceAnalyzer):
    section("6 — Approach 4: HarmonicLatentSpace")

    ls = analyzer.fit_latent(latent_dim=3)
    Z = ls.trajectory()
    info(f"latent trajectory shape {Z.shape}")

    evr = ls.explained_variance_ratio_
    report("variance ratios in [0,1] and ≤ 1 in total",
           (evr >= 0).all() and evr.sum() <= 1.0 + 1e-9,
           f"per-dim = {np.round(evr, 3).tolist()} → sum = {evr.sum():.3f}")

    X = analyzer.features
    Z2 = ls.encode(X)
    report("encode is invertible (≤ MSE bound)",
           ls.reconstruction_error_ < 1e-2,
           f"MSE = {ls.reconstruction_error_:.4g}")

    Z_path = ls.interpolate(Z[0], Z[-1], n_steps=5)
    report("interpolate returns (n_steps, latent_dim)",
           Z_path.shape == (5, ls.latent_dim),
           f"endpoints match: "
           f"{np.allclose(Z_path[0], Z[0])} / {np.allclose(Z_path[-1], Z[-1])}")


def check_topology(analyzer: HarmonicSequenceAnalyzer):
    section("7 — Approach 6: HarmonicTopology  (TDA)")

    try:
        # Use a smooth scalar so the Takens cloud is meaningful
        scalar = "harmsim" if analyzer._bt_list is not None else "mean_features"
        top = analyzer.fit_topology(scalar_key=scalar, embedding_dim=3, delay=1)
    except Exception as exc:
        report("topology fit", False, f"exception: {exc}")
        return

    cloud = top.takens_embedding_
    info(f"Takens cloud shape {cloud.shape}  (d={top.embedding_dim}, τ={top.delay})")

    diag = top.persistence_diagram_
    info(f"persistence diagrams: {len(diag)} (H0"
         + (", H1" if len(diag) > 1 else "") + ")")

    beta = top.betti_numbers_
    info(f"β0 = {beta[0]}, β1 = {beta[1]}")

    fp = top.session_fingerprint()
    report("session_fingerprint returns 6-vector",
           fp.shape == (6,) and np.all(np.isfinite(fp)),
           f"fingerprint = {np.round(fp, 3).tolist()}")


def check_grammar(analyzer: HarmonicSequenceAnalyzer):
    section("8 — Approach 7: HarmonicGrammar")

    gr = analyzer.fit_grammar(n_gram=2)
    vocab = gr.vocabulary_
    info(f"vocab size = {len(vocab)}; first token = {set(vocab[0])}")

    top = gr.top_ngrams(3)
    info("top-3 bigrams (chord1 | chord2 → count):")
    for gram, count in top:
        a, b = gram
        info(f"  {sorted(a)}  →  {sorted(b)}  ×{count}")

    H_g = gr.transition_entropy_
    report("grammar entropy ≥ 0", H_g >= 0.0, f"H = {H_g:.3f} bits")

    motifs = gr.top_motifs(min_length=2, max_length=3, top_k=3)
    report("top_motifs returns ≤ top_k entries",
           1 <= len(motifs) <= 3, f"{len(motifs)} motifs found")

    d = gr.levenshtein(gr.chord_sequence_, gr.chord_sequence_)
    report("levenshtein(seq, seq) == 0", d == 0, f"d = {d}")


def check_orchestrator_and_bridge(
    analyzer: HarmonicSequenceAnalyzer, outdir: Path, want_midi: bool,
):
    section("9 — HarmonicSequenceAnalyzer end-to-end + render bridge")

    print(analyzer.summary())

    # 'observed' source
    H_obs = analyzer.get_histograms(source="observed")
    report("get_histograms('observed') shape matches input",
           H_obs.shape == (len(analyzer.ratios_list), analyzer.n_hist_bins),
           f"shape = {H_obs.shape}")

    # Wasserstein-driven glissando
    try:
        H_glide = analyzer.get_histograms(
            source="wasserstein_interp", t1=0, t2=len(analyzer.ratios_list) - 1,
            n_steps=6,
        )
        report("wasserstein_interp source returns (n_steps, n_bins)",
               H_glide.shape == (6, analyzer.n_hist_bins),
               f"shape = {H_glide.shape}")
    except Exception as exc:
        report("wasserstein_interp source", False, f"exception: {exc}")

    # Latent decode
    try:
        H_latent = analyzer.get_histograms(
            source="latent_interp", t1=0, t2=len(analyzer.ratios_list) - 1,
            n_steps=6,
        )
        report("latent_interp source returns valid shape",
               H_latent.shape == (6, analyzer.n_hist_bins),
               f"shape = {H_latent.shape}")
    except Exception as exc:
        report("latent_interp source", False, f"exception: {exc}")

    # Markov centroids + sample
    try:
        H_cent = analyzer.get_histograms(source="markov_centroids")
        H_samp = analyzer.get_histograms(source="markov_sample", n_steps=8)
        report("markov_centroids / markov_sample shapes",
               H_cent.shape[1] == analyzer.n_hist_bins
               and H_samp.shape == (8, analyzer.n_hist_bins),
               f"cent = {H_cent.shape}, sample = {H_samp.shape}")
    except Exception as exc:
        report("markov_centroids / markov_sample", False, f"exception: {exc}")

    # Optional MIDI render through analyzer.to_midi
    if want_midi:
        try:
            outdir.mkdir(parents=True, exist_ok=True)
            cwd = os.getcwd()
            os.chdir(outdir)
            try:
                analyzer.to_midi(
                    filename="hseq_observed", source="observed",
                    duration_beats=0.5, n_peaks=4,
                )
            finally:
                os.chdir(cwd)
            report("analyzer.to_midi('observed') wrote a file",
                   (outdir / "hseq_observed.mid").exists(),
                   str(outdir / "hseq_observed.mid"))
        except Exception as exc:
            report("analyzer.to_midi('observed')", False, f"exception: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-bt", action="store_true",
                        help="Use only synthetic ratio sequences (skip "
                             "compute_biotuner / signal pipeline).")
    parser.add_argument("--no-midi", action="store_true",
                        help="Skip MIDI rendering.")
    parser.add_argument("--outdir", default="out_hseq",
                        help="Directory for generated MIDI/SCL artifacts.")
    args = parser.parse_args()

    outdir = Path(args.outdir).resolve()

    section("0 — Build a sequence")
    ratios_list = synthetic_ratio_sequence(repeats=4)
    info(f"synthetic ratio sequence: T = {len(ratios_list)} frames, "
         f"~{len(ratios_list[0])} ratios per frame, alternating major/minor")

    # Stateless encoders + bridge (no analyzer needed)
    H = check_encoders(ratios_list)
    check_bridge(H, outdir, want_midi=not args.no_midi)

    # Build an analyzer
    if args.skip_bt:
        info("Building analyzer from ratio list (--skip-bt).")
        analyzer = HarmonicSequenceAnalyzer.from_ratios_list(ratios_list)
    else:
        try:
            info("Building bt_list (this is the slow step — ~10 s).")
            bts = build_bt_list(n_blocks=3)
            analyzer = HarmonicSequenceAnalyzer.from_biotuner_list(
                bts, tuning="peaks_ratios",
            )
            info(f"bt_list ready: {len(bts)} windows, "
                 f"each with ~{len(bts[0].peaks) if bts[0].peaks is not None else 0} peaks")
        except Exception as exc:
            print(f"  {RED}!{RESET} compute_biotuner pipeline failed "
                  f"({exc}). Falling back to ratio-list mode.")
            analyzer = HarmonicSequenceAnalyzer.from_ratios_list(ratios_list)

    # Approach-by-approach checks
    check_markov(analyzer)
    check_wasserstein(analyzer)
    check_dmd(analyzer)
    check_latent(analyzer)
    check_topology(analyzer)
    check_grammar(analyzer)
    check_orchestrator_and_bridge(analyzer, outdir, want_midi=not args.no_midi)

    # Final tally
    section("Verdict")
    n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    print(f"  {GREEN}{n_pass} passed{RESET}, {RED}{n_fail} failed{RESET} "
          f"({n_pass + n_fail} total assertions)")
    if n_fail:
        print(f"\n  {RED}Failed assertions:{RESET}")
        for name, status, msg in RESULTS:
            if status == "FAIL":
                print(f"    - {name}  ({msg})")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
