"""
build_harmonic_sequence_paper.py
=================================
One-shot builder for the ``harmonic_sequence`` use-case paper.

Orchestrates the full pipeline:
  1. Figure generation   — runs each generate_harmonic_sequence_*.py script
  2. LaTeX compilation   — compiles docs/papers/harmonic_sequence_use_cases.tex
                           via Tectonic (auto-downloads any missing TeX packages)

Usage
-----
    # Full build (figures + PDF):
    python scripts/build_harmonic_sequence_paper.py

    # Skip slow figure regeneration (use existing PNGs):
    python scripts/build_harmonic_sequence_paper.py --pdf-only

    # Regenerate figures but do not compile:
    python scripts/build_harmonic_sequence_paper.py --figs-only

Prerequisites
-------------
    pip install biotuner mne matplotlib numpy scipy scikit-learn ripser
    # Tectonic: https://tectonic-typesetting.github.io/
    cargo install tectonic          # via Rust
    # -- OR --  (Windows / conda)
    conda install -c conda-forge tectonic

The slow biotuner peaks_extraction steps cache their results to
``docs/papers/_*_cache.npz`` so subsequent runs are fast.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PAPERS = ROOT / "docs" / "papers"
TEX = PAPERS / "harmonic_sequence_use_cases.tex"

# Figure-generation scripts in the order they must run
# (later scripts can reuse cache produced by earlier ones).
FIGURE_SCRIPTS: list[tuple[str, str]] = [
    (
        "generate_harmonic_sequence_paper_figures.py",
        "Figs 01-07  basic approach demos (synthetic 3-regime cycle)",
    ),
    (
        "generate_harmonic_sequence_advanced_cases.py",
        "Figs 08-14  advanced scenarios + event detection / morphing",
    ),
    (
        "generate_harmonic_sequence_real_data.py",
        "Figs 15-17  real EEG as spatial harmonic sequence",
    ),
    (
        "generate_harmonic_sequence_real_tier1.py",
        "Figs 18-21  EEG sliding-window / cross-condition / band strat.",
    ),
    (
        "generate_harmonic_sequence_classical_apps.py",
        "Figs 22-27  classical-analogy demos (MIDI, DMD, topology …)",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(msg: str) -> None:
    width = 72
    print()
    print("=" * width)
    print(f"  {msg}")
    print("=" * width)


def _step(label: str) -> None:
    print(f"\n▶  {label}")


def _ok(elapsed: float) -> None:
    print(f"   ✓  done in {elapsed:.1f}s")


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    """Run a subprocess, streaming its stdout/stderr, raising on failure."""
    result = subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        text=True,
    )
    if result.returncode != 0:
        sys.exit(f"\n✗  command failed (exit {result.returncode}): {' '.join(cmd)}")


def _check_tectonic() -> str:
    """Return the tectonic executable path, or exit with an install hint."""
    path = shutil.which("tectonic")
    if path:
        return path
    sys.exit(
        "\n✗  tectonic not found on PATH.\n"
        "   Install it with one of:\n"
        "     cargo install tectonic\n"
        "     conda install -c conda-forge tectonic\n"
        "     https://tectonic-typesetting.github.io/\n"
    )


# ---------------------------------------------------------------------------
# Main stages
# ---------------------------------------------------------------------------

def build_figures() -> None:
    _banner("STAGE 1 — Figure generation")
    PAPERS.mkdir(parents=True, exist_ok=True)
    (PAPERS / "figures").mkdir(exist_ok=True)
    (PAPERS / "audio").mkdir(exist_ok=True)

    for script_name, description in FIGURE_SCRIPTS:
        script = SCRIPTS / script_name
        if not script.exists():
            print(f"   ⚠  {script_name} not found — skipping ({description})")
            continue
        _step(f"{script_name}  [{description}]")
        t0 = time.perf_counter()
        _run([sys.executable, str(script)])
        _ok(time.perf_counter() - t0)


def build_pdf() -> None:
    _banner("STAGE 2 — LaTeX → PDF  (Tectonic)")
    tectonic = _check_tectonic()

    if not TEX.exists():
        sys.exit(
            f"\n✗  LaTeX source not found: {TEX.relative_to(ROOT)}\n"
            "   Make sure docs/papers/harmonic_sequence_use_cases.tex is present."
        )

    _step(f"tectonic  {TEX.relative_to(ROOT)}")
    t0 = time.perf_counter()
    # --outdir places the PDF next to the .tex file
    _run([tectonic, "--outdir", str(PAPERS), str(TEX)])
    _ok(time.perf_counter() - t0)

    pdf = PAPERS / "harmonic_sequence_use_cases.pdf"
    if pdf.exists():
        size_mb = pdf.stat().st_size / 1e6
        print(f"\n   PDF written to:  {pdf.relative_to(ROOT)}  ({size_mb:.2f} MB)")
    else:
        print("\n   ⚠  PDF not found after compilation — check tectonic output above.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the harmonic_sequence use-case paper (figures + PDF)."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--pdf-only",
        action="store_true",
        help="Skip figure generation; compile LaTeX with existing PNGs.",
    )
    group.add_argument(
        "--figs-only",
        action="store_true",
        help="Generate figures only; do not compile LaTeX.",
    )
    args = parser.parse_args()

    t_total = time.perf_counter()

    if not args.pdf_only:
        build_figures()

    if not args.figs_only:
        build_pdf()

    elapsed = time.perf_counter() - t_total
    _banner(f"BUILD COMPLETE  ({elapsed:.0f}s total)")


if __name__ == "__main__":
    main()
