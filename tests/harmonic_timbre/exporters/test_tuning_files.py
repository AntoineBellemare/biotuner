"""Tests for biotuner.harmonic_timbre.exporters.tuning_files."""

from __future__ import annotations

import math
import os

import pytest

from biotuner.harmonic_timbre.exporters import (
    export_kbm,
    export_scl,
    export_tuning_files,
)


# ---------------------------------------------------------------------------
# .scl
# ---------------------------------------------------------------------------

def test_export_scl_from_ratios(tmp_path):
    ratios = [1.0, 9 / 8, 5 / 4, 4 / 3, 3 / 2, 5 / 3, 15 / 8, 2.0]
    path = export_scl(ratios, str(tmp_path / "ji"))
    assert path.endswith(".scl")
    assert os.path.exists(path)
    text = open(path).read()
    # The unison (1/1) is implicit; the file lists 7 steps including the octave
    # We check the count line and that the octave is the last cents value
    assert " 7\n" in text
    # Last numeric line should be 1200 cents (octave)
    last_cents = float(text.strip().splitlines()[-1].strip())
    assert math.isclose(last_cents, 1200.0, abs_tol=1e-3)


def test_export_scl_from_timbre(tmp_path, matched_timbre):
    path = export_scl(matched_timbre, str(tmp_path / "from_timbre"))
    assert os.path.exists(path)


def test_export_scl_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        export_scl([], str(tmp_path / "empty"))


# ---------------------------------------------------------------------------
# .kbm
# ---------------------------------------------------------------------------

def test_export_kbm_creates_file(tmp_path):
    path = export_kbm(
        str(tmp_path / "kb"),
        n_scale_steps=7,
        middle_note=60,
        reference_note=69,
        reference_freq=440.0,
    )
    assert path.endswith(".kbm")
    text = open(path).read()
    # spot-check a few required values appear in the file
    assert " 60" in text  # middle_note
    assert " 69" in text  # reference_note
    assert "440.000000" in text


# ---------------------------------------------------------------------------
# Pair convenience
# ---------------------------------------------------------------------------

def test_export_tuning_files_pair(tmp_path, matched_timbre):
    paths = export_tuning_files(
        matched_timbre, str(tmp_path / "bundle"),
        bundle_name="testbun", reference_freq=440.0,
    )
    assert os.path.exists(paths["scl"])
    assert os.path.exists(paths["kbm"])
    assert paths["scl"].endswith("testbun.scl")
    assert paths["kbm"].endswith("testbun.kbm")
