"""Tests for biotuner.harmonic_timbre.cross_modal."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from biotuner.harmonic_timbre import (
    Timbre,
    geometry_signature_image,
    write_sidecar,
)


# ---------------------------------------------------------------------------
# Geometry signature
# ---------------------------------------------------------------------------

def test_geometry_signature_descriptor_only(tmp_path):
    t = Timbre(partials_hz=[110.0, 220.0, 440.0], amplitudes=[1.0, 0.5, 0.25])
    descriptor = geometry_signature_image(t, kind="harmonograph", out_path=None)
    assert descriptor["kind"] == "harmonograph"
    assert descriptor["path"] is None


def test_geometry_signature_writes_png(tmp_path):
    t = Timbre(partials_hz=[110.0, 220.0, 440.0], amplitudes=[1.0, 0.5, 0.25])
    out = str(tmp_path / "sig.png")
    descriptor = geometry_signature_image(t, kind="harmonograph", out_path=out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 100  # non-empty
    assert descriptor["path"] == out


def test_geometry_signature_unsupported_kind_raises():
    t = Timbre(partials_hz=[110.0], amplitudes=[1.0])
    with pytest.raises(NotImplementedError, match="Phase 3"):
        geometry_signature_image(t, kind="lissajous")


# ---------------------------------------------------------------------------
# Sidecar writer
# ---------------------------------------------------------------------------

def test_write_sidecar_creates_metadata_and_image(tmp_path):
    t = Timbre(
        partials_hz=[110.0, 220.0, 440.0],
        amplitudes=[1.0, 0.5, 0.25],
        matched_tuning=[1.0, 2.0, 4.0],
        matching_method="direct",
        metadata={"origin": "test"},
    )
    paths = write_sidecar(t, str(tmp_path), include_image=True)
    assert "metadata" in paths
    assert "signature" in paths
    # metadata.json content
    with open(paths["metadata"]) as fp:
        meta = json.load(fp)
    assert meta["matching_method"] == "direct"
    assert meta["matched_tuning"] == [1.0, 2.0, 4.0]
    assert meta["metadata"]["origin"] == "test"


def test_write_sidecar_can_skip_image(tmp_path):
    t = Timbre(partials_hz=[110.0], amplitudes=[1.0])
    paths = write_sidecar(t, str(tmp_path), include_image=False)
    assert "signature" not in paths
    assert "metadata" in paths


def test_palette_field_round_trips_when_user_set(tmp_path):
    """The Timbre.palette field is a generic placeholder; if the user fills it,
    save/load preserves it (no auto-population by the sidecar writer)."""
    from biotuner.harmonic_timbre import Timbre as _T
    t = _T(
        partials_hz=[110.0, 220.0],
        amplitudes=[1.0, 0.5],
        palette=[(255, 0, 0), (0, 255, 0)],
    )
    stem = str(tmp_path / "tb")
    t.save(stem)
    t2 = _T.load(stem)
    assert t2.palette == [(255, 0, 0), (0, 255, 0)]
