"""Shared helpers for exporters.

Module type: Functions

Manifest writers, path normalization, and small utilities common to
multiple exporters. Kept private (underscore-prefixed module).
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_manifest(out_path: str, manifest: dict[str, Any]) -> str:
    """Write a JSON manifest describing an export bundle.

    The manifest captures provenance and the list of files in the
    bundle so a downstream consumer can validate and re-import.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, default=_json_default)
    return out_path


def bundle_paths(
    out_dir: str,
    bundle_name: str,
    *,
    extras: dict[str, str] | None = None,
) -> dict[str, str]:
    """Compute conventional file paths for a multi-file bundle.

    Returns a dict like:
        {
          'dir':      '<out_dir>',
          'name':     '<bundle_name>',
          'manifest': '<out_dir>/<bundle_name>.manifest.json',
          'samples':  '<out_dir>/<bundle_name>_samples',
          'tuning':   '<out_dir>/<bundle_name>',  # stem for .scl/.kbm
          'sidecar':  '<out_dir>/<bundle_name>_sidecar',
        }

    Pass ``extras`` to register additional named paths.
    """
    out_dir = os.path.abspath(out_dir)
    paths = {
        "dir": out_dir,
        "name": bundle_name,
        "manifest": os.path.join(out_dir, f"{bundle_name}.manifest.json"),
        "samples": os.path.join(out_dir, f"{bundle_name}_samples"),
        "tuning": os.path.join(out_dir, bundle_name),
        "sidecar": os.path.join(out_dir, f"{bundle_name}_sidecar"),
    }
    if extras:
        paths.update({k: os.path.join(out_dir, v) for k, v in extras.items()})
    return paths


def midi_to_hz(midi_note: int, *, a4: float = 440.0) -> float:
    """Convert a MIDI note number to frequency in Hz (12-TET reference)."""
    return float(a4) * (2.0 ** ((float(midi_note) - 69.0) / 12.0))


def midi_note_name(midi_note: int) -> str:
    """Standard pitch name for a MIDI note (e.g. ``60 -> 'C4'``)."""
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[midi_note % 12]}{midi_note // 12 - 1}"


def safe_filename(name: str) -> str:
    """Strip filesystem-unfriendly characters from ``name``."""
    out = []
    for ch in name:
        if ch.isalnum() or ch in "_-.":
            out.append(ch)
        elif ch in " /\\":
            out.append("_")
    return "".join(out) or "unnamed"
