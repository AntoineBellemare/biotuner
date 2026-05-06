"""biotuner.harmonic_timbre.cross_modal — visual identity sidecar for timbres.

Module type: Functions

Pure metadata. Never touches audio. Gives every Timbre a recognizable
visual fingerprint travelling alongside the audio.

Phase 1 surface
---------------
geometry_signature_image
    Render a small PNG/SVG of the Timbre's harmonic-geometry signature
    (Phase 1 supports the harmonograph kind; v1.1 adds Lissajous,
    Chladni, polygon, point cloud, polyrhythm phase wheel, etc.).
write_sidecar
    One-stop shop: writes ``signature.png`` and a metadata blob into a
    directory next to the audio. Every exporter (Phase 2) calls this.

The biocolors/biolelements integration (peak frequencies → RGB palette,
peak frequencies → atomic spectral lines) is deliberately not wired
here yet — the conceptual link to timbre needs more thought before it
becomes part of the default sidecar. ``Timbre.palette`` and
``Timbre.elements`` remain available as user-fillable fields and
round-trip through save/load, but no auto-population.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Iterable

import numpy as np

from biotuner.harmonic_timbre.timbre import Timbre

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry signature image
# ---------------------------------------------------------------------------

def geometry_signature_image(
    timbre: Timbre,
    *,
    kind: str = "harmonograph",
    size: tuple[int, int] = (512, 512),
    out_path: str | None = None,
    dpi: int = 100,
) -> dict:
    """Render a small image of the Timbre's harmonic-geometry signature.

    Phase 1 supports ``kind='harmonograph'``. Other kinds raise
    NotImplementedError until Phase 3 adds Lissajous / Chladni /
    polygon / polyrhythm-phase-wheel / second-order-tree options.

    Parameters
    ----------
    timbre : Timbre
        Source timbre (uses ``partials_hz``, ``amplitudes``, optional
        ``phases``, optional ``decay_times`` -> damping).
    kind : str, default='harmonograph'
    size : (width_px, height_px), default=(512, 512)
    out_path : str, optional
        If given, save as PNG (or SVG when path ends in ``.svg``).
        If None, no file is written but the descriptor still records
        kind/size for downstream consumers.
    dpi : int, default=100

    Returns
    -------
    dict
        ``{'kind': str, 'path': str | None, 'size': (w, h)}``.
        Suitable for ``Timbre.geometry_signature``.
    """
    timbre.validate()

    if kind != "harmonograph":
        raise NotImplementedError(
            f"geometry_signature_image: kind={kind!r} is planned for Phase 3; "
            "Phase 1 supports only 'harmonograph'."
        )

    from biotuner.harmonic_geometry.harmonograph import (
        harmonograph_from_peaks,
        derive_damping_from_linewidth,
    )

    peaks = list(timbre.partials_hz.tolist())
    amps = list(timbre.amplitudes.tolist())
    phases = (
        list(timbre.phases.tolist())
        if timbre.phases is not None
        else None
    )
    if timbre.bandwidths is not None:
        damping = derive_damping_from_linewidth(timbre.bandwidths.tolist()).tolist()
    else:
        damping = None

    geom = harmonograph_from_peaks(
        peaks=peaks,
        amps=amps,
        phases=phases,
        damping=damping,
        duration=30.0,
        sr=200,
    )

    descriptor: dict = {"kind": kind, "size": tuple(size), "path": None}
    if out_path is None:
        return descriptor

    # render via matplotlib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    coords = geom.coordinates
    # coordinates can be a single (N, D) array or a list of arrays per voice
    if isinstance(coords, list):
        for c in coords:
            arr = np.asarray(c, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                ax.plot(arr[:, 0], arr[:, 1], color="#5b6dcd", linewidth=0.6, alpha=0.7)
    else:
        arr = np.asarray(coords, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            ax.plot(arr[:, 0], arr[:, 1], color="#5b6dcd", linewidth=0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    descriptor["path"] = out_path
    return descriptor


# ---------------------------------------------------------------------------
# Sidecar writer
# ---------------------------------------------------------------------------

def write_sidecar(
    timbre: Timbre,
    out_dir: str,
    *,
    include_image: bool = True,
    image_kind: str = "harmonograph",
    stem: str = "signature",
) -> dict:
    """Write the Timbre's sidecar files into ``out_dir``.

    Always writes ``<stem>_metadata.json`` (full Timbre provenance).
    Optionally also writes ``<stem>.png`` (a harmonic-geometry visual
    fingerprint). Returns a dict of paths actually written.

    Used by every exporter so a musician opening e.g. an SFZ also gets
    a `.png` thumbnail and a metadata blob describing the matched
    tuning, matching method, and which biotuner fields were consumed.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths: dict[str, str] = {}

    # metadata always
    meta_path = os.path.join(out_dir, f"{stem}_metadata.json")
    meta = {
        "matched_tuning": list(timbre.matched_tuning) if timbre.matched_tuning is not None else None,
        "matching_method": timbre.matching_method,
        "metadata": dict(timbre.metadata),
        "n_partials": timbre.n_partials(),
        "base_freq": timbre.base_freq,
        "spectral_tilt": timbre.spectral_tilt,
        "noise_floor": timbre.noise_floor,
    }
    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2, default=_json_default)
    paths["metadata"] = meta_path

    if include_image:
        img_path = os.path.join(out_dir, f"{stem}.png")
        try:
            descriptor = geometry_signature_image(
                timbre, kind=image_kind, out_path=img_path,
            )
            paths["signature"] = descriptor["path"] or img_path
        except Exception as exc:
            logger.warning(
                "write_sidecar: geometry signature render failed (%s); "
                "skipping image", exc,
            )

    return paths


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
