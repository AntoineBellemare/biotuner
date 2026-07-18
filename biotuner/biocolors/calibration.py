"""Percentile calibration: raw descriptors -> ``[0, 1]``, fit to real data.

Module type: Functions + dataclass

Why this layer exists
---------------------
The prototype normalised mean consonance by dividing by 55 -- a hand-tuned
constant with no model of what the metric does on real signals. Measured on
real sleep EEG, mean consonance lives in roughly ``0.3 - 5``, so ``avg / 55``
maps every recording into the bottom 9% of the hue wheel. The consequence, over
30 EEG-like signals: anchor hues spanned 16.7-42.0 deg out of 360, the median
gap between adjacent signals was **0.20 deg**, and **242 of 435 pairs** were
within 5 deg of each other. Every signal came out the same orange-red. The
constant also saturates at the top: a harmonic series scores 59.6, so
``59.6/55`` clips to 1.0 and every strongly harmonic signal collapses onto the
identical hue.

A magic constant cannot be right, because the right value depends on the
distribution of the data. Percentile ranking against a reference corpus is
distribution-matched *by construction*: it spends colour budget exactly where
signals actually differ, it cannot saturate, and it is monotone so ordering is
preserved.

Stability
---------
A calibration is fit **once**, shipped, and versioned. It is never re-fit per
call: doing so would make a palette depend on which other signals happened to
be in the batch, destroying reproducibility. That is the whole point of
``mode="absolute"``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

from biotuner.biocolors.descriptors import (
    FINGERPRINT_FIELDS,
    Fingerprint,
    fingerprint,
    make_context,
)

__all__ = [
    "Calibration",
    "build_calibration",
    "load_calibration",
    "list_calibrations",
    "DATA_DIR",
]

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
N_QUANTILES = 101


@dataclass
class Calibration:
    """Percentile tables + a fixed PCA projection for a population of signals.

    Attributes
    ----------
    name : str
        Version tag, e.g. ``"eeg_sleep_v1"``.
    percentiles : dict
        descriptor name -> ``(N_QUANTILES,)`` sorted quantile values.
    pca_mean, pca_components : ndarray
        Fit on the *normalised* fingerprint vectors of the corpus. Stored, not
        re-fit, so that one signal always lands in the same place.
    fields : tuple
        Descriptor order the projection was fit against.
    meta : dict
        Provenance: corpus size, source, date.
    """

    name: str
    percentiles: Dict[str, np.ndarray]
    pca_mean: np.ndarray
    pca_components: np.ndarray
    pca_explained: np.ndarray
    fields: Sequence[str] = FINGERPRINT_FIELDS
    weights: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.weights is None:
            self.weights = np.ones(len(self.fields))

    # -- normalisation ---------------------------------------------------- #
    def normalize(self, name, value):
        """Percentile-rank ``value`` for descriptor ``name`` -> ``[0, 1]``.

        Values outside the corpus range saturate at 0 or 1 rather than being
        clipped mid-scale, and the mapping is monotone throughout.
        """
        q = self.percentiles.get(name)
        if q is None:
            return 0.5
        return float(np.interp(value, q, np.linspace(0.0, 1.0, len(q))))

    def normalize_vector(self, fp: Fingerprint):
        return fp.normalized(self, fields=self.fields)

    def informativeness(self):
        """Per-descriptor weight learned from the corpus. See :func:`_weights`."""
        return dict(zip(self.fields, np.round(self.weights, 4)))

    def saturation(self, fp: Fingerprint):
        """Fraction of a fingerprint's descriptors that fall outside the corpus.

        A percentile map is only meaningful inside the range it was fit on.
        Outside it, :meth:`normalize` pins to exactly 0.0 or 1.0 — the value
        stops carrying information, and if enough descriptors pin, the PCA
        direction collapses toward a handful of attractors and unrelated signals
        share a hue.

        This is the honest limit of a single-corpus calibration, and it bites in
        practice: ``eeg_sleep_v1`` is fit on ``peaks_function='fixed'``, and
        measured on the same recording, FOOOF saturates **82%** of descriptors,
        cepstrum 64%, HH1D_max 61% — because those extractors report a different
        kind of quantity. ``adapt``, a close cousin of ``fixed``, saturates 9%.

        Above ~0.3, treat the palette as out-of-domain: refit with
        :func:`build_calibration` on a corpus from *your* extractor, or pass
        ``calibration='none'``. Reported by
        :func:`~biotuner.biocolors.palettes.palette_report` as
        ``fingerprint_saturation``.
        """
        v = self.normalize_vector(fp)
        return float(np.mean((v <= 1e-9) | (v >= 1.0 - 1e-9)))

    # -- embedding -------------------------------------------------------- #
    def project(self, fp: Fingerprint):
        """Normalised fingerprint -> 2-D coordinates via the stored PCA.

        Returns ``(pc1, pc2)``. Hue comes from the *direction* of this vector
        and chroma scale from its *magnitude*, so an unusual signal (far from
        the corpus centroid) reads as vivid and a typical one as muted.
        """
        v = (self.normalize_vector(fp) - self.pca_mean) * self.weights
        return self.pca_components @ v

    def save(self, path=None):
        path = path or os.path.join(DATA_DIR, f"{self.name}.npz")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(
            path,
            name=self.name,
            fields=np.array(list(self.fields)),
            pca_mean=self.pca_mean,
            pca_components=self.pca_components,
            pca_explained=self.pca_explained,
            weights=self.weights,
            meta=np.array([repr(self.meta)]),
            **{f"pct__{k}": v for k, v in self.percentiles.items()},
        )
        return path

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        pcts = {k[5:]: z[k] for k in z.files if k.startswith("pct__")}
        try:
            meta = eval(str(z["meta"][0]))  # noqa: S307 - our own repr
        except Exception:
            meta = {}
        return cls(
            name=str(z["name"]),
            percentiles=pcts,
            pca_mean=z["pca_mean"],
            pca_components=z["pca_components"],
            pca_explained=z["pca_explained"],
            fields=tuple(str(x) for x in z["fields"]),
            weights=z["weights"] if "weights" in z.files else None,
            meta=meta,
        )

    @classmethod
    def identity(cls, fields=FINGERPRINT_FIELDS):
        """A no-op calibration: descriptors pass through, clipped to [0, 1].

        ``calibration="none"``. Reproduces raw-value behaviour; useful for
        debugging and for signals unlike any corpus.
        """
        pct = {f: np.linspace(0.0, 1.0, N_QUANTILES) for f in fields}
        n = len(fields)
        return cls(
            name="identity",
            percentiles=pct,
            pca_mean=np.zeros(n),
            pca_components=np.eye(n)[:2],
            pca_explained=np.array([0.5, 0.5]),
            fields=fields,
            meta={"note": "identity / uncalibrated"},
        )

    @classmethod
    def from_signals(cls, signals, name="custom", fields=FINGERPRINT_FIELDS, **kw):
        """Fit to your own corpus: ``signals`` is a list of ``(peaks, amps)``."""
        return build_calibration(signals, name=name, fields=fields, **kw)

    def __repr__(self):
        n = self.meta.get("n_signals", "?")
        return f"Calibration({self.name!r}, n_signals={n}, fields={len(self.fields)})"


def _weights(raw, n_bins=40):
    """How much does each descriptor actually tell you, on this corpus?

    Percentile normalisation has one failure mode: it is only meaningful for a
    descriptor with a spread-out distribution. Applied to a **degenerate**
    descriptor -- one that returns the same value for most signals -- it
    stretches quantisation noise across the full ``[0, 1]`` range and injects it
    into the palette as if it were signal.

    This is not hypothetical. On the HMC sleep corpus, ``n_peaks`` is 5 for ~98%
    of epochs (std 0.045) and ``complexity`` is 0 for most (Higuchi FD over five
    amplitude values is meaningless). Left unweighted, those two would contribute
    as much hue variance as ``harmonicity``, which genuinely spans 7.4 to 60.4.

    The weight is ``1 - modal_bin_fraction``: a descriptor whose values pile into
    one histogram bin scores ~0, one spread evenly scores ~1. Learned per corpus,
    so a descriptor that is useless here can still carry weight on a corpus where
    it genuinely varies -- the calibration adapts rather than the code hardcoding
    a field list.
    """
    n_sig, n_desc = raw.shape
    w = np.ones(n_desc)
    for j in range(n_desc):
        col = raw[:, j]
        if np.ptp(col) < 1e-12:
            w[j] = 0.0
            continue
        hist, _ = np.histogram(col, bins=n_bins)
        w[j] = 1.0 - hist.max() / n_sig
    # Renormalise so the mean informative descriptor has weight ~1.
    pos = w[w > 1e-6]
    if pos.size:
        w = w / pos.mean()
    return w


def build_calibration(signals, name="custom", fields=FINGERPRINT_FIELDS,
                      progress=False):
    """Fit percentile tables, descriptor weights and a PCA projection.

    Parameters
    ----------
    signals : sequence
        Each item is ``(peaks, amps)``, or a
        :class:`~biotuner.biocolors.descriptors.Fingerprint`.
    name : str
        Version tag to store under.

    Returns
    -------
    Calibration
    """
    fps = []
    for i, s in enumerate(signals):
        if progress and i % 200 == 0:
            print(f"  fingerprinting {i}/{len(signals)}")
        if isinstance(s, Fingerprint):
            fps.append(s)
            continue
        try:
            peaks, amps = s
            fps.append(fingerprint(make_context(peaks, amps), fields=fields))
        except Exception:
            continue
    if len(fps) < 8:
        raise ValueError(f"need >=8 usable signals to calibrate, got {len(fps)}")

    raw = np.array([fp.vector(fields) for fp in fps], float)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

    qs = np.linspace(0.0, 100.0, N_QUANTILES)
    percentiles = {}
    for j, f in enumerate(fields):
        col = np.sort(np.percentile(raw[:, j], qs))
        # np.interp needs a strictly increasing x; nudge ties apart.
        col = np.maximum.accumulate(col + np.arange(len(col)) * 1e-12)
        percentiles[f] = col

    w = _weights(raw)
    cal = Calibration(
        name=name,
        percentiles=percentiles,
        pca_mean=np.zeros(len(fields)),
        pca_components=np.eye(len(fields))[:2],
        pca_explained=np.array([0.5, 0.5]),
        fields=tuple(fields),
        weights=w,
        meta={"n_signals": len(fps)},
    )

    # Fit the PCA on *normalised, weighted* vectors so every axis is comparable
    # and degenerate descriptors cannot dominate.
    norm = np.array([fp.normalized(cal, fields) for fp in fps], float)
    mean = norm.mean(axis=0)
    X = (norm - mean) * w
    # Covariance eigendecomposition; no sklearn dependency.
    cov = np.cov(X, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    total = vals.sum()
    cal.pca_mean = mean
    cal.pca_components = vecs[:, :2].T
    cal.pca_explained = (vals[:2] / total) if total > 0 else np.array([0.5, 0.5])
    cal.meta["pca_explained_2d"] = float(cal.pca_explained.sum())
    cal.meta["weights"] = {f: round(float(x), 4) for f, x in zip(fields, w)}

    # The projection's scale depends on the corpus; store the radius
    # distribution so `project` magnitudes can be read as percentiles rather
    # than raw distances (which would differ per corpus and break temperament).
    proj = np.array([cal.pca_components @ ((v - mean) * w) for v in norm])
    radii = np.hypot(proj[:, 0], proj[:, 1])
    cal.percentiles["__radius__"] = np.maximum.accumulate(
        np.sort(np.percentile(radii, np.linspace(0, 100, N_QUANTILES)))
        + np.arange(N_QUANTILES) * 1e-12
    )
    return cal


_CACHE = {}


def list_calibrations():
    """Names of the shipped calibrations."""
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted(
        f[:-4] for f in os.listdir(DATA_DIR) if f.endswith(".npz")
    )


def load_calibration(name="eeg_sleep_v1"):
    """Load a shipped calibration by name. ``"none"`` gives the identity."""
    if name in (None, "none", "identity"):
        return Calibration.identity()
    if isinstance(name, Calibration):
        return name
    if name in _CACHE:
        return _CACHE[name]
    path = name if os.path.isfile(str(name)) else os.path.join(DATA_DIR, f"{name}.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"no calibration {name!r} at {path}. Available: {list_calibrations()} "
            f"(or use 'none')"
        )
    cal = Calibration.load(path)
    _CACHE[name] = cal
    return cal
