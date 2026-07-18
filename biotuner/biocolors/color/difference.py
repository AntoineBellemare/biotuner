"""Perceptual colour difference and colour-vision-deficiency simulation.

Module type: Functions

These turn "this palette looks good" into a measured claim. ``deltaE_ok`` is
plain Euclidean distance in OKLab -- that is the entire point of a perceptually
uniform space, and it is why the module works in OKLab rather than HSV.

Rough interpretation of ``deltaE_ok`` between two colours:

- < 0.02  imperceptible side by side
- < 0.05  "basically the same colour"
- ~0.10   comfortably distinguishable
- > 0.20  obviously different

CVD matrices from Machado, Oliveira & Fernandes (2009), applied in linear RGB.
"""

from __future__ import annotations

import numpy as np

from biotuner.biocolors.color.spaces import (
    linear_to_srgb,
    srgb_to_linear,
    srgb_to_oklab,
)

__all__ = ["deltaE_ok", "pairwise_deltaE", "simulate_cvd", "CVD_KINDS"]

CVD_KINDS = ("protan", "deutan", "tritan")

# Severity 1.0 matrices, linear RGB.
_CVD = {
    "protan": np.array([
        [0.152286, 1.052583, -0.204868],
        [0.114503, 0.786281, 0.099216],
        [-0.003882, -0.048116, 1.051998],
    ]),
    "deutan": np.array([
        [0.367322, 0.860646, -0.227968],
        [0.280085, 0.672501, 0.047413],
        [-0.011820, 0.042940, 0.968881],
    ]),
    "tritan": np.array([
        [1.255528, -0.076749, -0.178779],
        [-0.078411, 0.930809, 0.147602],
        [0.004733, 0.691367, 0.303900],
    ]),
}


def deltaE_ok(lab_a, lab_b):
    """Euclidean distance in OKLab. Inputs ``(..., 3)``, output ``(...,)``."""
    a = np.asarray(lab_a, float)
    b = np.asarray(lab_b, float)
    return np.linalg.norm(a - b, axis=-1)


def pairwise_deltaE(lab):
    """Full pairwise OKLab distance matrix for ``(N, 3)`` colours -> ``(N, N)``."""
    lab = np.asarray(lab, float)
    return np.linalg.norm(lab[:, None, :] - lab[None, :, :], axis=-1)


def simulate_cvd(rgb, kind="deutan", severity=1.0):
    """Simulate colour vision deficiency on sRGB (0-1) input, shape ``(..., 3)``.

    ``kind`` in :data:`CVD_KINDS`. ``severity`` linearly interpolates between
    normal vision (0) and full dichromacy (1).
    """
    if kind not in _CVD:
        raise ValueError(f"unknown CVD kind: {kind!r}; expected one of {CVD_KINDS}")
    m = _CVD[kind]
    if severity != 1.0:
        m = (1.0 - severity) * np.eye(3) + severity * m
    lin = srgb_to_linear(np.clip(np.asarray(rgb, float), 0.0, 1.0))
    out = np.einsum("ij,...j->...i", m, lin)
    return linear_to_srgb(np.clip(out, 0.0, 1.0), clip=True)


def min_separation(rgb, under_cvd=True):
    """Smallest pairwise OKLab distance in a palette, optionally worst-case CVD.

    Returns a dict: ``normal`` plus one entry per CVD kind, and ``worst``.
    A palette whose ``worst`` is below ~0.05 has two swatches that some readers
    cannot tell apart.
    """
    rgb = np.asarray(rgb, float).reshape(-1, 3)
    if len(rgb) < 2:
        return {"normal": float("inf"), "worst": float("inf")}

    def _min(x):
        d = pairwise_deltaE(srgb_to_oklab(x))
        iu = np.triu_indices(len(x), k=1)
        return float(d[iu].min())

    out = {"normal": _min(rgb)}
    if under_cvd:
        for k in CVD_KINDS:
            out[k] = _min(simulate_cvd(rgb, kind=k))
    out["worst"] = float(min(out.values()))
    return out
