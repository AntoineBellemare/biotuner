Harmonic spectrum
=================

Per-frequency harmonicity H(f) on a single signal. This is the **H-only entry
point** of biotuner's spectral resonance machinery.

Quick start
-----------

::

   from biotuner.harmonic_spectrum import compute_harmonic_spectrum

   freqs, H, S, summary = compute_harmonic_spectrum(
       signal, precision_hz=0.5, fmin=2, fmax=30, fs=1000,
   )
   # H       — per-frequency harmonicity spectrum  (n_freqs,)
   # S       — N×N kernel similarity matrix
   # summary — dict of complexity metrics (flatness, entropy, higuchi, ...)

Sister modules
--------------

For the full **H × PC = R** pipeline (with surrogate normalization, swappable
kernels, and higher-order coupling extensions), use
:func:`biotuner.resonance.compute_resonance`.

For **cross-channel** analyses (two-or-more signals), use
:mod:`biotuner.harmonic_connectivity`.

Public API
----------

* :func:`compute_harmonic_spectrum` — main entry point
* :func:`harmonicity_matrices` — N×N similarity matrix (legacy helper)
* :func:`compute_harmonic_power` — probability-weighted reduction of S to H(f)
* :func:`find_spectral_peaks` — peak detector (shared across the resonance package)
* :func:`harmonic_entropy` — complexity DataFrame for H / PC / R together
* :func:`get_harmonic_ratio` — best (n, m) within tolerance for a freq pair
* :func:`count_theoretical_harmonic_partners` — bandwidth-correction helper

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   biotuner.harmonic_spectrum

.. automodule:: biotuner.harmonic_spectrum
   :members:
