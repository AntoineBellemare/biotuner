"""Unit conversions and electromagnetic-spectrum regions for bioelements.

Every conversion is vectorised (accepts scalars or arrays) and unit-explicit in
its name. Wavelengths are in angstrom (Å) throughout the package, to match the
NIST line tables; helpers convert to/from Hz, nm and eV.
"""
from __future__ import annotations

import numpy as np

# --- physical constants (SI) --------------------------------------------- #
C_M_PER_S = 2.99792458e8          # speed of light, m/s
C_ANGSTROM_PER_S = C_M_PER_S * 1e10   # speed of light, Å/s
H_PLANCK = 6.62607015e-34         # Planck constant, J·s
E_CHARGE = 1.602176634e-19        # elementary charge, C


def angstrom_to_hertz(wavelength_angstrom):
    """Å -> Hz (electromagnetic frequency of that wavelength)."""
    return C_ANGSTROM_PER_S / np.asarray(wavelength_angstrom, float)


def hertz_to_angstrom(frequency_hz):
    """Hz -> Å."""
    return C_ANGSTROM_PER_S / np.asarray(frequency_hz, float)


def nm_to_hertz(wavelength_nm):
    """nm -> Hz."""
    return C_M_PER_S / (np.asarray(wavelength_nm, float) * 1e-9)


def hertz_to_nm(frequency_hz):
    """Hz -> nm."""
    return C_M_PER_S / np.asarray(frequency_hz, float) * 1e9


def angstrom_to_nm(wavelength_angstrom):
    return np.asarray(wavelength_angstrom, float) / 10.0


def nm_to_angstrom(wavelength_nm):
    return np.asarray(wavelength_nm, float) * 10.0


def hertz_to_volt(frequency_hz):
    """Hz -> eV-equivalent voltage (photon energy / electron charge)."""
    return H_PLANCK * np.asarray(frequency_hz, float) / E_CHARGE


# --- electromagnetic spectrum regions ------------------------------------ #
#: Wavelength band edges (nm) for each named EM region.
SPECTRUM_NM = {
    "Gamma rays": (0.01, 0.1),
    "X-rays": (0.1, 10),
    "Ultraviolet": (10, 400),
    "Visible light": (400, 700),
    "Infrared": (700, 1_000_000),
    "Microwaves": (1_000_000, 1_000_000_000),
    "Radio waves": (1_000_000_000, 100_000_000_000),
}

#: Same bands in angstrom (the table's native unit).
SPECTRUM_ANGSTROM = {k: (v[0] * 10, v[1] * 10) for k, v in SPECTRUM_NM.items()}


def spectrum_region(wavelength_angstrom):
    """Name the EM region of a wavelength given in angstrom."""
    w = float(wavelength_angstrom)
    for region, (lo, hi) in SPECTRUM_ANGSTROM.items():
        if lo <= w <= hi:
            return region
    return "Unknown"


# Optical band used for octave-folding biosignal peaks into the visible/near band.
OPTICAL_BAND_ANGSTROM = (3000.0, 7000.0)


def fold_to_optical(frequency_or_wavelength, *, is_hz=True, band=OPTICAL_BAND_ANGSTROM):
    """Octave-transpose a value into the optical wavelength band [lo, hi] Å.

    Folding is done on **wavelength** (halving frequency doubles wavelength), so
    an octave in pitch is an octave in wavelength — the same octave-equivalence
    the rest of biotuner uses. Returns the folded wavelength in angstrom.
    """
    lo, hi = band
    wl = hertz_to_angstrom(frequency_or_wavelength) if is_hz else float(frequency_or_wavelength)
    wl = float(wl)
    # A wavelength above the band is too low a frequency -> halve wavelength (up an octave).
    while wl > hi:
        wl /= 2.0
    while wl < lo:
        wl *= 2.0
    return wl
