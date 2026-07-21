"""Frozen original ``bioelements`` API, preserved for back-compatibility.

This is the pre-package flat module. New code should use the package API
(:mod:`biotuner.bioelements.units`, ``.tables``, ``.spectrum``, ``.matching``,
``.composition``, ``.materials``). These names are re-exported from the package
``__init__`` so existing imports keep working.

The one substantive change from the original: ``compute_ratios_df`` used
``DataFrame.append`` (removed in pandas 2.0) and crashed; it is rewritten with
``pd.concat`` here so the frozen API still runs.
"""
import math
from fractions import Fraction

import numpy as np
import pandas as pd

from biotuner.bioelements.units import (  # re-exported for back-compat
    nm_to_hertz, hertz_to_nm, hertz_to_volt, SPECTRUM_NM as spectrum_nm,
)


def Angstrom_to_hertz(wavelength_in_Angstrom):
    c = 2.998e8
    return c / (wavelength_in_Angstrom * 1e-10)


spectrum_Angstrom = {k: [v[0] * 10, v[1] * 10] for k, v in spectrum_nm.items()}
spectrum_hertz = {k: [nm_to_hertz(v[0]), nm_to_hertz(v[1])] for k, v in spectrum_nm.items()}
spectrum_volt = {k: [hertz_to_volt(v[0]), hertz_to_volt(v[1])] for k, v in spectrum_hertz.items()}


def spectrum_region(wavelength):
    for region, (min_wl, max_wl) in spectrum_nm.items():
        if min_wl * 10 <= wavelength <= max_wl * 10:
            return region
    return "Unknown"


def find_matching_spectral_lines(df, peaks, tolerance=1e-9, max_divisions=10):
    """Original absolute-tolerance matcher (deprecated).

    Kept for back-compat; prefer :func:`biotuner.bioelements.matching.match_lines`,
    which uses a relative (cents/ppm) tolerance. An absolute Å tolerance is
    meaningless across a table spanning 56–46 525 Å.
    """
    min_wl, max_wl = df['wavelength'].min(), df['wavelength'].max()
    divided_peaks = []
    for peak in peaks:
        end_i = abs(int(math.floor(np.log2(min_wl / peak))))
        start_i = abs(int(math.ceil(np.log2(max_wl / peak))))
        divided_peaks.extend([peak / (2 ** i) for i in range(start_i, end_i + 1)])
    wavelengths = df['wavelength'].values
    frames = []
    for divided_peak in divided_peaks:
        matches = np.abs(wavelengths - divided_peak) <= tolerance
        if np.any(matches):
            temp_df = df[matches].copy()
            temp_df['peak_value'] = divided_peak
            frames.append(temp_df)
    if not frames:
        return pd.DataFrame(columns=df.columns.to_list() + ['peak_value'])
    return pd.concat(frames, ignore_index=True)


def plot_type_proportions(df):
    import matplotlib.pyplot as plt
    type_counts = df['type'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%')
    ax.set_title('Proportions of bioelements types')
    plt.show()


def compute_ratios_df(df, ratio_type, label_name):
    """Element-internal wavelength ratios (rewritten for pandas ≥ 2.0)."""
    rows = []
    for label, group_df in df.groupby(label_name):
        peak_values = group_df['wavelength'].values
        for i in range(len(peak_values)):
            for j in range(i + 1, len(peak_values)):
                x = int(peak_values[i])
                y = int(peak_values[j])
                if x == 0 or y == 0:
                    continue
                if ratio_type == 'harm':
                    (hi, lo) = (x, y) if x > y else (y, x)
                    rows.append({label_name: label, 'ratio': hi / lo, 'peak1': hi, 'peak2': lo})
                elif ratio_type == 'subharm':
                    (lo, hi) = (x, y) if x < y else (y, x)
                    rows.append({label_name: label, 'ratio': lo / hi, 'peak1': lo, 'peak2': hi})
                elif ratio_type == 'all':
                    rows.append({label_name: label, 'ratio': x / y, 'peak1': x, 'peak2': y})
                    rows.append({label_name: label, 'ratio': y / x, 'peak1': y, 'peak2': x})
    ratios_df = pd.DataFrame(rows, columns=[label_name, 'ratio', 'peak1', 'peak2'])
    merged_df = pd.merge(df, ratios_df, how='left', on=label_name)
    merged_df['temp_ratio'] = merged_df.apply(
        lambda row: Fraction(int(row['peak1']), int(row['peak2'])).limit_denominator(100)
        if pd.notna(row['peak1']) and pd.notna(row['peak2']) and row['peak2'] != 0 else pd.NA,
        axis=1,
    )
    merged_df['ratio1'] = merged_df['temp_ratio'].apply(lambda x: x.numerator if pd.notna(x) else pd.NA)
    merged_df['ratio2'] = merged_df['temp_ratio'].apply(lambda x: x.denominator if pd.notna(x) else pd.NA)
    merged_df.drop('temp_ratio', axis=1, inplace=True)
    return merged_df
