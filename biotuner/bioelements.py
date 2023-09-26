import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def Angstrom_to_hertz(wavelength_in_Angstrom):
    c = 2.998e+8 # speed of light in m/s
    wavelength_in_meter = wavelength_in_Angstrom * 1e-10
    frequency = c / wavelength_in_meter
    return frequency


def nm_to_hertz(wavelength_in_nm):
    c = 2.998e+8 # speed of light in m/s
    wavelength_in_meter = wavelength_in_nm * 1e-9
    frequency = c / wavelength_in_meter
    return frequency


def hertz_to_nm(frequency_in_hertz):
    c = 2.998e+8 # speed of light in m/s
    wavelength_in_meter = c / frequency_in_hertz
    wavelength_in_nm = wavelength_in_meter * 1e9
    return wavelength_in_nm


def hertz_to_volt(frequency_in_hertz):
    h = 6.62607015e-34 # Planck's constant in J.s
    e = 1.60217662e-19 # electron charge in C
    voltage = h*frequency_in_hertz/e
    return voltage

def find_matching_spectral_lines(df, peaks, tolerance=1e-9, max_divisions=10):
    min_wl, max_wl = df['wavelength'].min(), df['wavelength'].max()
     
    # calculate all divisions once for each peak
    divided_peaks = []
    for peak in peaks:
        end_i = abs(int(math.floor(np.log2(min_wl / peak))))
        start_i = abs(int(math.ceil(np.log2(max_wl / peak))))
        divided_peaks.extend([peak / (2**i) for i in range(start_i, end_i + 1)])

    # create a new numpy array for faster computation
    wavelengths = df['wavelength'].values

    # create a new dataframe with column 'peak_value' where we find matches
    matching_df = pd.DataFrame(columns=df.columns.to_list() + ['peak_value'])

    # iterate over the divided peaks
    for divided_peak in divided_peaks:
        # find where the absolute difference between wavelength and divided peak is less than tolerance
        matches = np.abs(wavelengths - divided_peak) <= tolerance

        if np.any(matches):
            temp_df = df[matches].copy()
            temp_df['peak_value'] = divided_peak
            matching_df = pd.concat([matching_df, temp_df])

    return matching_df

def plot_type_proportions(df):
    type_counts = df['type'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(type_counts, labels = type_counts.index, autopct='%1.1f%%')
    ax.set_title('Proportions of bioelements types')
    plt.show()
    
def spectrum_region(wavelength):
    for region, (min_wl, max_wl) in spectrum_nm.items():
        if min_wl*10 <= wavelength <= max_wl*10:
            return region
    return "Unknown"

spectrum_nm = {
    'Gamma rays': [0.01, 0.1],
    'X-rays': [0.1, 10],
    'Ultraviolet': [10, 400],
    'Visible light': [400, 700],
    'Infrared': [700, 1_000_000],
    'Microwaves': [1_000_000, 1_000_000_000],
    'Radio waves': [1_000_000_000, 100_000_000_000]
}

spectrum_Angstrom = {k: [v[0]*10, v[1]*10] for k, v in spectrum_nm.items()}

spectrum_hertz = {k: [nm_to_hertz(v[0]), nm_to_hertz(v[1])] for k, v in spectrum_nm.items()}

spectrum_volt = {k: [hertz_to_volt(v[0]), hertz_to_volt(v[1])] for k, v in spectrum_hertz.items()}

import pandas as pd
from fractions import Fraction

def compute_ratios_df(df, ratio_type, label_name):
    # Create an empty DataFrame to store the results
    ratios_df = pd.DataFrame(columns=[label_name, 'ratio', 'peak1', 'peak2'])
    
    # Group by the label and compute the ratios within each group
    for label, group_df in df.groupby(label_name):
        peak_values = group_df['wavelength'].values

        for i in range(len(peak_values)):
            for j in range(i + 1, len(peak_values)):
                x = int(peak_values[i])
                y = int(peak_values[j])
                
                if ratio_type == 'harm':
                    if x > y:
                        ratio = x / y
                        ratios_df = ratios_df.append({label_name: label, 'ratio': ratio, 'peak1': x, 'peak2': y}, ignore_index=True)
                    else:
                        ratio = y / x
                        ratios_df = ratios_df.append({label_name: label, 'ratio': ratio, 'peak1': y, 'peak2': x}, ignore_index=True)

                elif ratio_type == 'subharm':
                    if x < y:
                        ratio = x / y
                        ratios_df = ratios_df.append({label_name: label, 'ratio': ratio, 'peak1': x, 'peak2': y}, ignore_index=True)
                    else:
                        ratio = y / x
                        ratios_df = ratios_df.append({label_name: label, 'ratio': ratio, 'peak1': y, 'peak2': x}, ignore_index=True)
                        
                elif ratio_type == 'all':
                    ratio = x / y
                    ratios_df = ratios_df.append({label_name: label, 'ratio': ratio, 'peak1': x, 'peak2': y}, ignore_index=True)
                    ratio = y / x  # y/x ratio
                    ratios_df = ratios_df.append({label_name: label, 'ratio': ratio, 'peak1': y, 'peak2': x}, ignore_index=True)
    
    # Merge the ratios DataFrame with the original DataFrame based on the label_name
    merged_df = pd.merge(df, ratios_df, how='left', on=label_name)
    ratio_df = merged_df.copy()
    # Create 'ratio1' and 'ratio2' columns
    ratio_df['temp_ratio'] = ratio_df.apply(lambda row: Fraction(row['peak1'], row['peak2']).limit_denominator(100), axis=1)

    # Extract numerator and denominator
    ratio_df['ratio1'] = ratio_df['temp_ratio'].apply(lambda x: x.numerator)
    ratio_df['ratio2'] = ratio_df['temp_ratio'].apply(lambda x: x.denominator)

    # Drop the temporary column
    ratio_df.drop('temp_ratio', axis=1, inplace=True)
    return ratio_df
