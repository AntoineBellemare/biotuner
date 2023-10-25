import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from biotuner.scale_construction import tuning_reduction
from biotuner.biotuner_utils import compute_peak_ratios, create_SCL
from biotuner.bioelements import Angstrom_to_hertz
from biotuner.metrics import dyad_similarity, metric_denom

# import the data
data = pd.read_csv('../data/air_elements.csv')

n = 50
# Use groupby and nlargest to retrieve the top n wavelengths for each element based on intensity
top_n = data.groupby('element').apply(lambda x: x.nlargest(n, 'intensity')).reset_index(drop=True)
top_n['Hz'] = top_n['wavelength'].apply(Angstrom_to_hertz)

# Create a dictionary with the element as the key and the result of compute_peak_ratios as the value
result_dict = {}
n_steps = 7
for element in top_n['element'].unique():
    hertz_values = top_n[top_n['element'] == element]['Hz'].tolist()
    result_dict[element] = compute_peak_ratios(hertz_values, rebound=True)
    # round the values to 2 decimal places
    result_dict[element] = [round(x, 2) for x in result_dict[element]]
    #print(result_dict[element])
    try:
        _, mode, _ = tuning_reduction(result_dict[element], n_steps, function=dyad_similarity)
        mode = sorted(mode)
        if 2.0 not in mode:
            # add 2.0 at the last index to create a 7 note scale
            mode.append(2.0)
    except IndexError:
        print(f'IndexError for {element}')
        continue
    create_SCL(mode, f'../data/elemental_tunings/{element}_ratios_{n_steps}steps')
    
