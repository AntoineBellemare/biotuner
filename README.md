
# Biotuner
Python toolbox that incorporates tools from biological signal processing and musical theory to extract harmonic structures from biosignals.


Visit the [documentation page](https://sangfrois.github.io/biotuner/)
![Biotuner_pipeline (6)-page-001](https://user-images.githubusercontent.com/49297774/153693263-90c1e49e-a8c0-4a93-8219-491d1ede32e1.jpg)

# Installation

Create an environment with Python v3.8
```python
pip install biotuner
```

# Simple use case

```python
biotuning = biotuner(sf = 1000) #initialize the object
biotuning.peaks_extraction(data, peaks_function='FOOOF') #extract spectral peaks
biotuning.compute_peaks_metrics() #get consonance metrics for spectral peaks

```

## Peaks extraction methods

![biotuner_peaks_extraction](https://user-images.githubusercontent.com/49297774/156813349-ddcd40d0-57c9-41f2-b62a-7cbb4213e515.jpg)
