
# Biotuner 🧬🎵
🌳🧠 Python toolbox that incorporates tools from biological signal processing and musical theory to extract harmonic structures from biosignals. 🧠🌳

🔗 Visit the [documentation page](https://sangfrois.github.io/biotuner/)

![Biotuner_pipeline (6)-page-001](https://user-images.githubusercontent.com/49297774/153693263-90c1e49e-a8c0-4a93-8219-491d1ede32e1.jpg)

# Installation

To install Biotuner, follow the steps below:

1. **Create a Python environment with v3.8 using conda**:
```bash
conda create --name biotuner_env python=3.8
conda activate biotuner_env
```

2. Clone the Biotuner repository:
```
git clone https://github.com/AntoineBellemare/biotuner.git
cd biotuner
```
3. Install the package:
```
pip install .
```

# Simple use case

```python
biotuning = biotuner(sf = 1000) #initialize the object
biotuning.peaks_extraction(data, peaks_function='FOOOF') #extract spectral peaks
biotuning.compute_peaks_metrics() #get consonance metrics for spectral peaks

```

## Peaks extraction methods

![biotuner_peaks_extraction](https://user-images.githubusercontent.com/49297774/156813349-ddcd40d0-57c9-41f2-b62a-7cbb4213e515.jpg)
