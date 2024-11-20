<p align="center">
  <img src="https://github.com/AntoineBellemare/biotuner/assets/49297774/fc83d888-db2a-4f9f-ba26-65a58c42b72d" alt="biotuner_logo" width="250"/>
</p>

<h1 align="center">Biotuner</h1>
<h3 align="center"> Python toolbox that incorporates tools from biological signal processing and musical theory to extract harmonic structures from biosignals. </h3>

<p align="center">
  <a href="https://pypi.org/project/biotuner/">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/biotuner">
  </a>
  
  <a href="https://antoinebellemare.github.io/biotuner/">
    <img alt="Documentation" src="https://img.shields.io/website?down_message=offline&up_message=online&url=https%3A%2F%2Fantoinebellemare.github.io%2Fbiotuner%2F">
  </a>
  
  <a href="https://github.com/AntoineBellemare/biotuner/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/github/license/AntoineBellemare/biotuner">
  </a>
  
  <a href="https://github.com/AntoineBellemare/biotuner/stargazers">
    <img alt="GitHub stars" src="https://img.shields.io/github/stars/AntoineBellemare/biotuner?style=social">
  </a>

  <a href="https://pypi.org/project/biotuner/">
    <img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/biotuner">
  </a>
</p>

🧬🎵 Visit the [documentation page](https://antoinebellemare.github.io/biotuner/)

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
pip install -e .
```

# Simple use case

```python
biotuning = biotuner(sf = 1000) #initialize the object
biotuning.peaks_extraction(data, peaks_function='FOOOF') #extract spectral peaks
biotuning.compute_peaks_metrics() #get consonance metrics for spectral peaks

```
![Biotuner_pipeline (6)-page-001](https://user-images.githubusercontent.com/49297774/153693263-90c1e49e-a8c0-4a93-8219-491d1ede32e1.jpg)

## Peaks extraction methods

![biotuner_peaks_extraction](https://user-images.githubusercontent.com/49297774/156813349-ddcd40d0-57c9-41f2-b62a-7cbb4213e515.jpg)
