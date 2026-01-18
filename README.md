<p align="center">
  <img src="https://github.com/AntoineBellemare/biotuner/assets/49297774/fc83d888-db2a-4f9f-ba26-65a58c42b72d" alt="biotuner_logo" width="200"/>
</p>

<h1 align="center">Biotuner</h1>
<h3 align="center"> Python toolbox that incorporates tools from biological signal processing and musical theory to extract harmonic structures from biosignals. </h3>

<p align="center">
  <a href="https://github.com/AntoineBellemare/biotuner/actions/workflows/ci.yml">
    <img alt="Tests" src="https://github.com/AntoineBellemare/biotuner/actions/workflows/python-test.yml/badge.svg">
  </a>

  <a href="https://codecov.io/github/AntoineBellemare/biotuner">
    <img alt="Codecov" src="https://codecov.io/github/AntoineBellemare/biotuner/branch/main/graph/badge.svg?token=DW8JS03EV9">
  </a>

  <a href="https://pypi.org/project/biotuner/">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/biotuner">
  </a>
  
  <a href="https://antoinebellemare.github.io/biotuner/">
    <img alt="Biotuner Docs" src="https://img.shields.io/website?label=Docs&up_color=blue&url=https%3A%2F%2Fantoinebellemare.github.io%2Fbiotuner%2F">
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

## ✨ Features

- **🎵 Harmonic Analysis**: Extract harmonic structures from biosignals using music theory principles
- **📊 Multiple Peak Detection Methods**: FOOOF, EMD, fixed-frequency, and harmonic-recurrence based methods
- **🧮 Harmonicity Metrics**: Compute consonance, dissonance, harmonic similarity, Tenney height, and more
- **🎹 Musical Applications**: Generate musical scales, tuning systems, and MIDI output from biosignals
- **🔬 Group Analysis (BETA)**: Batch processing for multiple time series with automatic aggregation
- **📈 Rich Visualizations**: Publication-ready plots for spectral analysis and harmonic relationships
- **🧠 Multi-modal Support**: Compatible with EEG, ECG, EMG, plant signals, and other biosignals
- **🎨 Interactive GUI**: Graphical interface for easy exploration

<!-- 🧬![Biotuner](https://img.shields.io/badge/Biotuner-Documentation-blue?style=for-the-badge&logo=bookstack) 🎹 -->


# **Installation**

## **1. Install using PyPI (Recommended)**
To install the latest stable version of **Biotuner** from PyPI, run:
```bash
pip install biotuner
```

---

## **2. Install from the GitHub Repository (Development Version)**
If you want the latest development version or contribute to the code, follow these steps:

### **2.1. Automatically Setup the Environment (Recommended)**
The easiest way to set up a development environment is by using `invoke`, which will:

✅ Create a **Conda environment**  
✅ Install **dependencies**  
✅ Install **Biotuner in editable mode**  

```bash
# Clone the repository
git clone https://github.com/AntoineBellemare/biotuner.git
cd biotuner

# Install Invoke (if not already installed)
pip install invoke

# Automatically create a Conda environment and install Biotuner
invoke setup
```
👉 This will create a Conda environment named `biotuner_env` and install all dependencies.

To activate the Conda environment manually:
```bash
conda activate biotuner_env
```

---

### **2.2. Manual Setup (Alternative)**
If you prefer to set up the environment manually, follow these steps:

#### **1️⃣ Create a Conda environment**
```bash
conda create --name biotuner_env python=3.11 -y
conda activate biotuner_env
```

#### **2️⃣ Install dependencies**
```bash
pip install -r requirements.txt
pip install -e .
```

---

## **3. Verify Installation by Running Tests**
To confirm that Biotuner is installed correctly, run the test suite:
```bash
invoke test
```
or manually using:
```bash
pytest tests/
```
If all tests pass ✅, your installation is complete!

---

### **🎯 Summary**
- **For general users**: Install via `pip install biotuner`
- **For development**: Clone the repo and run `invoke setup`
- **To verify installation**: Run `invoke test`

# Simple use case

## Single Time Series Analysis

```python
from biotuner import biotuner

# Initialize the object
biotuning = biotuner(sf=1000)

# Extract spectral peaks
biotuning.peaks_extraction(data, peaks_function='FOOOF')

# Get consonance metrics for spectral peaks
biotuning.compute_peaks_metrics()
```

## Group Analysis (🧪 BETA)

Analyze multiple time series simultaneously with automatic aggregation and group comparisons:

```python
from biotuner import BiotunerGroup
import numpy as np

# Multiple trials or electrodes: shape (n_series, n_samples)
data = np.random.randn(10, 5000)

# Create group object
btg = BiotunerGroup(data, sf=1000, axis_labels=['trials'])

# Run analysis pipeline
btg.compute_peaks(peaks_function='FOOOF', min_freq=1, max_freq=50)
btg.compute_metrics(n_harm=10)

# Get summary statistics
summary = btg.summary()
```

> **Note:** The BiotunerGroup module is currently in beta. The API may change in future releases.

---

## 🌐 Biotuner Engine - Web Interface

Explore Biotuner's capabilities through our interactive web interface:

**[biotuner-engine.kairos-hive.org](https://biotuner-engine.kairos-hive.org)**

The Biotuner Engine provides a user-friendly web application to analyze biosignals, visualize harmonic structures, and explore musical applications directly in your browser—no installation required!

---

<div align="center" style="width: 50%; margin: auto; text-align: center;">

<h1 align="center">Multimodal Harmonic Analysis</h1>

  <p>
    <img src="https://github.com/user-attachments/assets/7e99e0ec-a1da-44f2-8ad9-bdfce8f4a36f" alt="biotuner_multimodal_02" width="50%">
  </p>

The figure above illustrates Biotuner's ability to extract harmonic structures across different biological and physical systems. It showcases harmonic ratios detected in biosignals from the **brain**, **heart**, and **plants**, as well as their correspondence with audio signals. By analyzing the fundamental frequency relationships in these diverse modalities, Biotuner enables a cross-domain exploration of resonance and tuning in biological and artificial systems.

</div>

![Biotuner_pipeline (6)-page-001](https://user-images.githubusercontent.com/49297774/153693263-90c1e49e-a8c0-4a93-8219-491d1ede32e1.jpg)

## Peaks extraction methods

![biotuner_peaks_extraction](https://user-images.githubusercontent.com/49297774/156813349-ddcd40d0-57c9-41f2-b62a-7cbb4213e515.jpg)

---

## 📚 Documentation & Resources

- **[Full Documentation](https://antoinebellemare.github.io/biotuner/)** - Complete API reference and tutorials
- **[Getting Started Guide](https://antoinebellemare.github.io/biotuner/getting_started.html)** - Step-by-step introduction
- **[API Reference](https://antoinebellemare.github.io/biotuner/api/index.html)** - Detailed function and class documentation
  - [BiotunerObject](https://antoinebellemare.github.io/biotuner/api/biotuner_object.html) - Single time series analysis
  - [BiotunerGroup (BETA)](https://antoinebellemare.github.io/biotuner/api/biotuner_group.html) - Group analysis
  - [Metrics](https://antoinebellemare.github.io/biotuner/api/metrics.html) - Harmonicity metrics
  - [Peak Extraction](https://antoinebellemare.github.io/biotuner/api/peaks_extraction.html) - Peak detection methods
- **[Examples & Notebooks](https://antoinebellemare.github.io/biotuner/examples/index.html)** - Jupyter notebook tutorials

## 🤝 Contributing

We welcome contributions! Whether it's:

- 🐛 Bug reports
- 💡 Feature requests
- 📝 Documentation improvements
- 🔧 Code contributions

Please feel free to open an issue or submit a pull request on [GitHub](https://github.com/AntoineBellemare/biotuner).

## 📄 License

Biotuner is licensed under the [MIT License](LICENSE.txt).

## 📖 Citation

If you use Biotuner in your research, please cite our work. See the [citation guide](https://antoinebellemare.github.io/biotuner/cite_us.html) for more information.

## 💬 Support

- **Issues**: [GitHub Issues](https://github.com/AntoineBellemare/biotuner/issues)
- **Email**: antoine.bellemare9@gmail.com
- **Documentation**: [https://antoinebellemare.github.io/biotuner/](https://antoinebellemare.github.io/biotuner/)

---

<p align="center">
  Made with ❤️ by the Biotuner development team
</p>
