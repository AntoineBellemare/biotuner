
# Biotuner
Python toolbox that incorporates tools from biological signal processing and musical computation to transform biosignals into microtonal musical structures

[Biotuner_pipeline (6).pdf](https://github.com/AntoineBellemare/biotuner/files/8052560/Biotuner_pipeline.6.pdf)

# Installation

```python
pip install biotuner
```

# Simple use case

```python
biotuning = biotuner(sf = 1000) #initialize the object
biotuning.peaks_extraction(data) #extract spectral peaks
biotuning.compute_peaks_metrics() #get consonance metrics for spectral peaks

```

## PyTuning issue

If you have a problem with the PyTuning library when it uses Sympy functions, you could replace the file 'constant.py' in 
'C:\Users\[xxx]\Anaconda3\envs\[xxx]\Lib\site-packages\pytuning' with the one found in the 'examples' folder
