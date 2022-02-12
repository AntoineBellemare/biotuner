
# Biotuner
Python toolbox that incorporates tools from biological signal processing and musical computation to transform biosignals into microtonal musical structures

![Biotuner_pipeline (6)-page-001](https://user-images.githubusercontent.com/49297774/153693263-90c1e49e-a8c0-4a93-8219-491d1ede32e1.jpg)

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
