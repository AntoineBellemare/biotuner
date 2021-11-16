
# Biotuner
Python toolbox that incorporates tools from biological signal processing and musical computation to transform biosignals into microtonal musical structures

![Biotuner_pipeline](https://user-images.githubusercontent.com/49297774/141891464-70781440-c98e-4385-887c-9101b098b851.png)

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
