
# Biotuner
Python toolbox that incorporates tools from biological signal processing and musical computation to transform biosignals into microtonal musical structures

![0001](https://user-images.githubusercontent.com/49297774/118917915-e7334400-b8ff-11eb-9bfd-e647c4500b33.jpg)

# Installation

```python
pip install biotuner
```

# Minimal usecase

```python
biotuning = biotuner(sf = 1000)
biotuning.peaks_extraction(data)
```

## PyTuning issue

If you have a problem with the PyTuning library when it uses Sympy functions, you could replace the file 'constant.py' in 
'C:\Users\[xxx]\Anaconda3\envs\[xxx]\Lib\site-packages\pytuning'
