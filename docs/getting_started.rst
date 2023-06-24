Getting started with Biotuner
#############################

Install using conda and pip
----------------------------

.. code-block:: bash
    
    conda create -n biotuner python=3.8
    conda activate biotuner
    pip install biotuner

Install from source
-------------------

.. code-block:: bash

    git clone git@github.com:AntoineBellemare/biotuner.git
    cd biotuner
    pip install -r requirements.txt
    python setup.py install

If you have issues with the installation, please check your Python and pip versions, or contact us.


Quick Start
-----------
The Biotuner can be used with any type of biosignal data. The following example shows how to use the Biotuner with a simple time series.

Example:
^^^^^^^^

.. code-block:: python

   from biotuner.biotuner_object import compute_biotuner
   data = np.random.rand(5000) # generate some random data 
   biotuning = compute_biotuner(sf = 1000) # initialize the object
   biotuning.peaks_extraction(data, peaks_function='FOOOF') # extract spectral peaks
   biotuning.compute_peaks_metrics() # get consonance metrics for spectral peaks
