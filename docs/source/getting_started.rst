================================
Getting Started with Biotuner
================================

Overview
========
.. contents::
   :depth: 2

What is Biotuner?
------------------
The Biotuner is a Python toolbox that incorporates tools from biological signal processing
and musical theory to extract harmonic structures from biosignals.

Installation
------------
Installing Biotuner is easy with pip, Python's package manager. You can install it using the following command:

.. code-block:: bash

   pip install biotuner

You can also install Biotuner from source. First, clone the repository:
.. code-block:: bash

   git clone https://github.com/AntoineBellemare/biotuner.git

Then, install the requirements:

.. code-block:: bash

   pip install -r requirements.txt

Finally, install Biotuner:

.. code-block:: bash

   python setup.py install


If you have issues with the installation, please check your Python and pip versions, or contact our support.

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

