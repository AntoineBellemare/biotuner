.. biotuner documentation master file, created by
   sphinx-quickstart on Tue Feb 28 12:31:54 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

API Reference
====================================

Complete API documentation for harmonic analysis of biosignals.

Core Objects
------------

Main classes for single and batch time series analysis.

.. toctree::
   :maxdepth: 2

   biotuner_object
   biotuner_group

Signal Processing Functions
----------------------------

Methods for spectral peak detection and extension (neuroscience/signal processing).

.. toctree::
   :maxdepth: 2

   peaks_extraction
   peaks_extension

Musical Analysis Functions
---------------------------

Music theory and harmonic metrics computation.

.. toctree::
   :maxdepth: 2

   metrics
   scale_construction
   rhythm_construction

Extended Analysis Objects
--------------------------

Advanced objects building on biotuner_object for multi-dimensional analysis.

* **harmonic_spectrum** - Analysis across frequencies
* **harmonic_connectivity** - Analysis across space/sensors  
* **transitional_harmony** - Analysis across time

.. toctree::
   :maxdepth: 2

   harmonic_spectrum
   harmonic_connectivity
   transitional_harmony

Utilities
---------

.. toctree::
   :maxdepth: 2

   utils
