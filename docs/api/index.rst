.. biotuner documentation master file, created by
   sphinx-quickstart on Tue Feb 28 12:31:54 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

API Reference
====================================

Complete API documentation for harmonic analysis of biosignals.

The package is organized by *kind* — stateful pipeline classes first, then
subpackages, then pure-function modules. Each module declares its kind in
its docstring header (``Module type: Functions / Object / Objects /
Data / Subpackage``). Not every Python module in the repository has a
documentation page yet — pages are added as modules are cleaned up and
stabilized.

Objects
-------

Stateful pipeline classes — instantiate and call methods.

.. toctree::
   :maxdepth: 2

   biotuner_object
   biotuner_group
   harmonic_connectivity
   transitional_harmony
   harmonic_sequence

Subpackages
-----------

Folders with their own internal structure; each lists its submodules.

.. toctree::
   :maxdepth: 2

   harmonic_geometry

Functions
---------

Pure-function modules — import what you need.

Peak extraction & extension:

.. toctree::
   :maxdepth: 2

   peaks_extraction
   peaks_extension

Scale, rhythm & metrics:

.. toctree::
   :maxdepth: 2

   metrics
   scale_construction
   rhythm_construction

Spectral analysis:

.. toctree::
   :maxdepth: 2

   harmonic_spectrum

Integrations:

.. toctree::
   :maxdepth: 2

   biotuner_mne

Statistics:

.. toctree::
   :maxdepth: 2

   stats
   surrogates

Visualization:

.. toctree::
   :maxdepth: 2

   plot_utils
   plot_config
   harmonic_sequence_viz

Helpers:

.. toctree::
   :maxdepth: 2

   utils
