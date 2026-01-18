BiotunerGroup (BETA)
=====================

.. warning::
   
   **🧪 BETA Feature**
   
   The BiotunerGroup module is currently in beta. The API may change in future releases.
   We welcome feedback and contributions to help stabilize this feature.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   biotuner.biotuner_group

.. automodule:: biotuner.biotuner_group
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

``BiotunerGroup`` is a powerful class for batch processing multiple time series with the Biotuner framework. 
It extends the single-object ``compute_biotuner`` to handle multi-dimensional datasets with automatic 
aggregation, group comparisons, and unified visualizations.

Key Features
~~~~~~~~~~~~

✅ **Batch Processing** - Run any biotuner method on multiple time series simultaneously  
✅ **Automatic Aggregation** - Summary statistics computed across all series  
✅ **Metadata Support** - Track experimental conditions, subjects, electrodes, etc.  
✅ **Group Comparisons** - Statistical testing between conditions  
✅ **Unified Visualizations** - Consistent, publication-ready plots  
✅ **Method Chaining** - Clean, Pythonic API  
✅ **Memory Efficient** - Option to store only summaries for large datasets  
✅ **Individual Access** - Can still analyze individual biotuner objects  

Quick Start
-----------

Basic 2D Example
~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from biotuner import BiotunerGroup

    # Your data: 10 trials × 5000 samples
    data = np.random.randn(10, 5000)

    # Create group object
    btg = BiotunerGroup(
        data=data,
        sf=1000,  # Sampling frequency
        axis_labels=['trials']
    )

    # Run analysis pipeline (method chaining!)
    btg.compute_peaks(peaks_function='FOOOF', min_freq=1, max_freq=50)
    btg.compute_metrics(n_harm=10)

    # Get summary statistics
    summary = btg.summary()
    print(summary.head())

3D Example with Metadata
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Multiple electrodes across trials: 20 trials × 64 channels × 5000 samples
    data = np.random.randn(20, 64, 5000)
    
    metadata = {
        'condition': ['rest']*10 + ['task']*10
    }
    
    btg = BiotunerGroup(
        data, 
        sf=1000, 
        axis_labels=['trials', 'electrodes'], 
        metadata=metadata
    )
    
    # Compute peaks and metrics
    btg.compute_peaks(peaks_function='EMD')
    btg.compute_metrics()
    
    # Compare groups
    comparison = btg.compare_groups('condition', metrics=['harm_sim', 'tenney'])
    print(comparison)

Main Methods
------------

Data Processing
~~~~~~~~~~~~~~~

* ``compute_peaks()`` - Extract spectral peaks from all series
* ``compute_extension()`` - Extend peaks to harmonic series
* ``compute_metrics()`` - Calculate harmonicity metrics
* ``compute_diss_curve()`` - Compute dissonance curves
* ``compute_harmonic_entropy()`` - Calculate harmonic entropy
* ``compute_euler_fokker()`` - Compute Euler-Fokker genus metrics
* ``compute_harmonic_tuning()`` - Extract harmonic tuning information

Analysis & Comparison
~~~~~~~~~~~~~~~~~~~~~

* ``summary()`` - Get aggregated statistics across all series
* ``tuning_summary()`` - Get musical tuning parameters
* ``get_tuning_scales()`` - Extract scale information
* ``compare_groups()`` - Statistical comparison between groups
* ``get_attribute()`` - Access specific attributes from all objects

Visualization
~~~~~~~~~~~~~

* ``plot_peaks()`` - Visualize peaks across series
* ``plot_metric_comparison()`` - Compare metrics between groups
* ``plot_euler_fokker()`` - Visualize Euler-Fokker genus
* ``plot_tuning_comparison()`` - Compare tuning across conditions
* ``plot_harmonic_spectrum()`` - Display harmonic spectra

See Also
--------

* :doc:`biotuner_object` - Single time series analysis
* :doc:`metrics` - Available harmonicity metrics
* :doc:`peaks_extraction` - Peak detection methods
