# Inside of __init__.py
from biotuner.biotuner_object import *
from biotuner.biotuner_group import BiotunerGroup
from biotuner.surrogates import (
    generate_surrogate,
    generate_surrogate_data,
    surrogate_group,
    plot_surrogate_distributions,
)
from biotuner.stats import (
    ttest_groups,
    ancova_groups,
    compare_all_metrics,
    correlate_metrics_peaks,
    plot_stats_comparison,
)
from biotuner.biotuner_mne import biotuner_mne
from biotuner.biotuner_utils import (
    slice_data,
    resample_2d,
    equate_dimensions,
    combine_dims,
)
from biotuner.harmonic_sequence import (
    HarmonicSequenceAnalyzer,
    HarmonicMarkov,
    WassersteinTrajectory,
    HarmonicDMD,
    HarmonicLatentSpace,
    HarmonicTopology,
    HarmonicGrammar,
    extract_tuning,
    encode_histograms,
    encode_scalar_metrics,
    encode_ji_matrix,
    TUNING_ATTRS,
)
from biotuner import harmonic_timbre  # noqa: F401  (subpackage; importable as biotuner.harmonic_timbre)
import numpy as np
import pandas as pd
import scipy
import matplotlib
import platform


# Maintainer info
__author__ = "The Biotuner development team"
__email__ = "antoine.bellemare9@gmail.com"

# Version info
__version__ = "0.1.0"


def version(silent=False):
    """Biotuner's version.
    This function is a helper to retrieve the version of the package.
    Examples
    ---------
    .. ipython:: python
      import biotuner as bt
      bt.version()
    """
    if silent is False:
        print(
            "- OS: " + platform.system(),
            "(" + platform.architecture()[1] + " " + platform.architecture()[0] + ")",
            "\n- Python: " + platform.python_version(),
            "\n\n- NumPy: " + np.__version__,
            "\n- Pandas: " + pd.__version__,
            "\n- SciPy: " + scipy.__version__,
            "\n- matplotlib: " + matplotlib.__version__,
        )
    else:
        return __version__
