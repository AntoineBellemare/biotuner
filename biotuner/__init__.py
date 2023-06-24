# Inside of __init__.py
from biotuner.biotuner_object import *
import numpy as np
import pandas as pd
import scipy
import sklearn
import matplotlib
import platform


# Maintainer info
__author__ = "The Biotuner development team"
__email__ = "antoine.bellemare9@gmail.com"

# Version info
__version__ = "0.0.1"

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
            "\n- NeuroKit2: " + __version__,
            "\n\n- NumPy: " + np.__version__,
            "\n- Pandas: " + pd.__version__,
            "\n- SciPy: " + scipy.__version__,
            "\n- sklearn: " + sklearn.__version__,
            "\n- matplotlib: " + matplotlib.__version__,
        )
    else:
        return __version__
