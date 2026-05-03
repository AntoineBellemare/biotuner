"""
MNE integration for biotuner.

Provides :func:`biotuner_mne`, which computes biotuner metrics for every
trial × electrode combination in an MNE Epochs object and returns a
pandas DataFrame (optionally saved to CSV).
"""

import numpy as np
import pandas as pd

from biotuner.biotuner_object import fit_biotuner


def biotuner_mne(epochs, bt_dict, savefile=False, savename=None):
    """Compute biotuner metrics for all trials and electrodes in MNE Epochs.

    Iterates over every trial × electrode combination, runs
    :func:`~biotuner.biotuner_object.fit_biotuner` with the given parameter
    dictionary, and collects the results into a :class:`pandas.DataFrame`.
    Original epoch metadata (if present) is merged into the output.

    Parameters
    ----------
    epochs : mne.Epochs
        MNE Epochs object. Data is accessed via ``epochs.get_data()``, which
        returns an array of shape ``(n_trials, n_electrodes, n_samples)``.
    bt_dict : dict
        Parameter dictionary for :func:`~biotuner.biotuner_object.fit_biotuner`.
        Keys are metric names; values are the corresponding parameter values.
    savefile : bool, default=False
        If ``True``, write the results DataFrame to CSV.
    savename : str, optional
        Base filename for the CSV (without extension). If ``None``, derived
        from ``epochs.filename`` by stripping the extension and appending
        ``'_biotuner'``.

    Returns
    -------
    df : pd.DataFrame
        One row per trial × electrode combination. Columns include all keys
        in *bt_dict* plus ``'trial'``, ``'electrode'``, and any metadata
        columns attached to the Epochs object.

    Examples
    --------
    >>> import mne
    >>> from biotuner.biotuner_mne import biotuner_mne
    >>>
    >>> epochs = mne.read_epochs('my_epochs-epo.fif')
    >>> bt_params = {'peaks_function': 'EMD', 'precision': 0.5, 'n_harm': 10}
    >>> df = biotuner_mne(epochs, bt_params, savefile=True)
    >>> df.head()
    """
    data = epochs.get_data()
    n_trials, n_electrodes, _ = data.shape
    metrics_list = []

    for j in range(n_trials):
        for k in range(n_electrodes):
            ts = data[j, k, :]
            metrics = fit_biotuner(ts, bt_dict)
            row = {name: value for name, value in metrics.items()}
            row['trial'] = j
            row['electrode'] = k
            metrics_list.append(row)

    df = pd.DataFrame(metrics_list)

    if savefile:
        if savename is None:
            savename = epochs.filename[:-4] + '_biotuner'

        # Merge epoch-level metadata if available
        if hasattr(epochs, 'metadata') and epochs.metadata is not None:
            metadata = epochs.metadata
            for key in metadata.columns:
                df[key] = np.tile(metadata[key].values, n_electrodes)

        metric_cols = list(bt_dict.keys())
        meta_cols = list(epochs.metadata.columns) if (
            hasattr(epochs, 'metadata') and epochs.metadata is not None
        ) else []
        ordered_cols = ['trial', 'electrode'] + metric_cols + meta_cols
        df = df[[c for c in ordered_cols if c in df.columns]]
        df.to_csv(savename + '.csv', index=False)

    return df
