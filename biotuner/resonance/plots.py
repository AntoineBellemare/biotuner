"""biotuner.resonance.plots — comparison plots over collections of resonance results.

These three plotters consume DataFrames containing the full H/PC/R triple — the
output of stacking multiple :func:`compute_resonance` (or legacy
``compute_global_harmonicity``) calls. They were moved here from
``biotuner.harmonic_spectrum`` because they visualize the FULL framework, not
just the harmonicity factor.

Expected DataFrame columns (any of the following per row):
    - 'harmonicity'      : 1-D ndarray, the H(f) spectrum
    - 'phase_coupling'   : 1-D ndarray, the PC(f) spectrum
    - 'trial'            : trial index (for ``plot_trial_corr``)

A convenience constructor that builds the expected DataFrame from a list of
:class:`ResonanceResult` objects lives at :func:`results_to_dataframe`.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, chi2, ttest_ind


def results_to_dataframe(results):
    """Convert a list of ResonanceResult into the legacy DataFrame format.

    Each result becomes one row with columns 'harmonicity', 'phase_coupling',
    'resonance', and 'trial' (0-indexed).
    """
    rows = []
    for i, r in enumerate(results):
        rows.append(
            {
                "trial": i,
                "harmonicity": r.factors["H"],
                "phase_coupling": r.factors["PC"],
                "resonance": r.resonance_spectrum,
            }
        )
    return pd.DataFrame(rows)


def harmonic_spectrum_plot_trial_corr(df_all, df_all_rnd, label1="Brain Signals", label2="Random Signals"):
    """Per-trial correlation between harmonicity and phase-coupling spectra.

    Expects df_all and df_all_rnd to have a 'trial' column and 'harmonicity' /
    'phase_coupling' array-valued columns.
    """
    try:
        from sklearn.preprocessing import MinMaxScaler
    except ImportError:
        raise ImportError(
            "The 'scikit-learn' package is required for this functionality. Install it with:\n\n"
            "    pip install scikit-learn\n"
        )
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    corrs = []
    ps = []
    reg_lines = []
    scaler = MinMaxScaler()
    for i in range(len(df_all)):
        harm_values = df_all[df_all["trial"] == i]["harmonicity"][0]
        phase_coupling_values = df_all[df_all["trial"] == i]["phase_coupling"][0]
        harm_values = scaler.fit_transform(harm_values.reshape(-1, 1))
        phase_coupling_values = scaler.fit_transform(phase_coupling_values.reshape(-1, 1))
        harm_values = harm_values.flatten()
        phase_coupling_values = phase_coupling_values.flatten()

        corr, p = pearsonr(harm_values, phase_coupling_values)
        corrs.append(corr)
        ps.append(p)
        z = np.polyfit(harm_values, phase_coupling_values, 1)
        reg_lines.append(z)
        ax1.plot(np.sort(harm_values), np.poly1d(z)(np.sort(harm_values)), color="darkblue", linestyle="--", alpha=0.5)
        ax1.set_title(label1)
        ax1.set_xlabel("Mean Harmonicity across freqs")
        ax1.set_ylabel("Mean Phase-Coupling across freqs")
    ax1.text(
        0.95, 0.95,
        f"r [{np.round(np.min(corrs), 2)}, {np.round(np.max(corrs), 2)}]",
        ha="right", va="top", transform=ax1.transAxes, fontsize=10, fontweight="bold",
    )

    corrs_rnd = []
    ps_rnd = []
    reg_lines_rnd = []
    for i in range(len(df_all_rnd)):
        harm_values_rnd = df_all_rnd[df_all_rnd["trial"] == i]["harmonicity"][0]
        phase_coupling_values_rnd = df_all_rnd[df_all_rnd["trial"] == i]["phase_coupling"][0]
        harm_values_rnd = scaler.fit_transform(harm_values_rnd.reshape(-1, 1))
        phase_coupling_values_rnd = scaler.fit_transform(phase_coupling_values_rnd.reshape(-1, 1))
        harm_values_rnd = harm_values_rnd.flatten()
        phase_coupling_values_rnd = phase_coupling_values_rnd.flatten()
        corr_rnd, p_rnd = pearsonr(harm_values_rnd, phase_coupling_values_rnd)
        corrs_rnd.append(corr_rnd)
        ps_rnd.append(p_rnd)
        z_rnd = np.polyfit(harm_values_rnd, phase_coupling_values_rnd, 1)
        reg_lines_rnd.append(z_rnd)
        ax2.plot(np.sort(harm_values_rnd), np.poly1d(z_rnd)(np.sort(harm_values_rnd)), color="darkblue", linestyle="--", alpha=0.5)
        ax2.set_title(label2)
        ax2.set_xlabel("Mean Harmonicity across freqs")
        ax2.set_ylabel("Mean Phase-Coupling across freqs")
    ax2.text(
        0.95, 0.95,
        f"r [{np.round(np.min(corrs_rnd), 2)}, {np.round(np.max(corrs_rnd), 2)}]",
        ha="right", va="top", transform=ax2.transAxes, fontsize=10, fontweight="bold",
    )

    corr_5th = np.percentile(corrs, 5)
    corr_95th = np.percentile(corrs, 95)
    corr_rnd_5th = np.percentile(corrs_rnd, 5)
    corr_rnd_95th = np.percentile(corrs_rnd, 95)
    t_stat, p_val = ttest_ind(corrs, corrs_rnd)
    sns.distplot(corrs, ax=ax3, label=label1, color="blue")
    ax3.axvline(corr_5th, color="blue", linestyle="--")
    ax3.axvline(corr_95th, color="blue", linestyle="--")
    sns.distplot(corrs_rnd, ax=ax3, label=label2, color="red")
    ax3.axvline(corr_rnd_5th, color="red", linestyle="--")
    ax3.axvline(corr_rnd_95th, color="red", linestyle="--")
    ax3.set_title("Distribution of correlation values")
    ax3.set_xlabel("Correlation (r)")
    ax3.set_ylabel("Density")
    ax3.legend()
    ax3.text(
        0.95, 0.05,
        f"t={np.round(t_stat, 2)}, p={np.round(p_val, 4)}",
        ha="right", va="top", transform=ax3.transAxes, fontsize=10, fontweight="bold",
    )

    fig.tight_layout()
    plt.show()


def harmonic_spectrum_plot_freq_corr(
    df1, df2, mean_phase_coupling=False, label1="Brain Signals", label2="Random Signals",
    fmin=2, fmax=30, xlim=None,
):
    """Per-frequency-bin correlation between harmonicity and phase-coupling."""
    try:
        from sklearn.preprocessing import MinMaxScaler
    except ImportError:
        raise ImportError(
            "The 'scikit-learn' package is required for this functionality. Install it with:\n\n"
            "    pip install scikit-learn\n"
        )
    n = len(df1)
    alpha = 0.05
    r_critical = np.sqrt(chi2.ppf(1 - alpha, df=1) / n)
    r_positive = r_critical
    r_negative = -r_critical

    freqs = np.linspace(fmin, fmax, len(df1["harmonicity"][0].tolist()[0]))
    corrs1 = np.zeros(len(freqs))
    corrs2 = np.zeros(len(freqs))
    mean_harmonicity1 = np.zeros(len(freqs))
    mean_harmonicity2 = np.zeros(len(freqs))
    mean_phase_coupling1 = np.zeros(len(freqs))
    mean_phase_coupling2 = np.zeros(len(freqs))

    for i in range(len(freqs)):
        harm_values1 = [row[i] for row in df1["harmonicity"]]
        mean_harmonicity1[i] = np.mean(harm_values1)
        phase_coupling_values1 = [row[i] for row in df1["phase_coupling"]]
        corrs1[i] = np.corrcoef(harm_values1, phase_coupling_values1)[0, 1]
        if mean_phase_coupling:
            mean_phase_coupling1[i] = np.mean(phase_coupling_values1)

        harm_values2 = [row[i] for row in df2["harmonicity"]]
        mean_harmonicity2[i] = np.mean(harm_values2)
        phase_coupling_values2 = [row[i] for row in df2["phase_coupling"]]
        corrs2[i] = np.corrcoef(harm_values2, phase_coupling_values2)[0, 1]
        if mean_phase_coupling:
            mean_phase_coupling2[i] = np.mean(phase_coupling_values2)

    scaler = MinMaxScaler()
    mean_harmonicity1_scaled = scaler.fit_transform(np.array(mean_harmonicity1).reshape(-1, 1)).flatten()
    mean_harmonicity2_scaled = scaler.fit_transform(np.array(mean_harmonicity2).reshape(-1, 1)).flatten()
    if mean_phase_coupling:
        mean_phase_coupling1_scaled = scaler.fit_transform(np.array(mean_phase_coupling1).reshape(-1, 1)).flatten()
        mean_phase_coupling2_scaled = scaler.fit_transform(np.array(mean_phase_coupling2).reshape(-1, 1)).flatten()

    plt.figure(figsize=(12.5, 4.5))
    corrs_min = min(corrs1.min(), corrs2.min()) - 0.05
    corrs_max = max(corrs1.max(), corrs2.max()) + 0.05

    for subplot_idx, (corrs, mean_h_scaled, mean_pc_scaled, label) in enumerate(
        [
            (corrs1, mean_harmonicity1_scaled, mean_phase_coupling1_scaled if mean_phase_coupling else None, label1),
            (corrs2, mean_harmonicity2_scaled, mean_phase_coupling2_scaled if mean_phase_coupling else None, label2),
        ],
        start=1,
    ):
        plt.subplot(1, 2, subplot_idx)
        ax1 = plt.gca()
        line1, = ax1.plot(freqs, corrs, color="black", label="Correlation (Harm x Phase)")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Correlation")
        ax1.set_ylim(corrs_min, corrs_max)
        ax1.axhline(r_positive, color="k", linestyle="--", label="p=0.05")
        ax1.axhline(r_negative, color="k", linestyle="--")
        ax2 = ax1.twinx()
        line2, = ax2.plot(freqs, mean_h_scaled, color="mediumblue", label="Mean Harmonicity")
        lines = [line1, line2]
        if mean_phase_coupling and mean_pc_scaled is not None:
            line3, = ax2.plot(freqs, mean_pc_scaled, color="deeppink", label="Mean Phase-Coupling")
            lines.append(line3)
        ax2.set_ylabel("Normalized measures")
        ax2.set_ylim(0, 1)
        if xlim is not None:
            ax1.set_xlim(xlim[0], xlim[1])
            ax2.set_xlim(xlim[0], xlim[1])
        ax1.legend(lines, [l.get_label() for l in lines], loc="upper right")
        ax1.set_title(label)

    plt.tight_layout()
    plt.show()


def harmonic_spectrum_plot_avg_corr(df1, df2, label1="Brain Signals", label2="Random Signals"):
    """Scatter mean-harmonicity vs mean-phase-coupling for two groups."""
    harm_values = [np.mean(row) for row in df1["harmonicity"]]
    phase_coupling_values = [np.mean(row) for row in df1["phase_coupling"]]
    harm_values_rnd = [np.mean(row) for row in df2["harmonicity"]]
    phase_coupling_values_rnd = [np.mean(row) for row in df2["phase_coupling"]]

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.scatter(harm_values, phase_coupling_values, color="darkblue", alpha=0.5)
    plt.xlabel("Averaged Harmonicity")
    plt.ylabel("Averaged Phase Coupling")
    z = np.polyfit(harm_values, phase_coupling_values, 1)
    plt.plot(np.sort(harm_values), np.poly1d(z)(np.sort(harm_values)), "r--")
    corr, p = pearsonr(harm_values, phase_coupling_values)
    print(f"{label1} - correlation: ", corr, "p-value: ", p)
    plt.title(label1)
    vmin_x = min(harm_values + harm_values_rnd)
    vmax_x = max(harm_values + harm_values_rnd)
    vmin_y = min(phase_coupling_values + phase_coupling_values_rnd)
    vmax_y = max(phase_coupling_values + phase_coupling_values_rnd)
    plt.xlim(vmin_x - (vmin_x / 100), vmax_x + (vmax_x / 100))
    plt.ylim(vmin_y - (vmin_y / 100), vmax_y + (vmax_y / 100))

    plt.subplot(1, 2, 2)
    plt.scatter(harm_values_rnd, phase_coupling_values_rnd, color="darkblue", alpha=0.5)
    plt.xlabel("Averaged Harmonicity")
    plt.ylabel("Averaged Phase Coupling")
    z_rnd = np.polyfit(harm_values_rnd, phase_coupling_values_rnd, 1)
    plt.plot(np.sort(harm_values_rnd), np.poly1d(z_rnd)(np.sort(harm_values_rnd)), "r--")
    corr_rnd, p_rnd = pearsonr(harm_values_rnd, phase_coupling_values_rnd)
    print(f"{label2} - correlation: ", corr_rnd, "p-value: ", p_rnd)
    plt.title(label2)
    plt.xlim(vmin_x - (vmin_x / 100), vmax_x + (vmax_x / 100))
    plt.ylim(vmin_y - (vmin_y / 100), vmax_y + (vmax_y / 100))

    plt.tight_layout()
    plt.show()
