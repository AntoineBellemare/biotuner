import numpy as np
import sys
sys.path.insert(0, 'C:/Users/Antoine/github/MEG_pareidolia/python_scripts/Functions')
import MEG_pareidolia_utils
from MEG_pareidolia_utils import get_pareidolia_bids, p_values_boolean_1d
from PARAMS import FOLDERPATH
import pandas as pd
import mne
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.pyplot as plt

GLM_name = 'GLM_DMT_1Hz_EMD_5sec_5min_nAGQ0_rand_int_elec'
savename = 'DMT_1Hz_EMD_5sec_5min_nAGQ0_rand_int_noFDR'
path = 'C:/Users/Antoine/github/biotuner/OUTPUT/DMT_analyses/r_models/'+GLM_name
from mne.viz import plot_topomap

import numpy as np

import mne

ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz',
            'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'POz']

# Create a 10-20 standard montage
montage_1020 = mne.channels.make_standard_montage('standard_1020')

# Get positions of the specified channels
sensor_pos_3d = np.array([montage_1020.get_positions()['ch_pos'][ch_name] for ch_name in ch_names])

import numpy as np

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi

def azimuthal_equidistant_projection(x, y, z):
    r, theta, phi = cartesian_to_spherical(x, y, z)
    x_2d = theta * np.cos(phi)
    y_2d = theta * np.sin(phi)
    return x_2d, y_2d


SENSOR_POS = np.array([azimuthal_equidistant_projection(x, y, z) for x, y, z in sensor_pos_3d])


def merge_multi_GLM(path, n_electrodes=270, graph=False, ch_xy=None, savename='_', pval_thresh=0.01, FDR=True):

    df = pd.read_csv(path+'0.csv')
    df = df.rename(columns={"Unnamed: 0": "fixed_effect"})
    fixed_eff_names = list(df['fixed_effect'])
    fixed_eff_names_mod = [a.replace(':', '_by_') for a in fixed_eff_names]
    pval_name = df.columns[-1]
    dict_final = dict.fromkeys(fixed_eff_names_mod)
    
    for effect, new_name in zip(fixed_eff_names, fixed_eff_names_mod):
        effects = []
        pvals = []
        for n in range(0, n_electrodes):
            df_ = pd.read_csv(path+str(n)+'.csv')
            df_ = df_.rename(columns={"Unnamed: 0": "fixed_effect"})
            df_['rescale'] = (df_['Estimate'] - min(df_['Estimate'])) / (max(df_['Estimate']) - min(df_['Estimate']))
            #print(df_['rescale'])
            #print(df_)
            effect_ = float(df_.loc[df_['fixed_effect'] == effect, 'Estimate'])
            #print('EFFECT', df_['Estimate'])
            effects.append(effect_)
            pval = float(df_.loc[df_['fixed_effect'] == effect, pval_name])
            if pval < 0.01:
                print(effect, ':', n, 'pval', pval, 'size ', effect_)
            pvals.append(pval)
        dict_final[new_name] = [effects, pvals]
    DF_final = pd.DataFrame.from_dict(dict_final, orient='index')
    DF_final.to_csv('dict_final.csv')
    if graph is True:

        for e, effect in enumerate(list(dict_final.keys())):
            value_to_plot = dict_final[effect][0]
            #print('VALUE', value_to_plot)
            pvals = dict_final[effect][1]
            #print(value_to_plot, pvals)
            if FDR is True:
                _, pvals = fdrcorrection(pvals, alpha=pval_thresh, method='indep')
            mask = p_values_boolean_1d(pvals, threshold = pval_thresh)
            extreme = np.max((abs(np.min(np.min(np.array(value_to_plot)))), abs(np.max(np.max(np.array(value_to_plot)))))) # adjust the range of values
            vmax = 20
            vmin = -20
            reportpath = 'fig_GLM_'+savename+effect+'.png'
            print(reportpath)
            #image,_ = mne.viz.plot_topomap(data=value_to_plot, pos=ch_xy, cmap='Spectral_r', vmin=vmin, vmax=vmax, axes=None, show=True, mask = p_welch_multitaper)
            fig, ax = topoplot(value_to_plot, ch_xy, vmin=vmin, vmax=vmax, showtitle=True,
                               mask = mask, figpath = reportpath, ax_title='fixed-effect estimate')
    return dict_final

def topoplot(toplot, ch_xy, showtitle=False, titles=None, savefig=True,
             figpath=r'C:\Users\Dell\Jupyter\BrainHackSchool2019_AB\EEG_music_scripts', vmin=-1, vmax=1,
             ax_title = 't values', mask = None):
    #create fig
    fig, ax = plt.subplots(1,1, figsize=(10,5))
    #create a topomap for each data array

    image,_ = mne.viz.plot_topomap(data=toplot, pos=ch_xy, cmap='Spectral_r', vmin=vmin, vmax=vmax,
                                   outlines='skirt', sphere=0.75, axes=ax, show=False, mask = mask,
                                   mask_params=dict(marker='*', markerfacecolor='w', markeredgecolor='w',
                                   linewidth=0, markersize=10))
    #option for title
    if showtitle == True:
        ax.set_title(titles, fontdict={'fontsize': 10, 'fontweight': 'heavy'})
    #add a colorbar at the end of the line (weird trick from https://www.martinos.org/mne/stable/auto_tutorials/stats-sensor-space/plot_stats_spatio_temporal_cluster_sensors.html#sphx-glr-auto-tutorials-stats-sensor-space-plot-stats-spatio-temporal-cluster-sensors-py)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(image, cax=ax_colorbar)
    ax_colorbar.set_title(ax_title)
    ax_colorbar.tick_params(labelsize=8)
    #save plot if specified
    if savefig == True:
        plt.savefig(figpath, dpi=300)
    #plt.show()
    return fig, ax

dict_final = merge_multi_GLM(path, n_electrodes=31, graph=True, ch_xy=SENSOR_POS, savename=savename, pval_thresh=0.05, FDR=False)
