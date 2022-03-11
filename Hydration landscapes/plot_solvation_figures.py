import matplotlib.pyplot as plt
import matplotlib
import seaborn as sbn
import numpy as np
import pickle
import pyemma
from utils import *

def plot_solvation_fig(tica_arr, solvation_arr, nbins=(200,65), title='', range_x=[], range_y=[], save_path='', weights=None, max_energy=6):
    X = tica_arr
    Y = solvation_arr
#     print(X.shape)
#     print(Y.shape)
    x, y, z = get_histogram(
        X, Y, nbins=nbins, weights=weights,
        avoid_zero_count=False)
    free_energy = _to_free_energy(z, minener_zero=True) * 0.596 # Conversion from kT to kcal/mol
    
    plt.figure(figsize=(11,8))
    
    plt.rcParams["axes.labelweight"] = "normal"
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["xtick.major.width"] = 1.5
    plt.rcParams["ytick.major.width"] = 1.5
    plt.rcParams['axes.unicode_minus'] = False
    matplotlib.rc('font',family='helvetica',size=24)
    plt.contourf(x, y, free_energy, np.linspace(0, max_energy, max_energy*5+1), vmin=0.0, vmax=max_energy, cmap='jet')
    if max_energy <= 7:
        colorbar_ticks = range(max_energy+1)
    else:
        colorbar_ticks = range(0, max_energy+1, 2)
    cbar = plt.colorbar(ticks=colorbar_ticks)
    cbar.set_label("Free Energy (kcal/mol)", size=24)
    cbar.ax.set_yticklabels(colorbar_ticks)
    cbar.ax.tick_params(labelsize=20)
    plt.tick_params(axis='both',labelsize=24)
    plt.xlim(range_x)
    plt.ylim(range_y)
    plt.title(title)
    plt.xlabel('tIC 1 (a.u.) - Binding coordinate')
    plt.ylabel('Number of waters 5 Å around ligand')
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
#     plt.show()
    plt.close()

if __name__ == '__main__': 
    tica_file = 'STR/first_tic.npy' # Change filepath here
    solvation_file = 'STR/second_shell.npy' # Change filepath here
    dtrajs = 'STR/dtrajs.pkl' # Change filepath here

    with open(dtrajs, 'rb') as f:
        dtrajs = pickle.load(f)

    weights = np.concatenate(pyemma.msm.estimate_markov_model(dtrajs, lag=30).trajectory_weights())

    # Make sure frames follow same stride or adjust stride if needed
    tica_arr = np.load(tica_file) 
    solvation_arr = np.load(solvation_file)

    plot_solvation_fig(tica_arr, 
                       solvation_arr, 
                       nbins=(200,98), # Adjust as appropriate
                       title="", 
                       save_path='STR/solvation_landscape.png', # Change filepath here
                       range_x=[-4, 2], range_y=[0, 100], # Adjust as appropriate
                       weights=weights)