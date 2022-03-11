import matplotlib.pyplot as plt
import matplotlib
import seaborn as sbn
import numpy as np
import pickle
import pyemma
from utils import *

tic1 = np.load('first_tic.npy')
tic2 = np.load('second_tic.npy')

with open('dtrajs.pkl', 'rb') as f:
    dtrajs = pickle.load(f)
    
msm = pyemma.msm.estimate_markov_model(dtrajs, lag=30)

X = tic1
Y = tic2
x, y, z = get_histogram(
    X, Y, nbins=150, weights=np.concatenate(msm.trajectory_weights()),
    avoid_zero_count=False)
free_energy = _to_free_energy(z, minener_zero=True) * 0.596 # Conversion from kT to kcal/mol
max_energy = 6
plt.figure(figsize=(5.5, 4))
plt.rcParams["axes.labelweight"] = "normal"
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["xtick.major.width"] = 1
plt.rcParams["ytick.major.width"] = 1
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rc('font', family='helvetica', size=16, weight="normal")
plt.rcParams.update({"axes.labelweight": "normal"})
plt.contourf(x, y, free_energy, np.linspace(0, max_energy, max_energy*5+1), vmin=0.0, vmax=max_energy, cmap='jet')
cbar = plt.colorbar(ticks=range(max_energy+1))
cbar.set_label("Free Energy (kcal/mol)",size=16)
cbar.ax.set_yticklabels(range(max_energy+1))
cbar.ax.tick_params(labelsize=16)
plt.tick_params(axis='both',labelsize=16)
plt.xlim(-3, 2)
plt.ylim(-8, 2)
# plt.yticks(np.arange(-3,6,1))
plt.xlabel('tIC 1 (a.u.)')
plt.ylabel('tIC 2 (a.u.)')
plt.savefig('free_energy_map.png', dpi=600, bbox_inches='tight')
plt.close()