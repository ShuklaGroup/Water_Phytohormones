import matplotlib.pyplot as plt
import matplotlib
import seaborn as sbn
import numpy as np
import pyemma
import MDAnalysis as mda
from MDAnalysis.analysis import align, rms
import mdtraj as md
from glob import glob
import pickle
from utils import *

'''
This file contains the code used for the analysis of the simulations of the NPR4 salicylic acid binding unit-salicylic acid system.
'''

# LOAD FILES

bound_top = "dry_sbc.prmtop" # Starting conformation (from crystal structure)
starting_structure = "dry_sbc.pdb"
universe = mda.Universe(bound_top, starting_structure) # The .pdb file is provided for the starting coordinates

def get_fpaths(paths, base_name='', extension=''):
    files = []
    for p in paths:
        for fpath in glob(p + "/*/" + fname_base + extension):
            files.append(fpath)
    return files

path_to_holo_trajectories = ["../holo/parallel_runs"] # Trajectories starting with ligand inside of pocket
path_to_apo_trajectories = ["../apo/parallel_runs", "../apo/parallel_runs2"] # Trajectories starting with ligand outside of pocket
path_to_bind_trajectories = ["../holo/accelerated/alpha_3/structures_for_unbiased_simulation"] # Trajectories starting with ligand in tunnel
fname_base = "strip.prod" 
extension = ".xtc"

holo_files = get_fpaths(path_to_holo_trajectories, base_name=fname_base, extension=extension)
apo_files = get_fpaths(path_to_apo_trajectories, base_name=fname_base, extension=extension)
bind_files = get_fpaths(path_to_bind_trajectories, base_name=fname_base, extension=extension)

# Save the order in which the trajectories were analyzed
holo_traj_order = [hf.split('/')[3] for hf in holo_files]
apo_traj_order = [af.split('/')[3] for af in apo_files]
bind_traj_order = [bf.split('/')[5] for bf in bind_files]

with open("../holo/parallel_runs/traj_order.dat", "w") as out:
    for tr in holo_traj_order:
        out.write(tr + '\n')
with open("../apo/parallel_runs/traj_order.dat", "w") as out:
    for tr in apo_traj_order:
        if not ('SA' in tr):
            out.write(tr + '\n')
with open("../apo/parallel_runs2/traj_order.dat", "w") as out:
    for tr in apo_traj_order:
        if ('SA' in tr):
            out.write(tr + '\n')
with open("../holo/accelerated/alpha_3/structures_for_unbiased_simulation/traj_order.dat", "w") as out:
    for tr in bind_traj_order:
        out.write(tr + '\n')

# FEATURES
# Ligand pocket distances
pocket = universe.select_atoms('(name CA) and (byres around 5 resname SAL)')
pocket_idx = pocket.indices
ligand = universe.select_atoms('resname SAL and (not type H)')
ligand_idx = ligand.indices
pocket_index_pairs = [(i, j) for i in pocket_idx for j in ligand_idx]

# Distances between residues in alpha helices that are sparated by at least 5 residues
prot_CA = universe.select_atoms('name CA and (resid 9:26 or resid 45:74 or resid 100:137)') 
prot_CA_idx = prot_CA.indices
size = len(prot_CA_idx)
sep = 5
prot_index_pairs = []
for i in range(size-1):
    for j in range(i+sep, size):
        prot_index_pairs.append((prot_CA_idx[i], prot_CA_idx[j]))

pocket_featurizer = pyemma.coordinates.featurizer(bound_top)
pocket_featurizer.add_distances(pocket_index_pairs)
helices_featurizer = pyemma.coordinates.featurizer(bound_top)
helices_featurizer.add_distances(prot_index_pairs)
traj = holo_files + apo_files + bind_files
data_pocket = pyemma.coordinates.load(traj, features=pocket_featurizer, stride=1)
data_helices = pyemma.coordinates.load(traj, features=helices_featurizer, stride=1)

all_features_traj = [] # Combine all data in same array
for d_pocket, d_helices in zip(data_pocket, data_helices):
    d_all = np.append(d_pocket, d_helices, axis=1)
    all_features_traj.append(d_all)

# TIME-LAGGED INDEPENDENT COMPONENT ANALYSIS

# Perform tICA with 1-10 components
tica_fit = []
for i in range(1, 11):
    tica_fit.append(pyemma.coordinates.tica(all_features_traj, dim=i, lag=1))
# Save results
for i in range(10):
    tica_fit[i].save('tica_fit_' + str(i),  save_streaming_chain=True, overwrite=True)
    
# CROSS VALIDATION (number of clusters, tICA components)
n_clustercenters = [5, 10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

scores = np.zeros((len(n_clustercenters), 10, 10)) # Dimensions correspond to: (# clusters, # tica components, # trial)
for n, k in enumerate(n_clustercenters):
    for i in range(10):
        print("Testing k = {:2d} and # tIC = {:2d}".format(k, i+1))
        tica_data = pyemma.load('tica_fit_{:d}'.format(i)).get_output()
        for m in range(10):
            pyemma.util.config.show_progress_bars = False
            _cl = pyemma.coordinates.cluster_mini_batch_kmeans(
                tica_data, k=k, max_iter=10000, n_jobs=4)
            pyemma.util.config.show_progress_bars = True
            _msm = pyemma.msm.estimate_markov_model(_cl.dtrajs, 5)
            scores[n, i, m] = _msm.score_cv(
                _cl.dtrajs, n=1, score_method='VAMP1', score_k=min(10, k)) # Using VAMP-1/GMRQ
    print("Saving results for k =", k)
    score_for_give_k = scores[n, :, :].copy()
    np.save("VAMP1_score_" + str(k) + ".npy", score_for_give_k)

# Cross-validation plot
fig, ax = plt.subplots()
plt.rcParams.update({'font.size': 16, 'font.sans-serif': 'Helvetica'})
plt.rcParams["axes.labelweight"] = "heavy"

colors = [
    (27/256, 0/256, 128/256),
    (34/256, 0/256, 255/256),
    (0/256, 62/256, 255/256),
    (0/256, 207/256, 255/256),
    (0/256, 255/256, 145/256),
    (112/256, 155/256, 0/256),
    (255/256, 230/256, 0/256),
    (255/256, 95/256, 0/256),
    (255/256, 0/256, 0/256),
    (116/256, 0/256, 0/256)
]

fig.set_size_inches(5.5, 4)
for i in range(10):
    s = scores[2:, i, :]
    col = colors[i]
    lower, upper = pyemma.util.statistics.confidence_interval(s.T.tolist(), conf=0.95)
    ax.errorbar(n_clustercenters[2:], np.mean(s, axis=1), np.abs(np.vstack((lower, upper))-np.mean(s, axis=1)), color=col, capsize=2)
    ax.plot(n_clustercenters[2:], np.mean(s, axis=1), '-', color=col)
    frame = plt.legend(['{} tics'.format(n) for n in np.arange(1, 11)], loc='lower right', prop={'weight': 'bold', 'size': 8}).get_frame()
    frame.set_linewidth(1)
    frame.set_edgecolor('black')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('GMRQ')
    plt.ylim([8.5, 10.5])
    plt.xlim([0, 1400])
    plt.yticks(np.arange(8.5, 10.6, 0.5), weight = 'bold')
    plt.xticks(np.arange(0, 1001, 200), weight = 'bold')
    ax.tick_params(labeltop=False, labelright=False, right=True, top=True, direction='in')

plt.savefig('cross_validation.png', dpi=600, bbox_inches='tight')
plt.close()

# CONSTRUCT MSM

n_clusters = 500 # Selected based on cross-validation
tica_data = pyemma.load('tica_fit_9').get_output() # Selected based on cross-validation
tica_data = tica_fit.get_output()
kmeans_fit = pyemma.coordinates.cluster_kmeans(tica_data, k=n_clusters, max_iter=1000)

its = pyemma.msm.its(kmeans_fit.dtrajs, nits=10, lags=np.arange(1, 60, 10)) # Check implied timescales

# Plot implied timescales

from matplotlib import rc
from matplotlib.ticker import LogLocator
rc('font',**{'family':'sans-serif', 'sans-serif':['Helvetica'], 'size': '16', 'weight':'bold'})
params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)

from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('16')

fig, ax = plt.subplots(figsize=(5.5, 4))
colors = ['blue', 'green', 'red', (0/255, 184/255, 179/255), (199/255, 0, 186/255), (177/255, 186/255, 0), 'black', 'blue', 'green', 'red']
for i in range(10):
    c = colors[i]
    process = its.timescales[:, i]
    x = np.linspace(1, 44, 24)[1:]
    y = np.interp(x, its.lags, process)
    ax.plot(x, y, 'o', markersize=5, c=c, markeredgecolor='black', markeredgewidth=1/3)

plt.xlim([0, 50])
plt.ylim(1, 1e5)
plt.xticks(np.arange(0, 51, 10), weight = 'bold')
plt.yscale('log')
y_ticks = np.concatenate([np.linspace(l, l*10, 10) for l in np.logspace(0, 4, 5, base=10)])
plt.yticks(np.concatenate([np.linspace(l, l*10, 10) for l in np.logspace(0, 5, 6, base=10)])[::10], weight = 'bold')
ax.yaxis.set_minor_locator(FixedLocator(y_ticks))

ax.set_xlabel('Lag time (ns)', fontweight='bold')
ax.set_ylabel('Implied timescales (ns)', fontweight='bold')
ax.tick_params(labeltop=False, labelright=False, right=True, top=True, direction='in', which='both')
plt.savefig('timescales.png', dpi=300, bbox_inches='tight')
plt.close()

msm = pyemma.msm.estimate_markov_model(kmeans_fit.dtrajs, 20, score_method='VAMP1') # Lag selected from implied timescale convergence

# tIC 2 vs. tIC 1 free energy landscape

X = np.concatenate(tica_data)[:, 0]
Y = np.concatenate(tica_data)[:, 1]
x, y, z = get_histogram(
    X, Y, nbins=80, weights=np.concatenate(msm.trajectory_weights()),
    avoid_zero_count=False)
free_energy = _to_free_energy(z, minener_zero=True) * 0.596 # Conversion from kT to kcal/mol
max_energy = 12
plt.figure(figsize=(5.5, 4))
plt.rcParams["axes.labelweight"] = "normal"
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["xtick.major.width"] = 1
plt.rcParams["ytick.major.width"] = 1
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rc('font', family='helvetica', size=16, weight="normal")
plt.rcParams.update({"axes.labelweight": "normal"})
plt.contourf(x, y, free_energy, np.linspace(0, max_energy, max_energy*5+1), vmin=0.0, vmax=max_energy, cmap='jet')
cbar = plt.colorbar(ticks=range(0, max_energy+1, 4))
cbar.set_label("Free Energy (kcal/mol)",size=16)
cbar.ax.set_yticklabels(range(0, max_energy+1, 4))
cbar.ax.tick_params(labelsize=16)
plt.tick_params(axis='both',labelsize=16)
plt.xlim(-4, 1)
plt.ylim(-3, 5)
plt.yticks(np.arange(-3,6,1))
plt.xlabel('tIC 1 (a.u.)')
plt.ylabel('tIC 2 (a.u.)')
plt.savefig('free_energy_map.png', dpi=600, bbox_inches='tight')
plt.close()

# TRANSITION PATH ANALYSIS

# Classify clusters into bound or unbound based on distance between ligand and pocket residue
n_clusters = 500
ligand_distance_per_state = {}
inp = pyemma.coordinates.source(traj, top=bound_top)

for i in range(n_clusters):
    ligand_distance_per_state[i] = []

for i in range(inp.ntraj):
    d_pocket = data_pocket[i][:,85:87].mean(axis=0)
    k_clusters = kmeans_fit.dtrajs[i]
    for k in k_clusters:
        for d in d_pocket:
            ligand_distance_per_state[k].append(d)

for i in range(n_clusters):
    ligand_distance_per_state[i] = np.asarray(ligand_distance_per_state[i])

min_per_state = np.empty(n_clusters)
for i in range(n_clusters):
    min_per_state[i] = ligand_distance_per_state[i].min()

bound_states = np.where(np.logical_and(min_per_state < 0.34, kmeans_fit.cluster_centers_[:, 0] > 0.5))[0]
unbound_states = np.where(np.logical_and(min_per_state > 2.5, kmeans_fit.cluster_centers_[:, 0] < -1))[0]

bound_states_indices = np.in1d(msm.connected_sets[0], bound_states).nonzero()[0]
unbound_states_indices = np.in1d(msm.connected_sets[0], unbound_states).nonzero()[0]

# Get first pathway

tpt = pyemma.msm.tpt(msm, unbound_states_indices, bound_states_indices)
pathways = tpt.pathways(fraction=0.9)
main_pathway = pathways[0][0] # Contains clusters in main pathway
np.save('main_pathway.npy', main_pathway)