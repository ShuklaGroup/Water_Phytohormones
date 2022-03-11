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
This code was used to generate the free energy landscape of the RMSD of alphaN helix of NPR4 SBC vs. the RMSD of the lignad (w.r.t. bound position).
'''

# LOAD DATA
bound_top = "dry_sbc.prmtop" # Starting conformation (from crystal structure)
starting_structure = "dry_sbc.pdb"
universe = mda.Universe(bound_top, starting_structure) # The .pdb file is provided for the starting coordinates
bind_traj = mda.Universe(bound_top, '/mnt/sda/DIEGO/0-Plant_Hormones/SA/holo/accelerated/alpha_3/structures_for_unbiased_simulation/strip.compressed_full.xtc') # Binding simulation data

# SELECT RELEVANT HELIX RESIDUES
alpha_helix_sel = "backbone and (resid 45:74 or resid 100:137)" # From VMD
bb_alpha_helix_native = universe.select_atoms(alpha_helix_sel)
bb_alpha_helix_bind = bind_traj.select_atoms(alpha_helix_sel)

# ALIGN STRUCTURE FOR RMSD ANALYSIS
alignment = align.AlignTraj(bind_traj, universe, select=alpha_helix_sel, in_memory=False)
alignment.run() # Outputs "rmsfit_strip.compressed_full.xtc" file
rmsd_data = alignment.rmsd # Not used, only for checking quality of alignment

# COMPUTE RMSD

# Compute RMSD of alphaN helix
bind_traj = mda.Universe(bound_top, "rmsfit_strip.compressed_full.xtc")
alpha_N_sel = "backbone and (resid 9:26)"
bb_alpha_N_native = universe.select_atoms(alpha_N_sel)
bb_alpha_N_bind = bind_traj.select_atoms(alpha_N_sel)
rmsd_alpha_N = np.empty(bind_traj.trajectory.n_frames)
for i in range(bind_traj.trajectory.n_frames):
    bind_traj.trajectory[i]
    rmsd_alpha_N[i] = rms.rmsd(bb_alpha_N_bind.positions, bb_alpha_N_native.positions)

# Compute RMSD of ligand
ligand_sel = "resname SAL"
ligand_native = universe.select_atoms(ligand_sel)
ligand_bind = bind_traj.select_atoms(ligand_sel)
rmsd_ligand = np.empty(bind_traj.trajectory.n_frames)
for i in range(bind_traj.trajectory.n_frames):
    bind_traj.trajectory[i]
    rmsd_ligand[i] = rms.rmsd(ligand_bind.positions, ligand_native.positions)

# CREATE PLOT
X = rmsd_ligand
Y = rmsd_alpha_N
x, y, z = get_histogram(
    X, Y, nbins=100, weights=np.load('../../water_shell_figures/SAL/weights.npy'), # Must provide path to weights from MSM
    avoid_zero_count=False)
free_energy = _to_free_energy(z, minener_zero=True) * 0.596 # Conversion from kT to kcal/mol
max_energy = 10
plt.figure(figsize=(5.5, 4))
plt.rcParams["axes.labelweight"] = "normal"
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["xtick.major.width"] = 1
plt.rcParams["ytick.major.width"] = 1
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rc('font', family='helvetica', size=16, weight="normal")
plt.rcParams.update({"axes.labelweight": "normal"})
plt.contourf(x, y, free_energy, np.linspace(0, max_energy, max_energy*5+1), vmin=0.0, vmax=max_energy, cmap='jet')
cbar = plt.colorbar(ticks=range(0, max_energy+1, 2))
cbar.set_label("Free Energy (kcal/mol)",size=16)
cbar.ax.set_yticklabels(range(0, max_energy+1, 2))
cbar.ax.tick_params(labelsize=16)
plt.tick_params(axis='both',labelsize=16)
plt.xlim(0, 60)
plt.ylim(0, 60)
plt.xlabel('RMSD of SA from \n the bound structure (Å)')
plt.ylabel(r'$\alpha$' + 'N helix RMSD from \n the bound pose (Å)')
plt.tight_layout()
plt.savefig('free_energy_map_rmsd.png', dpi=600, bbox_inches='tight')
plt.close()