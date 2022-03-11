import numpy as np
from matplotlib import cm

counts = np.loadtxt("count_mat.dat")
eq     = np.loadtxt("msm_eq_pop.dat")
dat1   = np.load   ("data_x.npy", allow_pickle=True)
dat2   = np.load   ("data_y.npy", allow_pickle=True)

xmax =  2
xmin = -4
ymax =  4
ymin = -3
bin_size = 200

i=0 
hists = np.zeros((bin_size, bin_size))
for i in range(len(eq)):
    hist  = np.histogram2d( dat1[i], dat2[i], bins=bin_size, range=[ [xmin, xmax], [ymin, ymax] ])[0]
    hists = hists + hist * eq[i] / len(dat1[i])
energy = -0.6*np.log(hists) - np.min(np.hstack(-0.6*np.log(hists)))
energy = np.transpose(energy)

import matplotlib.pyplot as plt
import math
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'], 'size': '12', 'weight':'bold'})
params = {'mathtext.default': 'regular' }
plt.rcParams.update(params)

from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('12')

ax0 = plt.subplot2grid((2,2),(0,0))

#c=ax0.imshow(energy, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=6, interpolation='nearest')

#xs  = np.linspace(xmin, xmax, bin_size)
#ys  = np.linspace(xmin, ymax, bin_size)
c=ax0.contourf(energy, np.linspace(0, 6, 33), origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='jet')

ax0.set_xlim(xmin, xmax)
ax0.set_ylim(ymin, ymax)
ax0.set_xticks((xmin, -3, -2, -1, 0, 1, xmax))
ax0.set_yticks((ymin, -2, -1, 0, 1, 2, 3, ymax))
ax0.set_xlabel(r'tIC 1', fontweight="bold")
ax0.set_ylabel(r'tIC 2', fontweight="bold")
cbar = plt.colorbar(c,ticks=[0,2,4,6])
cbar.ax.set_ylabel('Free energy (kcal/mol)', fontweight="bold")

paths = np.load("../transition_paths/paths.npy", allow_pickle=True)
data   = np.loadtxt("avg_dist.txt")

for i, path in enumerate(paths):
  #if i != 0: continue
  states = path #[14, 18, 34, 74, 85, 88] 
  tic1   = [ data[i][1] for i in states ]
  tic2   = [ data[i][3] for i in states ]
  if tic2[0] > 1: print(path)
  #print(tic1, tic2)
  ax0.plot(tic1, tic2, color='black')
  #ax0.scatter(tic1, tic2, marker='*', color='magenta', alpha=0.5, s=30, linewidths=0.5)


#for i in range(len(paths[1])):
  #ax0.annotate(str(i+1), (tic1[i], tic2[i]))

plt.tight_layout()
plt.savefig("tIC1-tIC2-paths.png", dpi=300, bbox_inches='tight')
