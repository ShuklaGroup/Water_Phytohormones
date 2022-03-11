import numpy as np
from matplotlib import cm

counts = np.loadtxt("count_mat.dat")
eq     = np.loadtxt("msm_eq_pop.dat")
dat1   = np.load   ("data_x.npy",allow_pickle=True)
dat2   = np.load   ("data_y.npy",allow_pickle=True)

xmax =  2
xmin = -3
ymax =  4
ymin = -2
bin_size = 300

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
c=ax0.contourf(energy, np.linspace(0, 6, 25), origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='jet')
#ax0.set_xlim(xmin, xmax)
#ax0.set_ylim(ymin, ymax)
#ax0.set_xticks((xmin, -3, -2, -1, 0, 1, 2, 3, xmax))
#ax0.set_yticks((ymin, -3, -2, -1, 0, 1, 2, 3, ymax))
ax0.set_xlabel(r'tIC 1', fontweight="bold")
ax0.set_ylabel(r'tIC 2', fontweight="bold")
cbar = plt.colorbar(c,ticks=[0,2,4,6])
cbar.ax.set_ylabel('Free energy (kcal/mol)', fontweight="bold")

states = [0, 5, 9, 20, 25, 36, 44, 54, 70, 74, 78, 84, 86, 87, 99, 116, 119, 122, 141, 152, 157, 167, 168, 173, 186, 187] 
data   = np.loadtxt("avg_dist.txt")

tic1   = [ data[i][1] for i in states ]
tic2   = [ data[i][3] for i in states ]
ax0.scatter(tic1, tic2, marker='*', color='magenta', alpha=0.5, s=30, linewidths=0.5)

states = [19, 43, 55, 71, 98, 139, 179]

tic1   = [ data[i][1] for i in states ]
tic2   = [ data[i][3] for i in states ]
ax0.scatter(tic1, tic2, marker='*', color='black', alpha=0.5, s=30, linewidths=0.5)

print(tic1, tic2)

texts  = ['1', '2', '3']

for i, txt in enumerate(texts):
    ax0.annotate(txt, (tic1[i], tic2[i]))

plt.tight_layout()
plt.savefig("tIC1-tIC2-landscape.png", dpi=300, bbox_inches='tight')
