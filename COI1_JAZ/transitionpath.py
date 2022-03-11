from msmbuilder.utils import io
from msmbuilder.tpt import net_fluxes, top_path, paths
import numpy as np

msm = io.load("../MSMs/MSM30.pkl")

data = np.loadtxt("avg_dist.txt")
lys  = np.transpose(data)[1]
rmsd = np.transpose(data)[3]

sources = []
sinks   = []
for i in range(190):
  if lys[i] < 0.35 and rmsd[i] < 6:
    sinks.append(i)
  if rmsd[i] > 30:
    sources.append(i)

print(sources,sinks)

net_flux = net_fluxes( sources, sinks, msm)
#path     = top_path( sources, sinks, net_flux)
paths    = paths(sources, sinks, net_flux, num_paths=10)
np.save('paths.npy', paths)
print(paths)
