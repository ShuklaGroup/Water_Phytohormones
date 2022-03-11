from msmbuilder.utils import io
from msmbuilder.tpt import net_fluxes, top_path, paths
import numpy as np

msm = io.load("../MSMs/MSM30.pkl")

data = np.loadtxt("avg_dist.txt")
lys  = np.transpose(data)[1]
rmsd = np.transpose(data)[3]
print(np.min(lys),  np.max(lys) )
print(np.min(rmsd), np.max(rmsd))

sources = []
sinks   = []
for i in range(200):
  if lys[i] < 0.4 and rmsd[i] < 4:
    sinks.append(i)
  if lys[i] > 5:
    sources.append(i)

print(sources,sinks)

net_flux = net_fluxes( sources, sinks, msm)
#path     = top_path( sources, sinks, net_flux)
paths    = paths(sources, sinks, net_flux, num_paths=10)
print(paths)
np.save('paths.npy', paths)
