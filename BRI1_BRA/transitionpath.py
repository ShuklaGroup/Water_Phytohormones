from msmbuilder.utils import io
from msmbuilder.tpt import net_fluxes, top_path, paths
import numpy as np

msm = io.load("../MSMs/msm.pkl")

data = np.loadtxt("avg_dist.txt")
lys  = np.transpose(data)[3]
rmsd = np.transpose(data)[1]

sources = []
sinks   = []
for i in range(200):
  if lys[i] < 0.35 and rmsd[i] < 0.15:
    sinks.append(i)
  if lys[i] > 4.5:
    sources.append(i)

print("Sources:", sources, "\n")
print("Sink:", sinks, "\n")

net_flux = net_fluxes( sources, sinks, msm)
#path     = top_path( sources, sinks, net_flux)
paths    = paths(sources, sinks, net_flux, num_paths=10)
print(paths)
print(np.unique(np.hstack(paths[0])))
np.save('paths.npy', paths)
