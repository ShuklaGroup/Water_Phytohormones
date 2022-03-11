from msmbuilder.utils import io
from msmbuilder.tpt import net_fluxes, top_path, paths
import numpy as np

msm = io.load("../MSMs/MSM20.pkl")

data = np.loadtxt("avg_dist.txt")
lys  = np.transpose(data)[1]
rmsd = np.transpose(data)[3]

sources = []
sinks   = []
for i in range(325):
  if rmsd[i] < 0.7:
    sinks.append(i)
  if rmsd[i] > 3.5:
    sources.append(i)

print("Sources:", sources, "\n")
print("Sink:", sinks, "\n")

net_flux = net_fluxes( sources, sinks, msm)
#path     = top_path( sources, sinks, net_flux)
paths    = paths(sources, sinks, net_flux, num_paths=100)
print(paths)
np.save("paths.npy", paths[0])
