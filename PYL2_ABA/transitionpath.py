from msmbuilder.utils import io
from msmbuilder.tpt import net_fluxes, top_path, paths
import numpy as np

msm = io.load("../../MSMs/MSM40.pkl")

data = np.loadtxt("avg_dist.txt")
lys  = np.transpose(data)[1]
rmsd = np.transpose(data)[3]

sources = []
sinks   = []
for i in range(300):
  if lys[i] < 3.5 and rmsd[i] < 0.3:
    sinks.append(i)
  if lys[i] > 30:
    sources.append(i)

print(sources,sinks)

net_flux = net_fluxes( sources, sinks, msm)
path     = top_path( sources, sinks, net_flux)
paths    = paths(sources, sinks, net_flux, num_paths=10)
#print(paths)
np.save('paths.npy', paths)

data = np.loadtxt("../lys_loop_rmsd/avg_dist.txt")
lys  = np.transpose(data)[1]
rmsd = np.transpose(data)[3]

for path in paths[0]:
  metrics = [ [lys[i], rmsd[i]] for i in path ]
  print(path, metrics)

"""
import networkx as nx
import matplotlib.pyplot as plt
G=nx.DiGraph()
G.add_nodes_from(nodes)
#for path in paths[0]:
#  G.add_path(path)
G.add_weighted_edges_from([(45,48,0.1),(215,5,0.2),(112,6,0)])
print(nx.info(G))
nx.draw(G)
plt.show()
"""
