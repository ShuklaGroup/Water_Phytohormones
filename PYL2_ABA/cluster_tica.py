import numpy as np
import msmbuilder.cluster
import glob
from msmbuilder.utils import io

dataset = np.load('pyl2_mono_features.npy')

from msmbuilder.decomposition import tICA
tica = tICA(n_components=4, lag_time=1)
tica.fit(dataset)
tica_traj = tica.transform(dataset)
np.save('tica_traj', tica_traj)

states = msmbuilder.cluster.KMeans(n_clusters=300)
states.fit(tica_traj)
io.dump(states,'clustering_tica.pkl')
