import pickle
import numpy as np
from msmbuilder.msm import MarkovStateModel
from msmbuilder.msm import implied_timescales
import pylab as plt
import matplotlib as mpl
from msmbuilder.utils import io

font = {'family':'Times New Roman', 'size': 12}
plt.rc('font', **font)
cl = pickle.load(open('clustering_tica_normalized_inverse.pkl','rb'))
n_timescales=10

msm=MarkovStateModel(lag_time=300, n_timescales=n_timescales)
msm.fit_transform(cl.labels_)
io.dump(msm,'MSM30.pkl')
