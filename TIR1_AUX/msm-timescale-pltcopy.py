import pickle
import numpy as np
from msmbuilder.msm import MarkovStateModel
from msmbuilder.msm import implied_timescales
import pylab as plt
import matplotlib as mpl
from msmbuilder.utils import io

import matplotlib.pyplot as plt
import math
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'], 'size': '16', 'weight':'bold'})
params = {'mathtext.default': 'regular' }
plt.rcParams.update(params)

from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('16')

cl = pickle.load(open('clustering_tica_normalized_inverse.pkl','rb'))
n_timescales=10
stepS = 0.1
lag_times=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
lag_times=[ lag_t * 18 for lag_t in lag_times ] 
l = len(lag_times)

ts=np.zeros([10,l])

ns_lt=np.ndarray.tolist(stepS*np.array(lag_times))
index = 0


for i in lag_times:
    msm=MarkovStateModel(lag_time=i, n_timescales=n_timescales)
    msm.fit_transform(cl.labels_)
    ts[:,index]=msm.timescales_
    index=index+1
#   io.dump(msm,'MSM'+str(i)+'.pkl')

"""
for i in lag_times:
    msm = io.load('MSM'+str(i)+'.pkl')
    ts[:,index]=msm.timescales_
    index=index+1
"""

fig, ax = plt.subplots(1,1)

ax.set_xlim(0,50)
ax.set_ylim(1,10000)

for i in range(10):
  j=i+1
  if j==1:
    k='st'
  elif j==2:
    k='nd'
  elif j==3:
    k='rd'
  elif j>3:
    k='th'
  l=str(j)+k
  ax.plot(ns_lt[0:-1],stepS*ts[i,0:-1],'o',label="%s timescale" %l)



fig.set_figheight(4)
fig.set_figwidth(5.5)

#fig.set_size_inches(3.5, 2.333)
#fig.set_size_inches(3.5, 2.333)

ax.set_xlabel('Lag time (ns)', fontweight="bold")
ax.set_ylabel('Implied timescales (ns)', fontweight="bold")
ax.semilogy()

fig.savefig('test3.png',dpi=300, bbox_inches='tight')
fig.show()
