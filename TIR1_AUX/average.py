import numpy as np

dist1 = np.loadtxt("dat1.dat")
dist2 = np.loadtxt("dat2.dat") 
count = np.loadtxt("count_mat.dat")

sums  = []
avg_dist1  = []
err_dist1  = []
avg_dist2  = []
err_dist2  = [] 

sum = 0
for i in range(len(count)):
    sum = sum + count[i]
    sums.append(int(sum))

for i in range(len(count)):
    if i == 0:
        avg_dist1.append(np.mean(dist1[0:sums[i]]))
        err_dist1.append(np.std(dist1[0:sums[i]]))
        avg_dist2.append(np.mean(dist2[0:sums[i]]))
        err_dist2.append(np.std(dist2[0:sums[i]]))
    if i > 0:
        avg_dist1.append(np.mean(dist1[sums[i-1]:sums[i]]))
        err_dist1.append(np.std(dist1[sums[i-1]:sums[i]]))
        avg_dist2.append(np.mean(dist2[sums[i-1]:sums[i]]))
        err_dist2.append(np.std(dist2[sums[i-1]:sums[i]]))

states = [ state for state in range(len(count)) ]

dataset = np.transpose([ states, avg_dist1, err_dist1, avg_dist2, err_dist2 ])
print(np.argmax(avg_dist1))
np.savetxt("avg_dist.txt", dataset)

"""
classes = []
for i in range(len(count)):
    if avg_dist1[i] < 4 and avg_dist2[i] < 0.3:
        classes.append(0)
    if avg_dist1[i] < 4 and avg_dist2[i] >= 0.3:
        classes.append(1)
    if avg_dist1[i] >= 4 and avg_dist2[i] >= 0.3:
        classes.append(2)
    if avg_dist1[i] >= 4 and avg_dist2[i] < 0.3:
        classes.append(3)

np.savetxt("categories.dat", classes)
"""
