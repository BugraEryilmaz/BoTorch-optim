# %% A code to generate gaussian samples and tests them in 50 batches
from numpy import random
import numpy
import matplotlib.pyplot as plt

# XLE1 XLE2 CHORD1_1 CHORD1_2 CHORD2_1 CHORD2_2 SSPAN1_2 SSPAN2_2
batch_number = 7813
test_points = random.normal(
        [1.5, 3.1, 0.25, 0.045, 0.25, 0.125, 0.2, 0.2], 
        [0.25/2, 0.1/2, 0.15/2, 0.045/2, 0.15/2, 0.125/2, 0.1/2, 0.1/2], 
        (batch_number,50,8))
clipped_points = test_points.clip(
        [1.25, 3.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.1],
        [1.75, 3.2, 0.4, 0.09, 0.4, 0.25, 0.3, 0.3])


# %%
import my_datcom_env

dat = my_datcom_env.myDatcomEnv()
cl_cd, xcp, cd = numpy.empty((batch_number,50)), numpy.empty((batch_number,50)), numpy.empty((batch_number,50))
index = 0
for points in clipped_points:
    try:
        cl_cd[index,:], xcp[index,:], cd[index,:] = dat.step_batch(clipped_points[index])
    except:
        print("exception")
    index += 1

cl_cd = cl_cd.flatten()
xcp =xcp.flatten()
cd = cd.flatten()
clipped_points = clipped_points.reshape((-1, 8))
numpy.savetxt('datasets/cl_cd.csv', cl_cd,delimiter=',')
numpy.savetxt('datasets/xcp.csv', xcp,delimiter=',')
numpy.savetxt('datasets/cd.csv', cd,delimiter=',')
numpy.savetxt('datasets/data.csv', clipped_points,delimiter=',')
# %%
