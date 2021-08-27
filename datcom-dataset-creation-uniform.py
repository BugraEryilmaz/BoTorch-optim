# %% A code to generate gaussian samples and tests them in 50 batches
from numpy import random
import numpy
import numpy as np
import matplotlib.pyplot as plt

# XLE1 XLE2 CHORD1_1 CHORD1_2 CHORD2_1 CHORD2_2 SSPAN1_2 SSPAN2_2
dim = 5
XLE1 = np.hstack((np.linspace(1.25,1.75,dim).reshape((dim,1)), np.zeros((dim,7))))
XLE2 = np.hstack((np.zeros((dim,1)), np.linspace(3.0,3.2,dim).reshape((dim,1)), np.zeros((dim,6))))
CHORD1_1 = np.hstack((np.zeros((dim,2)), np.linspace(0.1,0.4,dim).reshape((dim,1)), np.zeros((dim,5))))
CHORD1_2 = np.hstack((np.zeros((dim,3)), np.linspace(0,0.09,dim).reshape((dim,1)), np.zeros((dim,4))))
CHORD2_1 = np.hstack((np.zeros((dim,4)), np.linspace(0.1,0.4,dim).reshape((dim,1)), np.zeros((dim,3))))
CHORD2_2 = np.hstack((np.zeros((dim,5)), np.linspace(0,0.25,dim).reshape((dim,1)), np.zeros((dim,2))))
SSPAN1_2 = np.hstack((np.zeros((dim,6)), np.linspace(0.1,0.3,dim).reshape((dim,1)), np.zeros((dim,1))))
SSPAN2_2 = np.hstack((np.zeros((dim,7)), np.linspace(0.1,0.3,dim).reshape((dim,1))))

XLE1 = XLE1.reshape((1,1,1,1,1,1,1,dim,8))
XLE2 = XLE2.reshape((1,1,1,1,1,1,dim,1,8))
CHORD1_1 = CHORD1_1.reshape((1,1,1,1,1,dim,1,1,8))
CHORD1_2 = CHORD1_2.reshape((1,1,1,1,dim,1,1,1,8))
CHORD2_1 = CHORD2_1.reshape((1,1,1,dim,1,1,1,1,8))
CHORD2_2 = CHORD2_2.reshape((1,1,dim,1,1,1,1,1,8))
SSPAN1_2 = SSPAN1_2.reshape((1,dim,1,1,1,1,1,1,8))
SSPAN2_2 = SSPAN2_2.reshape((dim,1,1,1,1,1,1,1,8))

dataset = XLE1 + XLE2 + CHORD1_1 + CHORD1_2 + CHORD2_1 + CHORD2_2 + SSPAN1_2 + SSPAN2_2

dataset = dataset.reshape((-1, dim*dim, 8))


# %%
import my_datcom_env
import time

dat = my_datcom_env.myDatcomEnv()
cl_cd, xcp, cd = np.zeros(dataset.shape[0:2]), np.zeros(dataset.shape[0:2]), np.zeros(dataset.shape[0:2])
index = 0
for points in dataset:
    try:
        cl_cd[index,:], xcp[index,:], cd[index,:] = dat.step_batch(dataset[index])
    except:
        print(dataset[index])
        time.sleep(5)
    index += 1

cl_cd = cl_cd.flatten()
xcp =xcp.flatten()
cd = cd.flatten()
dataset = dataset.reshape((-1, 8))
numpy.savetxt('cl_cd-uniform.csv', cl_cd,delimiter=',')
numpy.savetxt('xcp-uniform.csv', xcp,delimiter=',')
numpy.savetxt('cd-uniform.csv', cd,delimiter=',')
numpy.savetxt('data-uniform.csv', dataset,delimiter=',')
# %%
