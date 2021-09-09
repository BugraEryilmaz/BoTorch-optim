# %% A code to generate gaussian samples and tests them in 50 batches
from numpy import random
import numpy
import numpy as np
import matplotlib.pyplot as plt

# XLE1 XLE2 CHORD1_1 CHORD1_2 CHORD2_1 CHORD2_2 SSPAN1_2 SSPAN2_2
from ax import ParameterType, RangeParameter, SearchSpace
from ax.modelbridge import get_sobol
from torch._C import set_anomaly_enabled

search_space_datcom = SearchSpace(
    parameters=[
        RangeParameter(
            name="XLE1", parameter_type=ParameterType.FLOAT, lower=1.25, upper=1.75
        ),
        RangeParameter(
            name="XLE2", parameter_type=ParameterType.FLOAT, lower=3, upper=3.2
        ),
        RangeParameter(
            name="CHORD1_1", parameter_type=ParameterType.FLOAT, lower=0.1, upper=0.4
        ),
        RangeParameter(
            name="CHORD1_2", parameter_type=ParameterType.FLOAT, lower=0, upper=0.09
        ),
        RangeParameter(
            name="CHORD2_1", parameter_type=ParameterType.FLOAT, lower=0.1, upper=0.4
        ),
        RangeParameter(
            name="CHORD2_2", parameter_type=ParameterType.FLOAT, lower=0, upper=0.25
        ),
        RangeParameter(
            name="SSPAN1_2", parameter_type=ParameterType.FLOAT, lower=0.1, upper=0.3
        ),
        RangeParameter(
            name="SSPAN2_2", parameter_type=ParameterType.FLOAT, lower=0.1, upper=0.3
        ),
    ]
)

dataset = np.zeros((2,25,8))

sobol = get_sobol(search_space_datcom)

gr = sobol.gen(50)
for i, arm in enumerate(gr.arms):
    param = arm.parameters
    dataset[i//25, i%25] = np.asarray([param['XLE1'], param['XLE2'], param['CHORD1_1'], param['CHORD1_2'], param['CHORD2_1'], param['CHORD2_2'], param['SSPAN1_2'], param['SSPAN2_2']])
# %%
import my_datcom_env
import time

dat = my_datcom_env.myDatcomEnv(transfer=True)
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
numpy.savetxt(f'cl_cd-uniform.csv', cl_cd,delimiter=',')
numpy.savetxt(f'xcp-uniform.csv', xcp,delimiter=',')
numpy.savetxt(f'cd-uniform.csv', cd,delimiter=',')
numpy.savetxt(f'data-uniform.csv', dataset,delimiter=',')
# %%
