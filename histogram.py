# %% A code to generate and draw histograms of both datasets
# Gaussian
import pandas as pd
import matplotlib.pyplot as plt

data = pd.DataFrame(pd.read_csv('datasets/data.csv', sep=','))
cl_cd = pd.DataFrame(pd.read_csv('datasets/cl_cd.csv', sep=','))
cd = pd.DataFrame(pd.read_csv('datasets/cd.csv', sep=','))
xcp = pd.DataFrame(pd.read_csv('datasets/xcp.csv', sep=','))

dataset = data.join(cl_cd).join(cd).join(xcp)
dataset = dataset[cl_cd["CL_CD"] != 0].reset_index(drop=True)

dataset.to_csv("datasets/dataset_gaussian.csv", index=False)

# %% uniform
data1 = pd.DataFrame(pd.read_csv('datasets/data-uniform.csv', sep=','))
cl_cd1 = pd.DataFrame(pd.read_csv('datasets/cl_cd-uniform.csv', sep=','))
cd1 = pd.DataFrame(pd.read_csv('datasets/cd-uniform.csv', sep=','))
xcp1 = pd.DataFrame(pd.read_csv('datasets/xcp-uniform.csv', sep=','))

dataset1 = data1.join(cl_cd1).join(cd1).join(xcp1)
dataset1 = dataset1[cl_cd1["CL_CD"] != 0].reset_index(drop=True)

dataset1.to_csv("datasets/dataset_uniform.csv", index=False)


# %%
cl_cd.hist()
plt.xlim(1,3.5)
plt.ylim(0,120000)
plt.savefig("images/hist_gaussian.svg")
# %%
cl_cd1.hist()
plt.xlim(1,3.5)
plt.ylim(0,120000)
plt.savefig("images/hist_uniform.svg")
