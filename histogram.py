# %% A code to generate gaussian dataset and draw histograms of both datasets
import pandas as pd
import matplotlib.pyplot as plt

data = pd.DataFrame(pd.read_csv('data.csv', sep=','))
cl_cd = pd.DataFrame(pd.read_csv('cl_cd.csv', sep=','))
cd = pd.DataFrame(pd.read_csv('cd.csv', sep=','))
xcp = pd.DataFrame(pd.read_csv('xcp.csv', sep=','))
# %%
dataset = data.join(cl_cd).join(cd).join(xcp)
dataset = dataset[cl_cd["CL_CD"] != 0].reset_index(drop=True)

dataset.to_csv("dataset_gaussian.csv")

# %%
data1 = pd.DataFrame(pd.read_csv('results-5.csv', sep=',')["CL_CD"])



# %%
cl_cd.hist()
plt.xlim(1,3.5)
plt.ylim(0,120000)
plt.savefig("hist_gaussian.svg")
# %%
data1.hist()
plt.xlim(1,3.5)
plt.ylim(0,120000)
plt.savefig("hist_uniform.svg")
