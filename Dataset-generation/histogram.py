# %% A code to generate and draw histograms of both datasets
# Gaussian
import pandas as pd
import matplotlib.pyplot as plt

location_gaus = 'datasets5/'
location_uniform = 'datasets5/'

data = pd.DataFrame(pd.read_csv(f'{location_gaus}data.csv', sep=',', names=["XLE1", "XLE2", "CHORD1_1", "CHORD1_2", "CHORD2_1", "CHORD2_2", "SSPAN1_2", "SSPAN2_2"]))
cl_cd = pd.DataFrame(pd.read_csv(f'{location_gaus}cl_cd.csv', sep=',', names=["CL_CD"]))
cd = pd.DataFrame(pd.read_csv(f'{location_gaus}cd.csv', sep=',', names=["CD"]))
xcp = pd.DataFrame(pd.read_csv(f'{location_gaus}xcp.csv', sep=',', names=["XCP"]))

dataset = data.join(cl_cd).join(cd).join(xcp)
dataset = dataset[cl_cd["CL_CD"] != 0].reset_index(drop=True)

dataset.to_csv(f"{location_gaus}dataset_gaussian.csv", index=False)

# %% uniform
data1 = pd.DataFrame(pd.read_csv(f'{location_uniform}data-uniform.csv', sep=',', names=["XLE1", "XLE2", "CHORD1_1", "CHORD1_2", "CHORD2_1", "CHORD2_2", "SSPAN1_2", "SSPAN2_2"]))
cl_cd1 = pd.DataFrame(pd.read_csv(f'{location_uniform}cl_cd-uniform.csv', sep=',', names=["CL_CD"]))
cd1 = pd.DataFrame(pd.read_csv(f'{location_uniform}cd-uniform.csv', sep=',', names=["CD"]))
xcp1 = pd.DataFrame(pd.read_csv(f'{location_uniform}xcp-uniform.csv', sep=',', names=["XCP"]))

dataset1 = data1.join(cl_cd1).join(cd1).join(xcp1)
dataset1 = dataset1[cl_cd1["CL_CD"] != 0].reset_index(drop=True)

dataset1.to_csv(f"{location_uniform}dataset_uniform.csv", index=False)

# %% Downsampled
# %%
cl_cd.hist()
plt.xlim(1,3.5)
plt.ylim(0,120000)
plt.savefig("hist_gaussian.svg")
# %%
cl_cd1.hist()
plt.xlim(1,3.5)
plt.ylim(0,120000)
plt.savefig("hist_uniform.svg")
