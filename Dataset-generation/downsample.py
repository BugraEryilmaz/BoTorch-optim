# %%
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Parse the DATCOM output file')
parser.add_argument('dataset_loc',nargs='?',help='''The location of the dataset to be used. Default: "datasets6/dataset_uniform"''',default='datasets6/dataset_uniform')         

args = parser.parse_args()

location = args.dataset_loc

dataset = pd.DataFrame(pd.read_csv(f'{location}.csv', sep=','))

dfs = {}

bin_size = 0.2
bin_count = int((3.1-2.0)//bin_size)

max_size = dataset[(dataset['CL_CD']>3.1-bin_size) & (dataset['CL_CD']<3.1)].shape[0]

print(dfs)
for i in np.linspace(3.1-bin_count*bin_size, 3.1-bin_size, bin_count):
    dfs[f'{i:.1f}_to_{i+bin_size:.1f}'] = dataset[(dataset['CL_CD']>i) & (dataset['CL_CD']<i+bin_size)]
    dfs[f'{i:.1f}_to_{i+bin_size:.1f}'] = dfs[f'{i:.1f}_to_{i+bin_size:.1f}'].sample(n=max_size, axis=0)

# %%
new_dataset = pd.concat(dfs, ignore_index=True)
new_dataset.to_csv(f'{location}_downsampled_{bin_size:.1f}binsize.csv')
