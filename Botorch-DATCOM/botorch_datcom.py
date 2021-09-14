# %% Hyperparameters
NUMBER_OF_INIT_POINTS = 10
NUMBER_OF_ITERATIONS = 50

# %% DATCOM env
# to add file from parent directory
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from my_datcom_env import myDatcomEnv
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--transfer', dest='transfer', action='store_true', default=False, help='''To run without altered parameters. Default is with altered parameters''')
args = parser.parse_args()

datcom = myDatcomEnv(transfer=args.transfer)
datcom.reset()

ROUND_FACTOR = 4

import torch
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# %% Model definition
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
import pandas as pd

class SimpleCustomGP(ExactGP, GPyTorchModel):

    _num_outputs = 1  # to inform GPyTorchModel API
    
    def __init__(self, train_X, train_Y):
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=RBFKernel(ard_num_dims=train_X.shape[-1]),
        )
        self.to(train_X)  # make sure we're on the right device/dtype
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

# %% Fit the model
from botorch.fit import fit_gpytorch_model

def _get_and_fit_simple_custom_gp(Xs, Ys, **kwargs):
    model = SimpleCustomGP(Xs[0], Ys[0])
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model

# %% DATCOM evaluate    
# parameters = XLE1, XLE2, CHORD1_1, CHORD1_2, CHORD2_1, CHORD2_2, SSPAN1_2, SSPAN2_2
import numpy as np
import random

random.seed(100)
np.random.seed(100)
torch.manual_seed(100)

def datcom_eval(parameterization, *args):
    cl, cd, xcp, cl_cd = datcom.step(parameterization)
    print(cl_cd)
    return {"objective": (cl_cd, 0.0)}
    
# %%
from ax import ParameterType, RangeParameter, SearchSpace

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

# %%
from ax import SimpleExperiment
import time


datcom_exp = SimpleExperiment(
    name="test_datcom",
    search_space=search_space_datcom,
    evaluation_function=datcom_eval,
    objective_name="objective",
    minimize=False,
)

# %% Get initial random points
from ax.modelbridge import get_sobol

sobol = get_sobol(datcom_exp.search_space)
datcom_exp.new_batch_trial(generator_run=sobol.gen(NUMBER_OF_INIT_POINTS))

# %% Optimization loop & results
from ax.modelbridge.factory import get_botorch
import pprint

for i in range(NUMBER_OF_ITERATIONS):
    print(f"Running optimization batch {i+1}/{NUMBER_OF_ITERATIONS}...")
    model = get_botorch(
        experiment=datcom_exp,
        data=datcom_exp.eval(),
        search_space=datcom_exp.search_space,
        model_constructor=_get_and_fit_simple_custom_gp,
        device=device,
    )
    batch = datcom_exp.new_trial(generator_run=model.gen(1))
    
print("Done!")

# CD<0.453 & 0.53<XCP<0.6 is okey & look for 2.96
df = datcom_exp.fetch_data().df

objective_df = df[df['metric_name']=='objective']
objective_df = objective_df.reset_index(drop=True)

new_df = objective_df

result = [datcom_exp.arms_by_name[item].parameters for item in new_df['arm_name']]
new_df = new_df.join(pd.DataFrame({'parameters': result}))

constrained_objective_df = objective_df
idxmin = constrained_objective_df['mean'].idxmax()
arm_name, optimum_val = constrained_objective_df.iloc[idxmin,0], constrained_objective_df.iloc[idxmin,2]
optimum_param = datcom_exp.arms_by_name[arm_name].parameters

new_df.to_csv(f'results{"_transfer" if args.transfer else ""}.csv')

print('Parameters: \n')
pprint.pprint(optimum_param)
print(f'Best CL/CD: {optimum_val}')
"""
# get the index of minimum value
idxmax = datcom_exp.eval().df['mean'].idxmax()
# get the arm name and value at the minimum index
arm_name, optimum_val = datcom_exp.eval().df.iloc[idxmax,0], datcom_exp.eval().df.iloc[idxmax,2]
# get the parameters for the minimum output
optimum_param = datcom_exp.arms_by_name[arm_name].parameters
# get cl/cd
print('Parameters: \n')
pprint.pprint(optimum_param)
print(f'Best CL/CD: {optimum_val}')
"""


# %%
