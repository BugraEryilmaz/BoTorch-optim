# %% Hyperparameters
NUMBER_OF_INIT_POINTS = 20
NUMBER_OF_ITERATIONS = 100
TRANSFER = True

# %% DATCOM env
from my_datcom_env import myDatcomEnv

datcom = myDatcomEnv(transfer=TRANSFER)
datcom.reset()

print(datcom.base_cd)

ROUND_FACTOR = 4

import torch
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# %% Model definition
from botorch.models.gpytorch import GPyTorchModel, ModelListGPyTorchModel
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.models import ExactGP, IndependentModelList
from gpytorch.kernels import RBFKernel, ScaleKernel, MultitaskKernel
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood, LikelihoodList
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from gpytorch.priors import GammaPrior

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

class MultiOutputGP(IndependentModelList, ModelListGPyTorchModel):
    def __init__(self, model1, model2, model3):
        super().__init__(model1, model2, model3)
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
# %% Fit the model
from botorch.fit import fit_gpytorch_model
from botorch.optim.fit import fit_gpytorch_torch

def _get_and_fit_simple_custom_gp(Xs, Ys, **kwargs):
    model1 = SimpleCustomGP(Xs[0], Ys[0])
    model2 = SimpleCustomGP(Xs[0], Ys[1])
    model3 = SimpleCustomGP(Xs[0], Ys[2])
    model = MultiOutputGP(model1, model2, model3)
    likelihood = LikelihoodList(model1.likelihood, model2.likelihood, model3.likelihood)
    mll = SumMarginalLogLikelihood(likelihood, model)
    fit_gpytorch_model(mll, optimizer=fit_gpytorch_torch)
    return model

# %% DATCOM evaluate    
# parameters = XLE1, XLE2, CHORD1_1, CHORD1_2, CHORD2_1, CHORD2_2, SSPAN1_2, SSPAN2_2
import numpy as np
import random

random.seed(453453)
np.random.seed(453453)
torch.manual_seed(453453)

def datcom_eval(parameterization, *args):
    cl, cd, xcp, cl_cd = datcom.step(parameterization)
    return {"objective": (cl_cd, 0.0), "CD": (cd, 0.0), "XCP": (xcp, 0.0)}
    
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
from ax import Metric

CDmetric = Metric("CD")
XCPmetric = Metric("XCP")
objectivemetric = Metric("objective")

# %%
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.types import ComparisonOp

CDconstraint = OutcomeConstraint(CDmetric, op=ComparisonOp.LEQ, bound=datcom.base_cd, relative=False)
XCPconstraint1 = OutcomeConstraint(XCPmetric, op=ComparisonOp.GEQ, bound=0.53, relative=False)
XCPconstraint2 = OutcomeConstraint(XCPmetric, op=ComparisonOp.LEQ, bound=0.6, relative=False)

# %%
from ax import SimpleExperiment

datcom_exp = SimpleExperiment(
    name="test_datcom",
    search_space=search_space_datcom,
    evaluation_function=datcom_eval,
    objective_name="objective",
    minimize=False,
    outcome_constraints=[CDconstraint, XCPconstraint1, XCPconstraint2],
)

# %% Get initial random points
from ax.modelbridge import get_sobol

sobol = get_sobol(datcom_exp.search_space)
datcom_exp.new_batch_trial(generator_run=sobol.gen(NUMBER_OF_INIT_POINTS))

# %% Optimization loop & results
from ax.modelbridge.factory import get_botorch
import pprint
import pandas as pd
import time

for i in range(NUMBER_OF_ITERATIONS):
    print(f"Running optimization batch {i+1}/{NUMBER_OF_ITERATIONS}...")
    model = get_botorch(
        experiment=datcom_exp,
        data=datcom_exp.eval(),
        search_space=datcom_exp.search_space,
        model_constructor=_get_and_fit_simple_custom_gp,
        device=device,
        transforms=[],
    )
    batch = datcom_exp.new_trial(generator_run=model.gen(1))
    
print("Done!")
# CD<0.453 & 0.53<XCP<0.6 is okey & look for 2.96
df = datcom_exp.fetch_data().df

objective_df = df[df['metric_name']=='objective']
objective_df = objective_df.reset_index(drop=True)
CD_df = df[df['metric_name']=='CD']
CD_df = CD_df.reset_index(drop=True)
XCP_df = df[df['metric_name']=='XCP']
XCP_df = XCP_df.reset_index(drop=True)

new_df = objective_df.join(CD_df, lsuffix=' ', rsuffix='   ').join(XCP_df, lsuffix=' ', rsuffix='   ')

result = [datcom_exp.arms_by_name[item].parameters for item in new_df['arm_name ']]
new_df = new_df.join(pd.DataFrame({'parameters': result}))

constrain = CD_df['mean'] >= datcom.base_cd
constrain2 = XCP_df['mean'] < 0.53
constrain3 = XCP_df['mean'] > 0.6
not_eligible_arms = CD_df[constrain]['arm_name'].append(XCP_df[constrain2]['arm_name']).append(XCP_df[constrain3]['arm_name'])

constrained_objective_df = objective_df[~objective_df['arm_name'].isin(not_eligible_arms)].reset_index(drop=True)
idxmin = constrained_objective_df['mean'].idxmax()
arm_name, optimum_val = constrained_objective_df.iloc[idxmin,0], constrained_objective_df.iloc[idxmin,2]
optimum_param = datcom_exp.arms_by_name[arm_name].parameters

CDbest = CD_df[CD_df['arm_name']==arm_name]
XCPbest = XCP_df[XCP_df['arm_name']==arm_name]
if TRANSFER:
    new_df.to_csv(f'results_constrained_transfer.csv')
else:
    new_df.to_csv(f'results_constrained.csv')

print('Parameters: \n')
pprint.pprint(optimum_param)
print(f'Best CL/CD: {optimum_val}')
print(f'CD: {CDbest["mean"].array[0]}')
print(f'XCP: {XCPbest["mean"].array[0]}')

# %%
