
# %%
from botorch.models.gpytorch import GPyTorchModel, ModelListGPyTorchModel
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.models import ExactGP, IndependentModelList
from gpytorch.kernels import RBFKernel, ScaleKernel, MultitaskKernel
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood, LikelihoodList
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from gpytorch.priors import GammaPrior

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    def __init__(self, model1, model2):
        super().__init__(model1, model2)
        self.model1 = model1
        self.model2 = model2
        
# %%
from botorch.fit import fit_gpytorch_model

def _get_and_fit_simple_custom_gp(Xs, Ys, **kwargs):
    model1 = SimpleCustomGP(Xs[0], Ys[0])
    mll1 = ExactMarginalLogLikelihood(model1.likelihood, model1)
    model2 = SimpleCustomGP(Xs[0], Ys[1])
    mll2 = ExactMarginalLogLikelihood(model2.likelihood, model2)
    model = MultiOutputGP(model1, model2)
    likelihood = LikelihoodList(model1.likelihood, model2.likelihood)
    mll = SumMarginalLogLikelihood(likelihood, model)
    fit_gpytorch_model(mll)
    return model

# %%
import random
import numpy as np

random.seed(192)
np.random.seed(192)
torch.manual_seed(192)

def branin(parameterization, *args):
    x1, x2 = parameterization["x1"], parameterization["x2"]
    y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 * x1 / np.pi - 6) ** 2
    y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    return {"branin": (y, 0.0), "CD": (y*y,0.0)}

# %%
from ax import ParameterType, RangeParameter, SearchSpace

search_space = SearchSpace(
    parameters=[
        RangeParameter(
            name="x1", parameter_type=ParameterType.FLOAT, lower=-5, upper=10
        ),
        RangeParameter(
            name="x2", parameter_type=ParameterType.FLOAT, lower=0, upper=15
        ),
    ]
)

# %%
import pandas as pd
from ax import Metric
from ax.core.data import Data

class CDmetric_class(Metric):
    def f(self, x: np.ndarray) -> float:
        return float(branin(torch.tensor(x))[1])

class Braninmetric_class(Metric):
    def f(self, x: np.ndarray) -> float:
        return float(branin(torch.tensor(x))[0])



CDmetric = CDmetric_class("CD")

braninmetric = Braninmetric_class("branin")



# %%
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.types import ComparisonOp

CDconstraint = OutcomeConstraint(CDmetric, op=ComparisonOp.GEQ, bound=0.25, relative=False)

# %%
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
opt_config = OptimizationConfig(
        objective=Objective(braninmetric, minimize=True),
        outcome_constraints=[CDconstraint]
    )
# %%
from ax import SimpleExperiment, Experiment

exp = SimpleExperiment(
    name="test_branin",
    search_space=search_space,
    evaluation_function=branin,
    objective_name="branin",
    minimize=True,
    outcome_constraints=[CDconstraint]
)
# %%
from ax.modelbridge import get_sobol


sobol = get_sobol(exp.search_space)
exp.new_batch_trial(generator_run=sobol.gen(5))
# %%
from ax.modelbridge.factory import get_botorch

for i in range(50):
    print(f"Running optimization batch {i+1}/50...")
    model = get_botorch(
        experiment=exp,
        data=exp.eval(),
        search_space=exp.search_space,
        model_constructor=_get_and_fit_simple_custom_gp,
        device=device,
    )
    batch = exp.new_trial(generator_run=model.gen(1,))
print("Done!")

# %%
df = exp.fetch_data().df
branin_df = df[df['metric_name']=='branin']
CD_df = df[df['metric_name']=='CD']
constrain = CD_df['mean'] >= 0.25
eligible_arms = CD_df[constrain]['arm_name']
constrained_branin_df = branin_df[branin_df['arm_name'].isin(eligible_arms)]
idxmin = constrained_branin_df['mean'].idxmin()
arm_name, optimum_val = df.iloc[idxmin,0], df.iloc[idxmin,2]
optimum_param = exp.arms_by_name[arm_name].parameters
print(optimum_param, optimum_val)
