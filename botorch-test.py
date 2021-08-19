
# %%
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior


class SimpleCustomGP(ExactGP, GPyTorchModel):

    _num_outputs = 2  # to inform GPyTorchModel API
    
    def __init__(self, train_X, train_Y):
        # squeeze output dim before passing train_Y to ExactGP
        print(train_Y.squeeze(-1))
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
# %%
from botorch.fit import fit_gpytorch_model

def _get_and_fit_simple_custom_gp(Xs, Ys, **kwargs):
    print(Ys)
    model = SimpleCustomGP(Xs[0], Ys[0])
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model
# %%
import random
import numpy as np
def branin(parameterization, *args):
    x1, x2 = parameterization["x1"], parameterization["x2"]
    y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 * x1 / np.pi - 6) ** 2
    y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    return {"branin": (y, 0.0), "CD": (y,0.0)}

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

CDmetric = Metric("CD")

braninmetric = Metric("branin")



# %%
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.types import ComparisonOp

CDconstraint = OutcomeConstraint(CDmetric, op=ComparisonOp.GEQ, bound=0.5, relative=False)

# %%
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
opt_config = OptimizationConfig(
        objective=Objective(braninmetric, minimize=True),
        outcome_constraints=[CDconstraint]
    )
# %%
from ax import SimpleExperiment

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

for i in range(20):
    print(f"Running optimization batch {i+1}/20...")
    model = get_botorch(
        experiment=exp,
        data=exp.eval(),
        search_space=exp.search_space,
        model_constructor=_get_and_fit_simple_custom_gp,
    )
    batch = exp.new_trial(generator_run=model.gen(1,))
print("Done!")

# %%
# get the index of minimum value
idxmin = exp.eval().df['mean'].idxmin()
# get the arm name and value at the minimum index
arm_name, optimum_val = exp.eval().df.iloc[idxmin,0], exp.eval().df.iloc[idxmin,2]
# get the parameters for the minimum output
optimum_param = exp.arms_by_name[arm_name].parameters
print(optimum_param, optimum_val)

