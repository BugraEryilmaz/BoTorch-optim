# %% DATCOM env
import gym
import datcom_gym_env
env = gym.make('Datcom-v1')

# parameters = XLE1, XLE2, CHORD1_1, CHORD1_2, CHORD2_1, CHORD2_2, SSPAN1_2, SSPAN2_2
def evaluate_param(XLE1, XLE2, CHORD1_1, CHORD1_2, CHORD2_1, CHORD2_2, SSPAN1_2, SSPAN2_2):
    defaultXLE1, maxXLE1, minXLE1 = 1.72, 1.75, 1.25
    defaultXLE2, maxXLE2, minXLE2 = 3.2, 3.2, 3.0
    defaultCHORD1_1, maxCHORD1_1, minCHORD1_1 = 0.29, 0.4, 0.1
    defaultCHORD1_2, maxCHORD1_2, minCHORD1_2 = 0.06, 0.09, 0
    defaultCHORD2_1, maxCHORD2_1, minCHORD2_1 = 0.38, 0.4, 0.1
    defaultCHORD2_2, maxCHORD2_2, minCHORD2_2 = 0.19, 0.25, 0
    defaultSSPAN1_2, maxSSPAN1_2, minSSPAN1_2 = 0.23, 0.3, 0.1
    defaultSSPAN2_2, maxSSPAN2_2, minSSPAN2_2 = 0.22, 0.3, 0.1
    
    # Calculate the action taken
    # action = (new - old)/(high - low)
    deltaXLE1 = (XLE1 - defaultXLE1)/(maxXLE1 - minXLE1)
    deltaXLE2 = (XLE2 - defaultXLE2)/(maxXLE2 - minXLE2)
    deltaCHORD1_1 = (CHORD1_1 - defaultCHORD1_1)/(maxCHORD1_1 - minCHORD1_1)
    deltaCHORD1_2 = (CHORD1_2 - defaultCHORD1_2)/(maxCHORD1_2 - minCHORD1_2)
    deltaCHORD2_1 = (CHORD2_1 - defaultCHORD2_1)/(maxCHORD2_1 - minCHORD2_1)
    deltaCHORD2_2 = (CHORD2_2 - defaultCHORD2_2)/(maxCHORD2_2 - minCHORD2_2)
    deltaSSPAN1_2 = (SSPAN1_2 - defaultSSPAN1_2)/(maxSSPAN1_2 - minSSPAN1_2)
    deltaSSPAN2_2 = (SSPAN2_2 - defaultSSPAN2_2)/(maxSSPAN2_2 - minSSPAN2_2)
    action = [deltaXLE1, deltaXLE2, deltaCHORD1_1, deltaCHORD1_2, deltaCHORD2_1, deltaCHORD2_2, deltaSSPAN1_2, deltaSSPAN2_2]
    
    # Calculate normalized state
    # normalizedState = (default-low)/(high-low)
    normalXLE1 = (defaultXLE1 - minXLE1)/(maxXLE1 - minXLE1)
    normalXLE2 = (defaultXLE2 - minXLE2)/(maxXLE2 - minXLE2)
    normalCHORD1_1 = (defaultCHORD1_1 - minCHORD1_1)/(maxCHORD1_1 - minCHORD1_1)
    normalCHORD1_2 = (defaultCHORD1_2 - minCHORD1_2)/(maxCHORD1_2 - minCHORD1_2)
    normalCHORD2_1 = (defaultCHORD2_1 - minCHORD2_1)/(maxCHORD2_1 - minCHORD2_1)
    normalCHORD2_2 = (defaultCHORD2_2 - minCHORD2_2)/(maxCHORD2_2 - minCHORD2_2)
    normalSSPAN1_2 = (defaultSSPAN1_2 - minSSPAN1_2)/(maxSSPAN1_2 - minSSPAN1_2)
    normalSSPAN2_2 = (defaultSSPAN2_2 - minSSPAN2_2)/(maxSSPAN2_2 - minSSPAN2_2)
    normalizedState = [normalXLE1, normalXLE2, normalCHORD1_1, normalCHORD1_2, normalCHORD2_1, normalCHORD2_2, normalSSPAN1_2, normalSSPAN2_2]
    
    # Reset the environment and take the action
    env.reset()
    newState, gain, done, info = env.step(action, normalizedState)
    
    return gain

# %%
def datcom_eval(parameterization, *args):
    XLE1, XLE2, CHORD1_1, CHORD1_2, CHORD2_1, CHORD2_2, SSPAN1_2, SSPAN2_2 = parameterization["XLE1"], \
        parameterization["XLE2"], parameterization["CHORD1_1"], parameterization["CHORD1_2"], parameterization["CHORD2_1"], \
        parameterization["CHORD2_2"], parameterization["SSPAN1_2"], parameterization["SSPAN2_2"]
    gain = evaluate_param(XLE1, XLE2, CHORD1_1, CHORD1_2, CHORD2_1, CHORD2_2, SSPAN1_2, SSPAN2_2)
    return {"datcom_res": (gain, 0.0)}
    
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

datcom_exp = SimpleExperiment(
    name="test_datcom",
    search_space=search_space_datcom,
    evaluation_function=datcom_eval,
    objective_name="datcom_res",
    minimize=False,
)

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
# %%
from botorch.fit import fit_gpytorch_model

def _get_and_fit_simple_custom_gp(Xs, Ys, **kwargs):
    model = SimpleCustomGP(Xs[0], Ys[0])
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model
# %%
import random
import numpy as np
total = 0
def branin(parameterization, *args):
    x1, x2 = parameterization["x1"], parameterization["x2"]
    y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 * x1 / np.pi - 6) ** 2
    y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    global total
    total += 1
    return {"branin": (y, 0.0)}
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
from ax import SimpleExperiment

exp = SimpleExperiment(
    name="test_branin",
    search_space=search_space,
    evaluation_function=branin,
    objective_name="branin",
    minimize=True,
)
# %%
from ax.modelbridge import get_sobol

sobol = get_sobol(exp.search_space)
exp.new_batch_trial(generator_run=sobol.gen(5))
# %%
from ax.modelbridge.factory import get_botorch

for i in range(100):
    print(f"Running optimization batch {i+1}/5...")
    model = get_botorch(
        experiment=exp,
        data=exp.eval(),
        search_space=exp.search_space,
        model_constructor=_get_and_fit_simple_custom_gp,
    )
    batch = exp.new_trial(generator_run=model.gen(1))
    
print("Done!")

# %%
# get the index of minimum value
idxmin = exp.eval().df['mean'].idxmin()
# get the arm name and value at the minimum index
arm_name, optimum_val = exp.eval().df.iloc[idxmin,0], exp.eval().df.iloc[idxmin,2]
# get the parameters for the minimum output
optimum_param = exp.arms_by_name[arm_name].parameters
print(optimum_param, optimum_val)

