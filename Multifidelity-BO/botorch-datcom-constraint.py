# %%
from argparse import Namespace

from ax.models.torch.botorch_defaults import get_NEI


SMOKE_TEST = True

# %% Hyperparameters
NUMBER_OF_ITERATIONS = 300 if not SMOKE_TEST else 2


if not SMOKE_TEST:

    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--grid', dest='grid', action='store_true', default=False, help='''To run grid search''')
    args = parser.parse_args()
else:
    args = Namespace(grid=False)

# %%

best_of_best_cl_cd = 0
best_of_best_iter = 0
best_of_best_seed= 0
best_of_best_NUMBER_OF_INIT_POINTS= 0
# %% Constraint function
def const(cd, xcp):
    return True if cd < datcom_high.base_cd and xcp < 0.6 and xcp > 0.53 else False
    
seed_range = range(101,112) if args.grid else [104] 
Init_range = [50,2,5,10,20] if args.grid else [5]

for seed in seed_range:
    for NUMBER_OF_INIT_POINTS in Init_range:
            # %% DATCOM env
            import os, sys
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from my_datcom_env import myDatcomEnv

            datcom_low = myDatcomEnv()
            datcom_low.reset()

            datcom_high = myDatcomEnv(transfer=True)
            datcom_high.reset()

            CD_ratio = datcom_high.base_cd/datcom_low.base_cd
            print(datcom_high.base_cd)
            print(CD_ratio)

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
            from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP

            class MultiOutputGP(IndependentModelList, ModelListGPyTorchModel):
                def __init__(self, model1, model2, model3):
                    super().__init__(model1, model2, model3)
                    self.model1 = model1
                    self.model2 = model2
                    self.model3 = model3
                def condition_on_observations(self, X, Y, **kwargs):
                    model1 = self.model1.condition_on_observations(X, Y)
                    model2 = self.model2.condition_on_observations(X, Y)
                    model3 = self.model3.condition_on_observations(X, Y)
                    model = MultiOutputGP(model1, model2, model3)
                    return model
            # %% Fit the model
            from botorch.fit import fit_gpytorch_model
            from botorch.optim.fit import fit_gpytorch_torch

            def _get_and_fit_simple_custom_gp(Xs, Ys, **kwargs):
                model1 = SingleTaskMultiFidelityGP(Xs[0], Ys[0], data_fidelity=8)
                model2 = SingleTaskMultiFidelityGP(Xs[0], Ys[1], data_fidelity=8)
                model3 = SingleTaskMultiFidelityGP(Xs[0], Ys[2], data_fidelity=8)
                model = MultiOutputGP(model1, model2, model3)
                likelihood = LikelihoodList(model1.likelihood, model2.likelihood, model3.likelihood)
                mll = SumMarginalLogLikelihood(likelihood, model)
                fit_gpytorch_model(mll, fit_gpytorch_torch)
                return model

            # %% Acqf getter
            from botorch import fit_gpytorch_model
            from botorch.models.cost import AffineFidelityCostModel
            from botorch.acquisition.cost_aware import InverseCostWeightedUtility
            from botorch.acquisition import PosteriorMean
            from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
            from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
            from botorch.optim.optimize import optimize_acqf
            from botorch.acquisition.utils import project_to_target_fidelity
            from botorch.acquisition.objective import IdentityMCObjective, MCAcquisitionObjective
            from botorch.acquisition.acquisition import AcquisitionFunction
            from botorch.models.model import Model
            from typing import Any, Callable, Dict, List, Optional, Tuple
            from typing import Any, Optional, Union
            from torch import Tensor
            from botorch.sampling.samplers import MCSampler, SobolQMCNormalSampler
            from typing import Callable, Dict, List, Optional
            from botorch.acquisition.objective import ConstrainedMCObjective, GenericMCObjective
            from botorch.acquisition.monte_carlo import MCAcquisitionFunction, qSimpleRegret
            from botorch.acquisition.utils import get_acquisition_function, get_infeasible_cost
            from botorch.utils.transforms import (
                concatenate_pending_points,
                match_batch_shape,
                t_batch_mode_transform,
            )
            from botorch.utils import (
                get_objective_weights_transform,
                get_outcome_constraint_transforms,
            )

            bounds = torch.tensor([[1.2500, 3.0000, 0.1000, 0.0000, 0.1000, 0.0000, 0.1000, 0.1000, 0.0000], \
                [1.7500, 3.2000, 0.4000, 0.0900, 0.4000, 0.2500, 0.3000, 0.3000, 1.0000]]).to(device=device)
            target_fidelities = {8: 1}

            cost_model = AffineFidelityCostModel(fidelity_weights={8: 1}, fixed_cost=5.0)
            cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

            def project(X):
                return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)

            def get_mfkg(
                model,
                objective_weights,
                outcome_constraints = None,
                X_observed = None,
                X_pending = None,
                **kwargs,):

                print(bounds)

                obj_tf = get_objective_weights_transform(objective_weights)

                def objective(samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
                    return obj_tf(samples)

                con_tfs = get_outcome_constraint_transforms(outcome_constraints)
                inf_cost = get_infeasible_cost(X=X_observed, model=model, objective=objective)
                objective = ConstrainedMCObjective(
                    objective=objective, constraints=con_tfs or [], infeasible_cost=inf_cost
                )
                
                curr_val_acqf = FixedFeatureAcquisitionFunction(
                    acq_function=qSimpleRegret(model, objective=objective),
                    d=9,
                    columns=[8],
                    values=[1],
                )
                
                _, current_value = optimize_acqf(
                    acq_function=curr_val_acqf,
                    bounds=bounds[:,:-1],
                    q=1,
                    num_restarts=10 if not SMOKE_TEST else 2,
                    raw_samples=1024 if not SMOKE_TEST else 4,
                    options={"batch_limit": 10, "maxiter": 200},
                )
                    
                return qMultiFidelityKnowledgeGradient(
                    model=model,
                    objective=objective,
                    num_fantasies=128 if not SMOKE_TEST else 2,
                    current_value=current_value,
                    cost_aware_utility=cost_aware_utility,
                    project=project,
                )

            # %% optimize acqf
            from botorch.optim.initializers import gen_one_shot_kg_initial_conditions
            from botorch.optim.optimize import optimize_acqf_mixed

            NUM_RESTARTS = 10 if not SMOKE_TEST else 2
            RAW_SAMPLES = 512 if not SMOKE_TEST else 4


            def optimize_mfkg_and_get_observation(
                acq_function,
                bounds,
                n,
                inequality_constraints,
                fixed_features,
                rounding_func,
                **optimizer_options,):
                """Optimizes MFKG and returns a new candidate, observation, and cost."""
                
                print(bounds)
                X_init = gen_one_shot_kg_initial_conditions(
                    acq_function = acq_function,
                    bounds=bounds,
                    q=n,
                    num_restarts=10,
                    raw_samples=512,
                )
                candidates, vals = optimize_acqf_mixed(
                    acq_function=acq_function,
                    bounds=bounds,
                    fixed_features_list=[{8: 0}, {8: 1}],
                    q=n,
                    num_restarts=NUM_RESTARTS,
                    raw_samples=RAW_SAMPLES,
                    batch_initial_conditions=X_init,
                    inequality_constraints=inequality_constraints,
                    options={"batch_limit": 5, "maxiter": 200},
                )
                return candidates, vals

            # %% DATCOM evaluate    
            # parameters = XLE1, XLE2, CHORD1_1, CHORD1_2, CHORD2_1, CHORD2_2, SSPAN1_2, SSPAN2_2
            import numpy as np
            import random

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            def datcom_eval(parameterization, *args):
                if parameterization['fidelity'] == 1:
                    cl, cd, xcp, cl_cd = datcom_high.step(parameterization)
                else:
                    cl, cd, xcp, cl_cd = datcom_low.step(parameterization)
                if const(cd, xcp):
                    print(f"VALID! CL_CD: {cl_cd}")
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
                    RangeParameter(
                        name="fidelity", parameter_type=ParameterType.INT, lower=0, upper=1
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

            CDconstraint = OutcomeConstraint(CDmetric, op=ComparisonOp.LEQ, bound=datcom_high.base_cd, relative=False)
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
            from botorch.optim.optimize import optimize_acqf_mixed

            for i in range(NUMBER_OF_ITERATIONS):
                print(f"Running optimization batch {i+1}/{NUMBER_OF_ITERATIONS}...")
                model = get_botorch(
                    experiment=datcom_exp,
                    data=datcom_exp.eval(),
                    search_space=datcom_exp.search_space,
                    model_constructor=_get_and_fit_simple_custom_gp,
                    device=device,
                    acqf_constructor=get_mfkg,
                    acqf_optimizer=optimize_mfkg_and_get_observation,
                    transforms=[]
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

            constrain = CD_df['mean'] >= datcom_high.base_cd
            constrain2 = XCP_df['mean'] < 0.53
            constrain3 = XCP_df['mean'] > 0.6
            not_eligible_arms = CD_df[constrain]['arm_name'].append(XCP_df[constrain2]['arm_name']).append(XCP_df[constrain3]['arm_name'])

            constrained_objective_df = objective_df[~objective_df['arm_name'].isin(not_eligible_arms)].reset_index(drop=True)
            idxmin = constrained_objective_df['mean'].idxmax()
            arm_name, optimum_val = constrained_objective_df.iloc[idxmin,0], constrained_objective_df.iloc[idxmin,2]
            optimum_param = datcom_exp.arms_by_name[arm_name].parameters

            CDbest = CD_df[CD_df['arm_name']==arm_name]
            XCPbest = XCP_df[XCP_df['arm_name']==arm_name]

            new_df.to_csv(f'results_constrained_multifidelity_{NUMBER_OF_INIT_POINTS}_{seed}.csv')


            print('Parameters: \n')
            pprint.pprint(optimum_param)
            print(f'Best CL/CD: {optimum_val}')
            print(f'CD: {CDbest["mean"].array[0]}')
            print(f'XCP: {XCPbest["mean"].array[0]}')

            if optimum_val > best_of_best_cl_cd:
                best_of_best_cl_cd = optimum_val
                best_of_best_iter = CDbest["trial_index"].array[0]
                best_of_best_NUMBER_OF_INIT_POINTS = NUMBER_OF_INIT_POINTS
                best_of_best_seed = seed

            # %%

print(f'best_of_best_cl_cd: {best_of_best_cl_cd}')
print(f'best_of_best_iter: {best_of_best_iter}')
print(f'best_of_best_NUMBER_OF_INIT_POINTS: {best_of_best_NUMBER_OF_INIT_POINTS}')
print(f'best_of_best_seed: {best_of_best_seed}')
