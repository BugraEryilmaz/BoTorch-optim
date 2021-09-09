# %% Hyperparameters
NUMBER_OF_INIT_POINTS = 20
NUMBER_OF_ITERATIONS = 100
# %% Variables
import numpy as np
import torch
# %% DATCOM env
from ax.models.torch.botorch_defaults import get_NEI
from torch._C import NoneType
from my_datcom_env import myDatcomEnv


datcom = myDatcomEnv(transfer=True)
datcom.reset()

print(datcom.base_cd)

ROUND_FACTOR = 4

import torch
#device = torch.device('cpu')
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
MEANFN = "constant"
for MATERNKERN in [True, False]:
    for EI in [True, False]:
        pred_cl_cd = []
        pred_cd = []
        pred_xcp = []
        best_point = torch.tensor(0)
        # %% Constraint function
        def const(cd, xcp):
            return True if cd < datcom.base_cd and xcp < 0.6 and xcp > 0.53 else False
        # %% Model definition
        from botorch.models.gpytorch import GPyTorchModel, ModelListGPyTorchModel
        from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
        from gpytorch.means import ConstantMean, MultitaskMean, ZeroMean
        from gpytorch.models import ExactGP, IndependentModelList
        from gpytorch.kernels import RBFKernel, ScaleKernel, MultitaskKernel, MaternKernel
        from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood, LikelihoodList
        from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
        from gpytorch.priors import GammaPrior

        class SimpleCustomGP(ExactGP, GPyTorchModel):

            _num_outputs = 1  # to inform GPyTorchModel API
            
            def __init__(self, train_X, train_Y):
                # squeeze output dim before passing train_Y to ExactGP
                super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
                if MEANFN == "zero":
                    self.mean_module = ZeroMean()
                else:
                    self.mean_module = ConstantMean()
                if MATERNKERN:
                    self.covar_module = ScaleKernel(
                        base_kernel=MaternKernel(ard_num_dims=train_X.shape[-1]),
                    )
                else:
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

        # %% Read model location

        import argparse

        parser = argparse.ArgumentParser(description='Parse the DATCOM output file')
        parser.add_argument('model_loc',nargs='?',help='''The location of the model to be used. Default: "model_all_uniform.pt"''',default='model_all_uniform.pt')         

        args = parser.parse_args()
        #get model location: args.model_loc
        #get network architecture: [int(i) for i in args.model_loc.split('_')[2][1:-1].split(', ')]

        # %% Model definition
        from torch.nn import Module, ModuleList, Linear, ReLU, MSELoss, L1Loss
        from torch.nn.init import xavier_uniform_, calculate_gain
        import torch.nn.functional as F

        layer_conf = [100, 50, 20, 10, 10]

        class MLP(Module):
            # define model elements
            def __init__(self, n_inputs=8, n_outputs=3):
                super(MLP, self).__init__()
                prev_neuron_num = n_inputs
                self.linears = ModuleList([])
                for neuron_num in layer_conf:
                    new_layer = Linear(prev_neuron_num, neuron_num)
                    xavier_uniform_(new_layer.weight,calculate_gain('relu'))
                    self.linears.append(new_layer)
                    prev_neuron_num = neuron_num
                self.out_layer = Linear(prev_neuron_num, n_outputs)
                xavier_uniform_(self.out_layer.weight)

            # forward propagate input
            def forward(self, X):
                for layer in self.linears:
                    X = layer(X)
                    X = F.relu(X)
                X = self.out_layer(X)
                return X

        # %% Import dnn
        n_inputs, n_outputs = 8, 3

        dnn = MLP(n_inputs, n_outputs)
        dnn.load_state_dict(torch.load(args.model_loc))
        dnn = dnn.to(device)
        dnn.eval()

        # %% DATCOM evaluate    
        # parameters = XLE1, XLE2, CHORD1_1, CHORD1_2, CHORD2_1, CHORD2_2, SSPAN1_2, SSPAN2_2
        import numpy as np
        import random


        def datcom_eval(parameterization, *args):
            cl, cd, xcp, cl_cd = datcom.step(parameterization)
            XLE1, XLE2, CHORD1_1, CHORD1_2, CHORD2_1, CHORD2_2, SSPAN1_2, SSPAN2_2 = \
                parameterization["XLE1"], parameterization["XLE2"], parameterization["CHORD1_1"], parameterization["CHORD1_2"],\
                parameterization["CHORD2_1"], parameterization["CHORD2_2"], parameterization["SSPAN1_2"], parameterization["SSPAN2_2"]
            
            inps = torch.tensor([XLE1, XLE2, CHORD1_1, CHORD1_2, CHORD2_1, CHORD2_2, SSPAN1_2, SSPAN2_2]).to(device)
            preds = dnn.forward(inps)

            ret_cl_cd, ret_cd, ret_xcp = cl_cd-preds[0], cd-preds[1], xcp-preds[2]

            pred_cl_cd.append(float(preds[0].cpu().detach().numpy()))
            pred_cd.append(float(preds[1].cpu().detach().numpy()))
            pred_xcp.append(float(preds[2].cpu().detach().numpy()))

            global best_point
            if const(cd, xcp) and cl_cd > best_point:
                print("VALID!")
                if EI:
                    best_point = torch.tensor(cl_cd)
            
            return {"objective": (ret_cl_cd, 0.0), "CD": (ret_cd, 0.0), "XCP": (ret_xcp, 0.0)}
        # %% Definiton of acqf function
        from botorch.acquisition.objective import IdentityMCObjective, MCAcquisitionObjective
        from botorch.acquisition.acquisition import AcquisitionFunction
        from botorch.models.model import Model
        from typing import Any, Callable, Dict, List, Optional, Tuple
        from typing import Any, Optional, Union
        from torch import Tensor
        from botorch.sampling.samplers import MCSampler, SobolQMCNormalSampler
        from typing import Callable, Dict, List, Optional
        from botorch.acquisition.objective import ConstrainedMCObjective, GenericMCObjective
        from botorch.acquisition.monte_carlo import MCAcquisitionFunction
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

        class myqExpectedImprovement(MCAcquisitionFunction):
            r"""MC-based batch Expected Improvement.

            This computes qEI by
            (1) sampling the joint posterior over q points
            (2) evaluating the improvement over the current best for each sample
            (3) maximizing over q
            (4) averaging over the samples

            `qEI(X) = E(max(max Y - best_f, 0)), Y ~ f(X), where X = (x_1,...,x_q)`

            Example:
                >>> model = SingleTaskGP(train_X, train_Y)
                >>> best_f = train_Y.max()[0]
                >>> sampler = SobolQMCNormalSampler(1024)
                >>> qEI = qExpectedImprovement(model, best_f, sampler)
                >>> qei = qEI(test_X)
            """

            def __init__(
                self,
                model: Model,
                best_f: Union[float, Tensor],
                sampler: Optional[MCSampler] = None,
                objective: Optional[MCAcquisitionObjective] = None,
                X_pending: Optional[Tensor] = None,
                **kwargs: Any,
            ) -> None:
                r"""q-Expected Improvement.

                Args:
                    model: A fitted model.
                    best_f: The best objective value observed so far (assumed noiseless). Can be
                        a `batch_shape`-shaped tensor, which in case of a batched model
                        specifies potentially different values for each element of the batch.
                    sampler: The sampler used to draw base samples. Defaults to
                        `SobolQMCNormalSampler(num_samples=512, collapse_batch_dims=True)`
                    objective: The MCAcquisitionObjective under which the samples are evaluated.
                        Defaults to `IdentityMCObjective()`.
                    X_pending:  A `m x d`-dim Tensor of `m` design points that have been
                        submitted for function evaluation but have not yet been evaluated.
                        Concatenated into X upon forward call. Copied and set to have no
                        gradient.
                """
                super().__init__(
                    model=model, sampler=sampler, objective=objective, X_pending=X_pending
                )
                self.register_buffer("best_f", torch.as_tensor(best_f, dtype=float))

            @concatenate_pending_points
            @t_batch_mode_transform()
            def forward(self, X: Tensor) -> Tensor:
                r"""Evaluate qExpectedImprovement on the candidate set `X`.

                Args:
                    X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                        points each.

                Returns:
                    A `batch_shape'`-dim Tensor of Expected Improvement values at the given
                    design points `X`, where `batch_shape'` is the broadcasted batch shape of
                    model and input `X`.
                """
                posterior = self.model.posterior(X)
                samples = self.sampler(posterior)
                # add dnn results back to the samples
                # X in shape 20,1,8
                # samples in shape 512,20,1,3 cd,xcp,cl_cd
                preds = dnn.forward(X.float())
                # preds in shape 20,1,3
                samples[:,:,:,0] += preds[:,:,1]
                samples[:,:,:,1] += preds[:,:,2]
                samples[:,:,:,2] += preds[:,:,0]
                obj = self.objective(samples, X=X)
                # obj in shape 512,20,1
                obj = (obj - self.best_f.unsqueeze(-1).to(obj)).clamp_min(0)
                q_ei = obj.max(dim=-1)[0].mean(dim=0)
                return q_ei


        def my_get_acquisition_function(
            model: Model,
            objective: MCAcquisitionObjective,
            X_observed: Tensor,
            X_pending: Optional[Tensor] = None,
            constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
            mc_samples: int = 500,
            qmc: bool = True,
            seed: Optional[int] = None,
            **kwargs,
        ) -> MCAcquisitionFunction:
            r"""Convenience function for initializing botorch acquisition functions.

            Args:
                acquisition_function_name: Name of the acquisition function.
                model: A fitted model.
                objective: A MCAcquisitionObjective.
                X_observed: A `m1 x d`-dim Tensor of `m1` design points that have
                    already been observed.
                X_pending: A `m2 x d`-dim Tensor of `m2` design points whose evaluation
                    is pending.
                constraints: A list of callables, each mapping a Tensor of dimension
                    `sample_shape x batch-shape x q x m` to a Tensor of dimension
                    `sample_shape x batch-shape x q`, where negative values imply
                    feasibility. Used when constraint_transforms are not passed
                    as part of the objective.
                mc_samples: The number of samples to use for (q)MC evaluation of the
                    acquisition function.
                qmc: If True, use quasi-Monte-Carlo sampling (instead of iid).
                seed: If provided, perform deterministic optimization (i.e. the
                    function to optimize is fixed and not stochastic).

            Returns:
                The requested acquisition function.

            Example:
                >>> model = SingleTaskGP(train_X, train_Y)
                >>> obj = LinearMCObjective(weights=torch.tensor([1.0, 2.0]))
                >>> acqf = get_acquisition_function("qEI", model, obj, train_X)
            """
            # initialize the sampler
            sampler = SobolQMCNormalSampler(num_samples=mc_samples, seed=seed)
            # instantiate and return the requested acquisition function
            best_f = best_point
            return myqExpectedImprovement(
                model=model,
                best_f=best_f,
                sampler=sampler,
                objective=objective,
                X_pending=X_pending,
            )


        def my__get_acqusition_func(
            model: Model,
            objective_weights: Tensor,
            outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
            X_observed: Optional[Tensor] = None,
            X_pending: Optional[Tensor] = None,
            **kwargs: Any,
        ) -> AcquisitionFunction:
            r"""Instantiates a acquisition function.

            Args:
                model: The underlying model which the acqusition function uses
                    to estimate acquisition values of candidates.
                acquisition_function_name: Name of the acquisition function.
                objective_weights: The objective is to maximize a weighted sum of
                    the columns of f(x). These are the weights.
                outcome_constraints: A tuple of (A, b). For k outcome constraints
                    and m outputs at f(x), A is (k x m) and b is (k x 1) such that
                    A f(x) <= b. (Not used by single task models)
                X_observed: A tensor containing points observed for all objective
                    outcomes and outcomes that appear in the outcome constraints (if
                    there are any).
                X_pending: A tensor containing points whose evaluation is pending (i.e.
                    that have been submitted for evaluation) present for all objective
                    outcomes and outcomes that appear in the outcome constraints (if
                    there are any).
                mc_samples: The number of MC samples to use (default: 512).
                qmc: If True, use qMC instead of MC (default: True).
                prune_baseline: If True, prune the baseline points for NEI (default: True).
                chebyshev_scalarization: Use augmented Chebyshev scalarization.

            Returns:
                The instantiated acquisition function.
            """
            if X_observed is None:
                raise ValueError("There are no feasible observed points.")
            # construct Objective module
            # objective_weights = objective_weights[:3]
            # outcome_constraints = outcome_constraints[0][:,:3], outcome_constraints[1][:,:3]
            obj_tf = get_objective_weights_transform(objective_weights)

            def objective(samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
                return obj_tf(samples)

            con_tfs = get_outcome_constraint_transforms(outcome_constraints)
            inf_cost = get_infeasible_cost(X=X_observed, model=model, objective=objective)
            objective = ConstrainedMCObjective(
                objective=objective, constraints=con_tfs or [], infeasible_cost=inf_cost
            )
            return my_get_acquisition_function(
                model=model,
                objective=objective,
                X_observed=X_observed,
                X_pending=X_pending,
                prune_baseline=kwargs.get("prune_baseline", True),
                mc_samples=kwargs.get("mc_samples", 512),
                qmc=kwargs.get("qmc", True),
                # pyre-fixme[6]: Expected `Optional[int]` for 9th param but got
                #  `Union[float, int]`.
                seed=torch.randint(1, 10000, (1,)).item(),
                marginalize_dim=kwargs.get("marginalize_dim"),
            )
            
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

        for seed in range(10):

            pred_cl_cd = []
            pred_cd = []
            pred_xcp = []
            best_point = torch.tensor(0)

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

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
                    acqf_constructor=my__get_acqusition_func,
                    device=device,
                    transforms=[],
                )
                # print(datcom_exp.fetch_data().df)
                batch = datcom_exp.new_trial(generator_run=model.gen(1))

            # %%
            print("Done!")
            # CD<0.453 & 0.53<XCP<0.6 is okey & look for 3.159
            df = datcom_exp.fetch_data().df

            objective_df = df[df['metric_name']=='objective']
            objective_df = objective_df.reset_index(drop=True)
            CD_df = df[df['metric_name']=='CD']
            CD_df = CD_df.reset_index(drop=True)
            XCP_df = df[df['metric_name']=='XCP']
            XCP_df = XCP_df.reset_index(drop=True)

            new_df = objective_df.join(CD_df, lsuffix=' ', rsuffix='   ').join(XCP_df, lsuffix=' ', rsuffix='   ')

            def array_from_parameters(parameters):
                return [parameters["XLE1"], parameters["XLE2"], parameters["CHORD1_1"], parameters["CHORD1_2"],\
                    parameters["CHORD2_1"], parameters["CHORD2_2"], parameters["SSPAN1_2"], parameters["SSPAN2_2"]]

            result = [datcom_exp.arms_by_name[item].parameters for item in new_df['arm_name ']]
            new_df = new_df.join(pd.DataFrame({'parameters': result}))
            new_df = new_df.join(pd.DataFrame({'pred_cl_cd': pred_cl_cd}))
            new_df = new_df.join(pd.DataFrame({'pred_cd': pred_cd}))
            new_df = new_df.join(pd.DataFrame({'pred_xcp': pred_xcp}))

            constrain = CD_df['mean']+new_df['pred_cd'] >= datcom.base_cd
            constrain2 = XCP_df['mean']+new_df['pred_xcp'] < 0.53
            constrain3 = XCP_df['mean']+new_df['pred_xcp'] > 0.6
            not_eligible_arms = CD_df[constrain]['arm_name'].append(XCP_df[constrain2]['arm_name']).append(XCP_df[constrain3]['arm_name'])

            new_df.to_csv(f'results_constrained_transfer_dnn_{"matern" if MATERNKERN else "RBF"}_{seed}_{MEANFN}_{"EI" if EI else "UCB"}.csv')
            constrained_objective_df = objective_df[~objective_df['arm_name'].isin(not_eligible_arms)].reset_index(drop=True)
            constrained_new_df = new_df[~new_df["arm_name "].isin(not_eligible_arms)].reset_index(drop=True)
            cl_cd_series = constrained_objective_df['mean']+constrained_new_df['pred_cl_cd']
            idxmin = cl_cd_series.idxmax()
            arm_name, optimum_val = constrained_objective_df.iloc[idxmin,0], constrained_objective_df.iloc[idxmin,2]
            df_best = constrained_new_df[constrained_new_df['arm_name'] == arm_name]
            optimum_val += df_best['pred_cl_cd'].array[0]
            optimum_param = datcom_exp.arms_by_name[arm_name].parameters

            CDbest = CD_df[CD_df['arm_name']==arm_name]["mean"].array[0]+df_best['pred_cd'].array[0]
            XCPbest = XCP_df[XCP_df['arm_name']==arm_name]["mean"].array[0]+df_best['pred_xcp'].array[0]

            print('Parameters: \n')
            pprint.pprint(optimum_param)
            print(f'Best CL/CD: {optimum_val}')
            print(f'CD: {CDbest}')
            print(f'XCP: {XCPbest}')

        # %%
