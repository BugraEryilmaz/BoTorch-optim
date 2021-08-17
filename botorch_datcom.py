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
