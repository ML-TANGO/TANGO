from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
from skopt.space import Real, Categorical, Integer

def returnParam(param=None):
    alg = str(param["algorithm"])
    epoch = int(param["epochs"])

    param_grid = {
        "n_estimators":Integer(1,epoch+1) if alg == "bayesian" else range(1, epoch+1),
        "max_depth":Integer(6,10) if alg == "bayesian" else range(6,10),
        "min_samples_split":Integer(2,10) if alg == "bayesian" else range(2,10),
        "min_samples_leaf":Integer(2,10) if alg == "bayesian" else range(2,10),
        "max_features":Categorical(["auto", "sqrt", "log2"]) if alg == "bayesian" else ["auto", "sqrt", "log2"],
        "max_leaf_nodes":[None],
        "random_state":[42],
        "n_jobs":[-1]
    }

    return param_grid


def createModel(param=None, saveMdlPath=None):
    model = ExtraTreesRegressor()

    return model