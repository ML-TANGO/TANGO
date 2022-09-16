from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
import numpy as np
from skopt.space import Real, Categorical, Integer

def returnParam(param=None):
    alg = str(param["algorithm"])
    epoch = int(param["epochs"])

    param_grid = {
        "max_iter":Integer(1,epoch+1) if alg == "bayesian" else range(1, epoch+1),
        "learning_rate":Real(0.1, 1) if alg == "bayesian" else np.arange(0.1, 1),
        "max_leaf_nodes":[31],
        "max_depth":Integer(6,10) if alg == "bayesian" else range(6,10),
        "min_samples_leaf":[20],
        "loss":Categorical(["auto"]) if alg == "bayesian" else ["auto"],
        "random_state": [42]
    }

    return param_grid

def createModel(param=None, saveMdlPath=None):
    model = HistGradientBoostingClassifier()

    return model