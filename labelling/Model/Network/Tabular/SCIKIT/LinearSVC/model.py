from sklearn.svm import LinearSVC
import numpy as np
from skopt.space import Real, Categorical, Integer

def returnParam(param=None):
    alg = str(param["algorithm"])
    epoch = int(param["epochs"])

    param_grid = {
        "max_iter":Integer(1,epoch+1) if alg == "bayesian" else range(1, epoch+1),
        "C":Integer(1,30) if alg == "bayesian" else range(1,30,2),
        "loss":Categorical(["hinge", "squared_hinge"]) if alg == "bayesian" else ["hinge", "squared_hinge"],
        "dual":[False]
    }

    return param_grid

def createModel(param=None, saveMdlPath=None):
    model = LinearSVC()

    return model