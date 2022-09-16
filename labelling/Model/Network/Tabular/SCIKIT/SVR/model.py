from sklearn.svm import SVR
import numpy as np
from skopt.space import Real, Categorical, Integer

def returnParam(param=None):
    alg = str(param["algorithm"])
    epoch = int(param["epochs"])

    param_grid = {
        "max_iter":Integer(1,epoch+1) if alg == "bayesian" else range(1, epoch+1),
        "C":Integer(1,30) if alg == "bayesian" else range(1,30,2),
        "gamma":Categorical(["scale", "auto"]) if alg == "bayesian" else ["scale", "auto"],
        "kernel":Categorical(["linear", "poly", "rbf"]) if alg == "bayesian" else ["linear", "poly", "rbf"],
        "degree":Integer(1,3) if alg == "bayesian" else range(1,3),
    }

    return param_grid

def createModel(param=None, saveMdlPath=None):
    model = SVR()

    return model