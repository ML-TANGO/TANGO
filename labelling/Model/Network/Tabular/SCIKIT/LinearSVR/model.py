from sklearn.svm import LinearSVR
import numpy as np
from skopt.space import Real, Categorical, Integer

def returnParam(param=None):
    alg = str(param["algorithm"])
    epoch = int(param["epochs"])

    param_grid = {
        "max_iter":Integer(1,epoch+1) if alg == "bayesian" else range(1, epoch+1),
        "C":Integer(1,30) if alg == "bayesian" else range(1,30,2),
        "loss":Categorical(["epsilon_insensitive", "squared_epsilon_insensitive"]) if alg == "bayesian" else ["epsilon_insensitive", "squared_epsilon_insensitive"],
        "dual":[False]
    }

    return param_grid

def createModel(param=None, saveMdlPath=None):
    model = LinearSVR()

    return model