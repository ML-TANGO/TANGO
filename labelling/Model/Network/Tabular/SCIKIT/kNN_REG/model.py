from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from skopt.space import Real, Categorical, Integer

def returnParam(param=None):
    alg = str(param["algorithm"])
    epoch = int(param["epochs"])

    param_grid = {
        "n_neighbors":Integer(1,epoch+1) if alg == "bayesian" else range(1, epoch+1),
        "weights":Categorical(["uniform", "distance"]) if alg == "bayesian" else ["uniform", "distance"],
        "leaf_size":Integer(30,40) if alg == "bayesian" else range(30,40),
        "algorithm":Categorical(["auto", "ball_tree", "kd_tree", "brute"]) if alg == "bayesian" else ["auto", "ball_tree", "kd_tree", "brute"],
        "n_jobs":[-1]
    }

    return param_grid

def createModel(param=None, saveMdlPath=None):
    model = KNeighborsRegressor()

    return model