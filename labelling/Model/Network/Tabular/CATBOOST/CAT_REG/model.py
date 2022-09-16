from catboost import CatBoostRegressor
import os
from Common.Logger.Logger import logger
import json
import torch
import numpy as np
log = logger("log")
from skopt.space import Real, Categorical, Integer

def returnParam(param=None):
    alg = str(param["algorithm"])
    epoch = int(param["epochs"])

    param_grid = {
        "iterations":Integer(1,epoch+1) if alg == "bayesian" else range(1, epoch+1),
        "grow_policy":["Depthwise"],
        "eval_metric":["RMSE"],
        "learning_rate":Real(0.1, 1) if alg == "bayesian" else np.arange(0.1, 1),
        "min_data_in_leaf":Integer(2,10) if alg == "bayesian" else range(2,10),
        "depth":Integer(6,10) if alg == "bayesian" else range(6,10),
        "bagging_temperature":Real(0.5,1) if alg == "bayesian" else [0.5, 1],
        "l2_leaf_reg":Integer(1,4) if alg == "bayesian" else range(1,4),
        "random_seed":[42],
        "metric_period":[1],
        "verbose":[False],
        "allow_writing_files":[False]
    }

    return param_grid

def createModel(param=None, iterations=None, saveMdlPath=None):
    if param == None:
        model = CatBoostRegressor(
            iterations=iterations,
            grow_policy="Depthwise",
            eval_metric="RMSE",
            metric_period=1,
            verbose=False,
            allow_writing_files=False
        )
    else:
        model = CatBoostRegressor(
            iterations=iterations,
            grow_policy="Depthwise",
            eval_metric="RMSE",
            metric_period=1,
            learning_rate=float(param["learning_rate"]) if "learning_rate" in param else 0.5,
            min_data_in_leaf=int(param["min_data_in_leaf"]) if "min_data_in_leaf" in param else 2,
            depth=int(param["depth"]) if "depth" in param else 6,
            bagging_temperature=int(param["bagging_temperature"]) if "bagging_temperature" in param else 1,
            l2_leaf_reg=float(param["l2_leaf_reg"]) if "l2_leaf_reg" in param else 3,
            random_seed=42,
            verbose=False,
            allow_writing_files=False
        )

    return model