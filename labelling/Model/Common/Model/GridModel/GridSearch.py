import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from itertools import product
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from sklearn.metrics import f1_score as f1

import pickle
import os, sys

import time

from Common.Logger.Logger import logger
log = logger("log")

from DatasetLib import DatasetLib
from Output.Output import sendMsg

def _get_param_grid(param_grid):
    items = sorted(param_grid.items())
    
    key, value = zip(*items)
    cartesian = product(*value)
    for v in cartesian:
        params = dict(zip(key, v))
        yield params

from copy import deepcopy
from joblib import Parallel, delayed
import pickle

def rmse(y_test, y_pred):
    return np.sqrt(mse(y_test, y_pred))

def saveModel(best_estimator, savePath):
    pickle.dump(best_estimator, open(savePath, 'wb'))

class GridSearch:
    def __init__(self, estimator, param_grid, param,
                  fit_params=None, verbose=True, n_jobs=-1,
                  pre_dispatch="2*n_jobs", refit=True, mode=None):

        self.refit = refit
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.estimator = estimator
        self.param_grid = param_grid
        self.fit_params = fit_params
        self.pre_dispatch = pre_dispatch
        self.param = param
        self.saveMdlPath = os.path.join(self.param["SERVER_PARAM"]["AI_PATH"], str(self.param["MDL_IDX"]), "weight.pkl")
        self.mode = mode

        if 'n_estimators' in param.keys():
            self.epochs = param["n_estimators"]
        elif 'max_iter' in param.keys():
            self.epochs = param["max_iter"]
        elif 'n_neighbors' in param.keys():
            self.epochs = param["n_neighbors"]
        elif 'iterations' in param.keys():
            self.epochs = param["iterations"]

        self.epochs = int(self.epochs)

        self.fstEpochStartTime = 0
        self.fstEpochEndTime = 0

    def fit(self, X, y):
        params_iterable = list(_get_param_grid(self.param_grid))
        fit_params = self.fit_params if self.fit_params is not None else {}
        count = 0
        best_score = -99999999999999
        min_score = 99999999999999

        regMaxModelList = ["r2"]
        regMinModelList = ["mse", "mae", "rmse"]
        clfMaxModelList = ["accuracy", "precision", "recall", "f1"]
        clfMinModelList = ["logloss"]

        for parameters in params_iterable:
            if self.mode.lower() == 'reg':
                score, modeName, monitorName = self._fit_and_score_reg(
                    self.estimator, X, y, parameters, fit_params
                )

                if modeName == "auto":
                    if monitorName in regMaxModelList:
                        if score > best_score:
                            best_score = score
                            saveModel(self.estimator, self.saveMdlPath)

                            if str(self.param["early_stopping"]) == "TRUE":
                                count = 0
                        else:
                            if str(self.param["early_stopping"]) == "TRUE":
                                count += 1
                    else:
                        if score < min_score:
                            min_score = score
                            saveModel(self.estimator, self.saveMdlPath)
                            if str(self.param["early_stopping"]) == "TRUE":
                                count = 0
                        else:
                            if str(self.param["early_stopping"]) == "TRUE":
                                count += 1

                elif modeName == "min":
                    if score < min_score:
                        min_score = score
                        saveModel(self.estimator, self.saveMdlPath)
                        if str(self.param["early_stopping"]) == "TRUE":
                            count = 0
                    else:
                        if str(self.param["early_stopping"]) == "TRUE":
                            count += 1

                elif modeName == "max":
                    if score > best_score:
                        best_score = score
                        saveModel(self.estimator, self.saveMdlPath)
                        if str(self.param["early_stopping"]) == "TRUE":
                            count = 0
                    else:
                        if str(self.param["early_stopping"]) == "TRUE":
                            count += 1

            elif self.mode.lower() == 'clf':
                score, modeName, monitorName = self._fit_and_score_clf(
                    self.estimator, X, y, parameters, fit_params
                )

                if modeName == "auto":
                    if monitorName in clfMaxModelList:
                        if score > best_score:
                            best_score = score
                            saveModel(self.estimator, self.saveMdlPath)
                            if str(self.param["early_stopping"]) == "TRUE":
                                count = 0
                        else:
                            if str(self.param["early_stopping"]) == "TRUE":
                                count += 1
                    else:
                        if score < min_score:
                            min_score = score
                            saveModel(self.estimator, self.saveMdlPath)
                            if str(self.param["early_stopping"]) == "TRUE":
                                count = 0
                        else:
                            if str(self.param["early_stopping"]) == "TRUE":
                                count += 1

                elif modeName == "min":
                    if score < min_score:
                        min_score = score
                        saveModel(self.estimator, self.saveMdlPath)
                        if str(self.param["early_stopping"]) == "TRUE":
                            count = 0
                    else:
                        if str(self.param["early_stopping"]) == "TRUE":
                            count += 1

                elif modeName == "max":
                    if score > best_score:
                        best_score = score
                        saveModel(self.estimator, self.saveMdlPath)
                        if str(self.param["early_stopping"]) == "TRUE":
                            count = 0
                    else:
                        if str(self.param["early_stopping"]) == "TRUE":
                            count += 1

            if count >= 10:
                break

        with open(self.saveMdlPath, "rb") as f:
            bestModel = pickle.load(f)

        return bestModel

    def _fit_and_score_reg(self, estimator, X, y, parameters, fit_params):
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=42, shuffle=True)

        estimator.set_params(**parameters)

        if 'n_estimators' in parameters.keys():
            iters = estimator.n_estimators
        elif 'max_iter' in parameters.keys():
            iters = estimator.max_iter
        elif 'n_neighbors' in parameters.keys():
            iters = estimator.n_neighbors
        elif 'iterations' in parameters.keys():
            iters = estimator.get_params('iterations')['iterations']

        if iters == 1:
            self.fstEpochStartTime = time.time()
        estimator.fit(X_train, y_train, **fit_params)
        if iters == 1:
            self.fstEpochEndTime = time.time() - self.fstEpochStartTime

        remTime = self.fstEpochEndTime * (self.epochs - iters)
        y_pred_train = estimator.predict(X_train)
        y_pred_test = estimator.predict(X_val)

        r2_val = r2(y_val, y_pred_test)
        mse_val = mse(y_val, y_pred_test)
        mae_val = mae(y_val, y_pred_test)
        rmse_val = rmse(y_val, y_pred_test)

        dataLib = DatasetLib.DatasetLib()
        output = {
            "ESTIMATOR": iters,
            "R2": r2_val,
            "MSE": mse_val,
            "MAE": mae_val,
            "RMSE": rmse_val,
            "SAVE_MDL_PATH": self.saveMdlPath,
            "REMANING_TIME": remTime
        }

        log.debug("[{}] {}".format(self.param["SERVER_PARAM"]["AI_CD"], output))
        result = dataLib.setTrainOutput(self.param, output)
        _ = sendMsg(result["SRV_ADDR"], result["SEND_DATA"])

        if str(self.param["early_stopping"]) == "TRUE":
            modeName = str(self.param["mode"])
            monitorName = str(self.param["monitor"])
            score = output[monitorName.upper()]
        else:
            modeName = "max"
            monitorName = "r2"
            score = output[monitorName.upper()]

        return score, modeName, monitorName

    def _fit_and_score_clf(self, estimator, X, y, parameters, fit_params):

        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=42, shuffle=True)

        estimator.set_params(**parameters)

        if 'n_estimators' in parameters.keys():
            iters = estimator.n_estimators
        elif 'max_iter' in parameters.keys():
            iters = estimator.max_iter
        elif 'n_neighbors' in parameters.keys():
            iters = estimator.n_neighbors
        elif 'iterations' in parameters.keys():
            iters = estimator.get_params('iterations')['iterations']

        if iters == 1:
            self.fstEpochStartTime = time.time()
        estimator.fit(X_train, y_train, **fit_params)

        remTime = self.fstEpochEndTime * (self.epochs - iters)

        y_pred_train = estimator.predict(X_train)
        y_pred_test = estimator.predict(X_val)

        precision_val = precision(y_val, y_pred_test, average='macro', zero_division=0)
        recall_val = recall(y_val, y_pred_test, average='macro', zero_division=0)
        f1_val = f1(y_val, y_pred_test, average='macro', zero_division=0)
        accuracy_val = accuracy(y_val, y_pred_test)

        dataLib = DatasetLib.DatasetLib()
        saveMdlPath = os.path.join(self.param["SERVER_PARAM"]["AI_PATH"], str(self.param["MDL_IDX"]), "weight.pkl")
        
        if iters == 1:
            self.fstEpochEndTime = time.time() - self.fstEpochStartTime

        output = {
            "ESTIMATOR": iters,
            "ACCURACY": accuracy_val,
            "PRECISION": precision_val,
            "RECALL": recall_val,
            "F1": f1_val,
            "SAVE_MDL_PATH": saveMdlPath,
            "REMANING_TIME": remTime
        }

        log.debug("[{}] {}".format(self.param["SERVER_PARAM"]["AI_CD"], output))

        result = dataLib.setTrainOutput(self.param, output)
        _ = sendMsg(result["SRV_ADDR"], result["SEND_DATA"])

        if str(self.param["early_stopping"]) == "TRUE":
            modeName = str(self.param["mode"])
            monitorName = str(self.param["monitor"])
            score = output[monitorName.upper()]
        else:
            modeName = "max"
            monitorName = "accuracy"
            score = output[monitorName.upper()]

        return score, modeName, monitorName
