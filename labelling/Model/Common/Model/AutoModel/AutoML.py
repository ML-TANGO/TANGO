#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import importlib
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from skopt import BayesSearchCV

from Common.Logger.Logger import logger
log = logger("log")

class AutoML:
    def __init__(self,
                  param,
                  iterations=None,
                  n_jobs=-1):

        self.param = param
        self.iterations = iterations
        self.n_jobs = n_jobs

    def searchML(self):
        # Test Sample
        # param["MODEL_PATHLIST"]: [
        #     "Network/Tabular/XGBOOST/XGB_REG", 
        #     "Network/Tabular/SCIKIT/RF_REG",
        #     "Network/Tabular/LGBM/LGBM_REG",
        #     "Network/Tabular/SCIKIT/ET_REG",
        #     "Network/Tabular/SCIKIT/HIST_REG",
        #     "Network/Tabular/SCIKIT/SVR",
        #     "Network/Tabular/SCIKIT/LinearSVR",
        #     "Network/Tabular/SCIKIT/kNN_REG"
        # ]
        modelPathList = self.param["MODEL_LIST"]

        models = list()
        predicts = list()
        modelParams = list()
        selectPaths = list()

        for modelPath in modelPathList:
            selectPaths.append(modelPath)
            modelPath = modelPath.replace("/", ".")
            modelModule = importlib.import_module(modelPath + ".model")
            predictModule = importlib.import_module(modelPath + ".predict")
            estimator = modelModule.createModel(self.param, self.iterations)
            param_grid = modelModule.returnParam(self.param)

            models.append(estimator)
            predicts.append(predictModule)
            modelParams.append(param_grid)

        return models, predicts, modelParams, selectPaths

    def fitSearch(self, xTrain, yTrain, xTest, yTest):

        models, predicts, modelParams, selectPaths = self.searchML()

        # Max Trial이 모델 갯수보다 커질때
        maxTrial = int(self.param["max_trial"])
        if maxTrial > len(models):
              maxTrial = len(models)

        modelResult = list()
        mode = self.param["algorithm"]
        for i in range(maxTrial):
            if mode == "greedy":
                bestEstimator, bestParam, bestScore = self.greedySearch(
                    estimator=models[i],
                    param_grid=modelParams[i],
                    xTrain=xTrain,
                    yTrain=yTrain,
                    xTest=xTest,
                    yTest=yTest
                )
            elif mode == "random":
                bestEstimator, bestParam, bestScore = self.randomSearch(
                    estimator=models[i],
                    param_grid=modelParams[i],
                    xTrain=xTrain,
                    yTrain=yTrain,
                    xTest=xTest,
                    yTest=yTest
                )
            elif mode == "bayesian":
                bestEstimator, bestParam, bestScore = self.bayesianSearch(
                    estimator=models[i],
                    param_grid=modelParams[i],
                    xTrain=xTrain,
                    yTrain=yTrain,
                    xTest=xTest,
                    yTest=yTest
                )

            modelResult.append({
                "ModelModule": models[i],
                "PredictModule": predicts[i],
                "SelectPath": selectPaths[i],
                "ModelParam": bestParam,
                "ModelScore": bestScore
            })
            log.debug("MODEL RESULT INFO : MODEL : {} SCORE : {} ".format(selectPaths[i], bestScore))
        
        # 전체 모델에서 베스트모델을 찾기 위한 방법
        maxScore = -9999999999999
        bestModel = None
        for i in range(len(modelResult)):
            if maxScore < modelResult[i]["ModelScore"]:
                maxScore = modelResult[i]["ModelScore"]
                bestModel = modelResult[i]
            else:
                continue

        selectPath = bestModel["SelectPath"]
        newParam = bestModel["ModelParam"]

        log.debug("BEST MODEL INFO : MODEL : {} SCORE : {} ".format(bestModel["SelectPath"], bestModel["ModelScore"]))
        return bestModel, selectPath, newParam

    def greedySearch(self, estimator, param_grid, xTrain, yTrain, xTest, yTest):
        model = GridSearchCV(estimator=estimator, param_grid=param_grid)
        model.fit(xTrain, yTrain)

        bestModel = model.best_estimator_
        bestParam = model.best_params_
        bestScore =  bestModel.score(xTest, yTest)

        return bestModel, bestParam, bestScore

    def randomSearch(self, estimator, param_grid, xTrain, yTrain, xTest, yTest):
        model = RandomizedSearchCV(estimator=estimator, param_distributions=param_grid)
        model.fit(xTrain, yTrain)

        bestModel = model.best_estimator_
        bestParam = model.best_params_
        bestScore = bestModel.score(xTest, yTest)

        return bestModel, bestParam, bestScore

    # 추가 테스트후 개발 예정입니다. - mjKim
    def bayesianSearch(self, estimator, param_grid, xTrain, yTrain, xTest, yTest):
        model = BayesSearchCV(estimator=estimator, search_spaces=param_grid)
        model.fit(xTrain, yTrain)

        bestModel = model.best_estimator_
        bestParam = model.best_params_
        bestScore = bestModel.score(xTest, yTest)

        return bestModel, bestParam, bestScore
