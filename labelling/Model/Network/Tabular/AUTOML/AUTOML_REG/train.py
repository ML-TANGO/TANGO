#-*- coding: utf-8 -*-
import json
import os
import pandas as pd
import numpy as np

import sys
import signal
import traceback
import tensorflow as tf

# Model 폴더 찾기 위한 방법
pathA = os.path.join(os.path.dirname(__file__), os.path.pardir, "../../../")
pathB = os.path.join(os.path.join(pathA))
basePath = os.path.abspath(pathB)
# Model Path 등록

sys.path.append(basePath)

from Common.Logger.Logger import logger
from Common.Utils.Utils import getConfig, getDictData
from Common.Process.Process import prcErrorData, prcSendData, prcGetArgs, prcLogData
from Common.Model.GridModel.GridSearch import GridSearch
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

from DatasetLib import DatasetLib
from Output.Output import sendMsg
from Network.Tabular.AUTOML.AUTOML_REG import model

log = logger("log")

def rmse(y_test, y_pred):
    return np.sqrt(mse(y_test, y_pred))
# 실제 코드 입니다.
if __name__ == '__main__':
    try:
        # data = '{"INPUT_DATA":{"TRAIN_PATH":[{"FILE_PATH":"/Users/gimminjong/Documents/data/iris.csv"}],"DB_INFO":null,"DELIMITER":",","SPLIT_YN":"Y","DATASET_SPLIT":20,"TEST_DATASET_CD":null,"TEST_PATH":null,"LABEL_COLUMN_NAME":"label","MAPING_INFO":[{"DATASET_CD":"RT210112","COLUMN_NM":"sepal_length","COLUMN_ALIAS":null,"DEFAULT_VALUE":null,"checked":1},{"DATASET_CD":"RT210112","COLUMN_NM":"sepal_width","COLUMN_ALIAS":null,"DEFAULT_VALUE":null,"checked":1},{"DATASET_CD":"RT210112","COLUMN_NM":"petal_length","COLUMN_ALIAS":null,"DEFAULT_VALUE":null,"checked":1},{"DATASET_CD":"RT210112","COLUMN_NM":"petal_width","COLUMN_ALIAS":null,"DEFAULT_VALUE":null,"checked":1},{"DATASET_CD":"RT210112","COLUMN_NM":"label","COLUMN_ALIAS":null,"DEFAULT_VALUE":null,"checked":1}]},"SERVER_PARAM":{"AI_CD":"RT210113","SRV_IP":"0.0.0.0","SRV_PORT":5000,"TRAIN_RESULT_URL":"/tab/binary/trainResultLog","TRAIN_STATE_URL":"/tab/binary/binaryStatusLog","AI_PATH":"/Users/gimminjong/Upload/RT210113/","TRAINING_INFO_URL":"/tab/binary/trainInfUrl"},"MODEL_INFO":{"DATA_TYPE":"T","OBJECT_TYPE":"C","MODEL_NAME":"AUTO_CLF","MODEL_TYPE":"ML","MDL_ALIAS":"AUTO_CLF","MDL_IDX":11,"MDL_PATH":"/Users/gimminjong/Documents/bluai_mlkit/Model/Network/Tabular/AUTOML/AUTO_CLF"}}'
        # param = json.loads(data)
        param = json.loads(sys.argv[1])

        #set Param
        dataLib = DatasetLib.DatasetLib()
        param = dataLib.setParams(param)

        saveMdlPath = os.path.join(param["SERVER_PARAM"]["AI_PATH"], str(param["MDL_IDX"]))
        trainStart = dataLib.setStatusOutput(param, "train start", os.getpid(), True)
        _ = sendMsg(trainStart["SRV_ADDR"], trainStart["SEND_DATA"])

        if not os.path.isdir(saveMdlPath):
            os.makedirs(saveMdlPath, exist_ok=True)

        xTrain, xTest, yTrain, yTest, colNames, labelNames, labels, output = dataLib.getNdArray(param)

        param["COLUMNS"] = colNames
        param["LABELS"] = labels
        param["MODEL_PATHLIST"] = [
            "Network/Tabular/XGBOOST/XGB_REG",
            "Network/Tabular/SCIKIT/RF_REG",
            "Network/Tabular/LGBM/LGBM_REG",
            "Network/Tabular/SCIKIT/ET_REG",
            "Network/Tabular/SCIKIT/HIST_REG",
            "Network/Tabular/CATBOOST/CAT_REG",
            "Network/Tabular/SCIKIT/LinearSVR",
            "Network/Tabular/SCIKIT/kNN_REG",
            "Network/Tabular/SCIKIT/SVR",
        ]

        param["MODEL_LIST"] = param["MODEL_PATHLIST"][0:int(param["max_trial"])]
        
        if "SVR" in param["MODEL_LIST"]:
            scaler = StandardScaler()
            scaler.fit(xTrain)
            xTrain = scaler.transform(xTrain)
            xTest = scaler.transform(xTest)

        if output["SUCCESS"]:
            try:
                autoML = model.createModel(param=param, iterations=int(param["epochs"]))

            except Exception as e:
                trainDone = dataLib.setStatusOutput(param, str(e), os.getpid(), False)
                log.error(str(e))
                log.error(traceback.format_exc())
                _ = sendMsg(trainDone["SRV_ADDR"], trainDone["SEND_DATA"])
                sys.exit()

            bestModel, selectPath, newParam = autoML.fitSearch(xTrain, yTrain, xTest, yTest)

            for k, v in newParam.items():
                param[k] = v

            param["SELECT_MDL_PATH"] = selectPath

            if "CAT" in selectPath:
                iterations = int(param["iterations"])
                for iteration in range(1, iterations+1):
                    model = bestModel["ModelModule"]
                    model.fit(xTrain, yTrain)
                    yPred = model.predict(xTest)

                    r2_val = r2(yTest, yPred)
                    mse_val = mse(yTest, yPred)
                    mae_val = mae(yTest, yPred)
                    rmse_val = rmse(yTest, yPred)

                    output = {
                        "SAVE_MDL_PATH": "{}/{}".format(saveMdlPath, "weight.cbm"),
                        "R2": r2_val,
                        "MSE": mse_val,
                        "MAE": mae_val,
                        "RMSE": rmse_val,
                        "ESTIMATOR": iteration
                    }
                    
                    log.debug("[{}] {}".format(param["SERVER_PARAM"]["AI_CD"], output))

                    result = dataLib.setTrainOutput(param, output)
                    _ = sendMsg(result["SRV_ADDR"], result["SEND_DATA"])

            else:  
                # DictValue를 리스트 형태로 변환하기 위함
                newParam_grid = getDictData(newParam)

                gs = GridSearch(
                    estimator=bestModel["ModelModule"],
                    param_grid=newParam_grid,
                    param=param,
                    mode='reg'
                )

                model = gs.fit(xTrain, yTrain)

            score, graph = bestModel["PredictModule"].runPredict(
                model,
                param=param,
                xTest=xTest,
                yTest=yTest,
                colNames=colNames,
                classes=labels,
                flag=1
            )

            output = {
                "SCORE_INFO": {
                    "R2_SCORE": score
                },
                "GRAPH_INFO": graph
            }

            predictData = dataLib.setPredictOutput(param, output)
            _ = sendMsg(predictData["SRV_ADDR"], predictData["SEND_DATA"])
      
            trainDone = dataLib.setStatusOutput(param, "train done", os.getpid(), False)
            _ = sendMsg(trainDone["SRV_ADDR"], trainDone["SEND_DATA"])

            with open(os.path.join(saveMdlPath, "param.json"), "w") as f: 
                json.dump(param, f)

    except Exception as e:
        log.error(e)
        log.error(traceback.format_exc())
        trainDone = dataLib.setStatusOutput(param, str(e), os.getpid(), False)
        _ = sendMsg(trainDone["SRV_ADDR"], trainDone["SEND_DATA"])

    finally:
        sys.exit()