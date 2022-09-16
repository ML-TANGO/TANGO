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
from Common.Utils.Utils import getConfig
from Common.Process.Process import prcErrorData, prcSendData, prcGetArgs, prcLogData
from Common.Model.GridModel.GridSearch import GridSearch

from DatasetLib import DatasetLib
from Output.Output import sendMsg
from Network.Tabular.SCIKIT.ET_CLF import predict, model

log = logger("log")

# 실제 코드 입니다.
if __name__ == '__main__':
    try:
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
        with open(os.path.join(saveMdlPath, "param.json"), "w") as f:
            json.dump(param, f)

        if output["SUCCESS"]:
            try:
                clf = model.createModel()

            except Exception as e:
                trainDone = dataLib.setStatusOutput(param, str(e), os.getpid(), False)
                log.error(str(e))
                log.error(traceback.format_exc())
                _ = sendMsg(trainDone["SRV_ADDR"], trainDone["SEND_DATA"])
                sys.exit()

            gs = GridSearch(
                estimator=clf,
                param_grid={
                    "n_estimators":range(1, (int(param["n_estimators"]) if "n_estimators" in param else 100)+1, 1),
                    "max_depth":[int(param["max_depth"])] if "max_depth" in param and param["max_depth"] != "None" else [None],
                    "min_samples_split":[int(param["min_samples_split"])] if "min_samples_split" in param else [2],
                    "min_samples_leaf":[int(param["min_samples_leaf"])] if "min_samples_leaf" in param else [1],
                    "max_features":[str(param["max_features"])] if "max_features" in param else ["auto"],
                    "max_leaf_nodes":[int(param["max_leaf_nodes"])] if "max_leaf_nodes" in param and param["max_leaf_nodes"] != "None" else [None],
                    "random_state":[42],
                    "n_jobs":[-1] 
                },
                param=param,
                mode='clf'
            )
            
            model = gs.fit(xTrain, yTrain)

            score, graph = predict.runPredict(
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
                    "AI_ACC": score
                },
                "GRAPH_INFO": graph
            }

            predictData = dataLib.setPredictOutput(param, output)
            _ = sendMsg(predictData["SRV_ADDR"], predictData["SEND_DATA"])

            trainDone = dataLib.setStatusOutput(param, "train done", os.getpid(), False)
            _ = sendMsg(trainDone["SRV_ADDR"], trainDone["SEND_DATA"])

    except Exception as e:
        log.error(e)
        log.error(traceback.format_exc())
        trainDone = dataLib.setStatusOutput(param, str(e), os.getpid(), False)
        _ = sendMsg(trainDone["SRV_ADDR"], trainDone["SEND_DATA"])

    finally:
        sys.exit()