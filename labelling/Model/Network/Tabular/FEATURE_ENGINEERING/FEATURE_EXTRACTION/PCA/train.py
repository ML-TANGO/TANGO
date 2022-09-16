# etc lib
import os
import sys
import json
import glob
import joblib
import traceback
import importlib
import numpy as np

# Model 폴더 찾기 위한 방법
pathA = os.path.join(os.path.dirname(__file__), os.path.pardir, "../../../../")
pathB = os.path.join(os.path.join(pathA))
basePath = os.path.abspath(pathB)
# Model Path 등록
print(basePath)
sys.path.append(basePath)

from Common.Logger.Logger import logger
from DatasetLib import DatasetLib
from Output.Output import sendMsg

log = logger("log")

from Network.Tabular.FEATURE_ENGINEERING.FEATURE_EXTRACTION.PCA import model

if __name__ == '__main__':
    try:
        param = json.loads(sys.argv[1])

        #set Param
        dataLib = DatasetLib.DatasetLib()
        param = dataLib.setParams(param)
        log.info(param)

        saveMdlPath = os.path.join(param["SERVER_PARAM"]["AI_PATH"], str(param["MDL_IDX"]))
        trainStart = dataLib.setStatusOutput(param, "train start", os.getpid(), True)
        _ = sendMsg(trainStart["SRV_ADDR"], trainStart["SEND_DATA"])
        if not os.path.isdir(saveMdlPath):
            os.makedirs(saveMdlPath, exist_ok=True)

        xTrain, yTrain, xTest, yTest, colNames, labelColName, labels, output = dataLib.getDataFrame(param)

        if output["SUCCESS"]:
            try:
                output = model.getFeature(param, xTrain, xTest, colNames, output)

            except Exception as e:
                trainDone = dataLib.setStatusOutput(param, str(e), os.getpid(), False)
                log.error(str(e))
                log.error(traceback.format_exc())
                _ = sendMsg(trainDone["SRV_ADDR"], trainDone["SEND_DATA"])
                sys.exit()

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
