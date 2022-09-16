import json
import os
import numpy as np
import sys
import traceback
from random import randrange
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import time

# Model 폴더 찾기 위한 방법
pathA = os.path.join(os.path.dirname(__file__), os.path.pardir, "../../../")
pathB = os.path.join(os.path.join(pathA))
basePath = os.path.abspath(pathB)
# Model Path 등록

sys.path.append(basePath)

from Common.Logger.Logger import logger

from DatasetLib import DatasetLib
from Output.Output import sendMsg
from Network.Tabular.CATBOOST.CAT_REG import predict, model



def rmse(y_test, y_pred):
    return np.sqrt(mse(y_test, y_pred))

log = logger("log")
# 실제 코드 입니다.
if __name__ == "__main__":
  
    param = json.loads(sys.argv[1])
    dataLib = DatasetLib.DatasetLib()
    param = dataLib.setParams(param)
    try:
        xTrain, yTrain, xTest, yTest, colNames, labelName, labels, output = dataLib.getDataFrame(param)
        param["COLUMNS"] = colNames
        param["LABELS"] = labels

        iterations = int(param["iterations"]) if "iterations" in param else 100

        saveMdlPath = os.path.join(param["SERVER_PARAM"]["AI_PATH"], str(param["MDL_IDX"]))
        trainStart = dataLib.setStatusOutput(param, "train start", os.getpid(), True)
        _ = sendMsg(trainStart["SRV_ADDR"], trainStart["SEND_DATA"])
        if not os.path.isdir(saveMdlPath):
            os.makedirs(saveMdlPath, exist_ok=True)

        count = 0
        best_score = -99999999999999
        min_score = 99999999999999
        fstEpochStartTime = 0
        fstEpochEndTime = 0

        if output["SUCCESS"]:
            for iteration in range(1, iterations+1):
                try:
                    reg = model.createModel(param=param, iterations=iteration, saveMdlPath=saveMdlPath)

                except Exception as e:
                    trainDone = dataLib.setStatusOutput(param, str(e), os.getpid(), False)
                    log.error(str(e))
                    log.error(traceback.format_exc())
                    _ = sendMsg(trainDone["SRV_ADDR"], trainDone["SEND_DATA"])
                    sys.exit()

                if iteration == 1:
                    fstEpochStartTime = time.time()
                reg.fit(xTrain, yTrain)

                remTime = fstEpochEndTime * (iterations - iteration)
                yPred = reg.predict(xTest)

                r2_val = r2(yTest, yPred)
                mse_val = mse(yTest, yPred)
                mae_val = mae(yTest, yPred)
                rmse_val = rmse(yTest, yPred)

                if iteration == 1:
                    fstEpochEndTime = time.time() - fstEpochStartTime

                output = {
                    "SAVE_MDL_PATH": "{}/{}".format(saveMdlPath, "weight.cbm"),
                    "R2": r2_val,
                    "MSE": mse_val,
                    "MAE": mae_val,
                    "RMSE": rmse_val,
                    "ESTIMATOR": iteration,
                    "REMANING_TIME": remTime
                }
                log.debug("[{}] {}".format(param["SERVER_PARAM"]["AI_CD"], output))

                result = dataLib.setTrainOutput(param, output)
                _ = sendMsg(result["SRV_ADDR"], result["SEND_DATA"])

                modeName = str(param["monitor"])
                score = output[modeName.upper()]

                regMaxModelList = ["r2"]
                regMinModelList = ["mse", "mae", "rmse"]

                if str(param["mode"]) == "auto":
                    if modeName in regMaxModelList:
                        if score > best_score:
                            best_score = score
                            reg.save_model("{}/{}".format(saveMdlPath, "weight.cbm"))
                            if str(param["early_stopping"]) == "TRUE":
                                count = 0
                        else:
                            if str(param["early_stopping"]) == "TRUE":
                                count += 1
                    else :
                        if score < min_score:
                            min_score = score                  
                            reg.save_model("{}/{}".format(saveMdlPath, "weight.cbm"))
                            if str(param["early_stopping"]) == "TRUE":
                                count = 0
                        else:
                            if str(param["early_stopping"]) == "TRUE":
                                count += 1

                elif str(param["mode"]) == "min":
                    if score < min_score:
                        min_score = score                  
                        reg.save_model("{}/{}".format(saveMdlPath, "weight.cbm"))
                        if str(param["early_stopping"]) == "TRUE":
                            count = 0
                    else:
                        if str(param["early_stopping"]) == "TRUE":
                            count += 1

                elif str(param["mode"]) == "max":
                    if score > best_score:
                        best_score = score                          
                        reg.save_model("{}/{}".format(saveMdlPath, "weight.cbm"))
                        if str(param["early_stopping"]) == "TRUE":
                            count = 0
                    else:
                        if str(param["early_stopping"]) == "TRUE":
                            count += 1

                if count >= 10:
                    break

            score, graph = predict.runPredict(
                reg,
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