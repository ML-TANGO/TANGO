import json
import os
import numpy as np
import sys
import traceback
from random import randrange
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from sklearn.metrics import f1_score as f1
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

from Network.Tabular.CATBOOST.CAT_CLF import predict, model


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
                    clf = model.createModel(param=param, iterations=iteration, saveMdlPath=saveMdlPath)

                except Exception as e:
                    trainDone = dataLib.setStatusOutput(param, str(e), os.getpid(), False)
                    log.error(str(e))
                    log.error(traceback.format_exc())
                    _ = sendMsg(trainDone["SRV_ADDR"], trainDone["SEND_DATA"])
                    sys.exit()

                if iteration == 1:
                    fstEpochStartTime = time.time()
                clf.fit(xTrain, yTrain)


                remTime = fstEpochEndTime * (iterations - iteration)
                yPred = clf.predict(xTest)

                precision_val = precision(yTest, yPred, average='macro', zero_division=0)
                recall_val = recall(yTest, yPred, average='macro', zero_division=0)
                f1_val = f1(yTest, yPred, average='macro', zero_division=0)
                accuracy_val = accuracy(yTest, yPred)

                if iteration == 1:
                    fstEpochEndTime = time.time() - fstEpochStartTime

                output = {
                    "SAVE_MDL_PATH": "{}/{}".format(saveMdlPath, "weight.cbm"),
                    "ACCURACY":accuracy_val, 
                    "PRECISION":precision_val, 
                    "RECALL":recall_val, 
                    "F1":f1_val,
                    "ESTIMATOR": iteration,
                    "REMANING_TIME": remTime
                }
                log.debug("[{}] {}".format(param["SERVER_PARAM"]["AI_CD"], output))

                result = dataLib.setTrainOutput(param, output)
                _ = sendMsg(result["SRV_ADDR"], result["SEND_DATA"])

                modeName = str(param["monitor"])
                score = output[modeName.upper()]

                clfMaxModelList = ["accuracy", "precision", "recall", "f1"]
                clfMinModelList = ["logloss"]

                if str(param["mode"]) == "auto":
                    if modeName in clfMaxModelList:
                        if score > best_score:
                            best_score = score                            
                            clf.save_model("{}/{}".format(saveMdlPath, "weight.cbm"))
                            if str(param["early_stopping"]) == "TRUE":
                                count = 0
                        else:
                            if str(param["early_stopping"]) == "TRUE":
                                count += 1
                    else :
                        if score < min_score:
                            min_score = score                  
                            clf.save_model("{}/{}".format(saveMdlPath, "weight.cbm"))
                            if str(param["early_stopping"]) == "TRUE":
                                count = 0
                        else:
                            if str(param["early_stopping"]) == "TRUE":
                                count += 1

                elif str(param["mode"]) == "min":
                    if score < min_score:
                        min_score = score                  
                        clf.save_model("{}/{}".format(saveMdlPath, "weight.cbm"))
                        if str(param["early_stopping"]) == "TRUE":
                            count = 0
                    else:
                        if str(param["early_stopping"]) == "TRUE":
                            count += 1

                elif str(param["mode"]) == "max":
                    if score > best_score:
                        best_score = score
                        clf.save_model("{}/{}".format(saveMdlPath, "weight.cbm"))
                        if str(param["early_stopping"]) == "TRUE":
                            count = 0
                    else:
                        if str(param["early_stopping"]) == "TRUE":
                            count += 1

                if count >= 10:
                    break

            score, graph = predict.runPredict(
                clf,
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

            with open(os.path.join(saveMdlPath, "param.json"), "w") as f: 
                json.dump(param, f)
                
    except Exception as e:
        log.error(e)
        log.error(traceback.format_exc())
        trainDone = dataLib.setStatusOutput(param, str(e), os.getpid(), False)
        _ = sendMsg(trainDone["SRV_ADDR"], trainDone["SEND_DATA"])

    finally:
        sys.exit()