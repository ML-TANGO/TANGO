import json
import os
import numpy as np
import sys
import traceback
import tensorflow as tf

import shutil

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
from Output.TrainOutput import tabnetState
from Network.Tabular.AUTOKERAS.AUTOKERAS_REG import predict, model

log = logger("log")

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

        callbacks = []
        batchState = tabnetState(param, saveMdlPath)
        callbacks.append(batchState)
        if str(param["early_stopping"]) == "TRUE":
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor=str(param['monitor']), patience=10, mode=str(param['mode'])) 
            callbacks.append(early_stopping)

        # modelCheckpoint = tf.keras.callbacks.ModelCheckpoint(
        #     filepath=os.path.join(saveMdlPath, "weight"),
        #     monitor="val_loss",
        #     mode="auto",
        #     save_base_only=True,
        #     # save_weights_only=True
        # )

        # callbacks.append(modelCheckpoint)

        # yTrainNew = label_binarize(yTrain, classes=range(0, len(labels)))
        # yTestNew = label_binarize(yTest, classes=range(0, len(labels)))

        if output["SUCCESS"]:
            log.info("START AUTOML")
            # Get AUTOML MODEL
            try:
                automlModel = model.createModel(param=param, saveMdlPath=saveMdlPath)

            except Exception as e:
                trainDone = dataLib.setStatusOutput(param, str(e), os.getpid(), False)
                log.error(str(e))
                log.error(traceback.format_exc())
                _ = sendMsg(trainDone["SRV_ADDR"], trainDone["SEND_DATA"])
                sys.exit()

            automlModel.fit(
                x=xTrain,
                y=yTrain,
                epochs=int(param["epochs"]) if "epochs" in param else 100,
                validation_data=(xTest, yTest),
                batch_size=int(param["batch_size"]) if "batch_size" in param else 32,
                verbose=0
                # callbacks=callbacks
            )

            # AUTOML MODEL ReTrain for result send Server
            model = automlModel.export_model()
            log.info("START ML")

            xTrain = xTrain.astype(np.unicode)
            xTest = xTest.astype(np.unicode)

            model.fit(
                x=xTrain,
                y=yTrain,
                epochs=int(param["epochs"]) if "epochs" in param else 100,
                validation_data=(xTest, yTest),
                batch_size=int(param["batch_size"]) if "batch_size" in param else 32,
                verbose=0,
                callbacks=callbacks
            )
            model.save(os.path.join(saveMdlPath, "weight"), save_format="tf")
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

            shutil.rmtree(os.path.join(saveMdlPath, "tempModel"))

    except Exception as e:
        log.error(e)
        log.error(traceback.format_exc())
        trainDone = dataLib.setStatusOutput(param, str(e), os.getpid(), False)
        _ = sendMsg(trainDone["SRV_ADDR"], trainDone["SEND_DATA"])

    finally:
        sys.exit()