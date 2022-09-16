import json
import os
import sys
import signal
import traceback
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from random import randrange
import numpy as np

# Model 폴더 찾기 위한 방법
pathA = os.path.join(os.path.dirname(__file__), os.path.pardir, "../../../")
pathB = os.path.join(os.path.join(pathA))
basePath = os.path.abspath(pathB)
# Model Path 등록

sys.path.append(basePath)

from Common.Logger.Logger import logger
from Common.Utils.Utils import getConfig
from Common.Process.Process import prcErrorData, prcSendData, prcGetArgs, prcLogData
from Output.TrainOutput import tabnetState
from DatasetLib import DatasetLib
from Output.Output import sendMsg

from Network.Tabular.TF.TabNetCLF import predict, model

log = logger("log")


# dataset 변형 함수
# input : dataset
# output : x, y
def transform(ds):
    features = tf.unstack(ds['features'])
    label = ds['label']

    x = dict(zip(colNames, features))
    y = tf.one_hot(label, len(labels))
    return x, y


# 테스트 코드 아닙니다.
# 실제 코드 입니다.
if __name__ == "__main__":
    try:
        param = json.loads(sys.argv[1])
        # param = json.loads(data)

        # prcSendData(__file__, json.dumps({"TYPE": "LOG", "DATA": "Trainer Run"}))

        dataLib = DatasetLib.DatasetLib()
        param = dataLib.setParams(param)

        saveMdlPath = os.path.join(param["SERVER_PARAM"]["AI_PATH"], str(param["MDL_IDX"]))
        trainStart = dataLib.setStatusOutput(param, "train start", os.getpid(), True)
        _ = sendMsg(trainStart["SRV_ADDR"], trainStart["SEND_DATA"])

        if not os.path.isdir(saveMdlPath):
            os.makedirs(saveMdlPath, exist_ok=True)

        trainDs, testDs, colNames, labelNames, labels, output = dataLib.getTFDS(param)

        param["COLUMNS"] = colNames
        param["LABELS"] = labels
        with open(os.path.join(saveMdlPath, "param.json"), "w") as f:
            json.dump(param, f)

        trainDs = trainDs.shuffle(150, seed=0)
        testDs = testDs.shuffle(150, seed=0)

        train = trainDs.take(len(trainDs))
        train = train.map(transform)
        train = train.batch(int(param["batch_size"]))

        test = testDs.take(len(testDs))
        test = test.map(transform)
        test = test.batch(len(testDs))

        # Hyper Param 부분은 setParam() 함수 개발 후 변경될 예정입니다. - smpark
        model = model.createModel(param=param, saveMdlPath=saveMdlPath)

        # Keras Hyper Param 부분, weight save(best로 하기로 함), early stopping 추가 예정입니다. - smpark
        initial_lr = float(param["learning_rate"] if "learning_rate" in param else 0.01)
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=100,
            decay_rate=0.9,
            staircase=False
        )

        if param["optimizer"] == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        elif param["optimizer"] == 'adadelta':
            optimizer = tf.keras.optimizers.Adadelta(learning_rate=lr)
        elif param["optimizer"] == 'adagrad':
            optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)
        elif param["optimizer"] == 'adamax':
            optimizer = tf.keras.optimizers.Adamax(learning_rate=lr)
        elif param["optimizer"] == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        elif param["optimizer"] == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
        elif param["optimizer"] == 'nadam':
            optimizer = tf.keras.optimizers.Nadam(learning_rate=initial_lr)

        callbacks = []
        batchState = tabnetState(param, saveMdlPath)
        callbacks.append(batchState)
        if str(param["early_stopping"]) == "TRUE":
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor=str(param['monitor']), patience=10, mode=str(param['mode'])) 
            callbacks.append(early_stopping)

        modelCheckpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(saveMdlPath, "weight"),
            monitor="val_loss",
            mode="auto",
            save_base_only=True,
            save_weights_only=True
        )

        callbacks.append(modelCheckpoint)

        model.compile(
            optimizer=optimizer,
            loss=str(param["loss"]) if "loss" in param else "categorical_crossentropy",
            metrics=['accuracy']
        )
        model.fit(
            train,
            epochs=int(param["epochs"]) if "epochs" in param else 100,
            validation_data=test,
            batch_size=int(param["batch_size"]) if "batch_size" in param else 32,
            verbose=0,
            callbacks=callbacks
        )

        # 그래프 표시용 Predict
        score, graph = predict.runPredict(
            model,
            param=param,
            testDs=test,
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
