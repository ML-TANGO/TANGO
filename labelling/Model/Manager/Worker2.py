# -*- coding:utf-8 -*-
'''
Worker Flask
predictor model 미리 띄워놓거나, predict 하는 스크립트

1. runPredictModel() : 모델 미리 띄워놓는 스크립트
2. runMiniPredictor() : predictor
3. runAutoLabeling() : autoLabeling

'''

import json
import sys
import os
import random
import time
from numpy.core.arrayprint import printoptions
import tensorflow as tf

import traceback

# flask lib
from flask import Flask
from flask_restful import Api
from flask import request, make_response
from flask import jsonify

# Model 폴더 찾기 위한 방법
basePath = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# Model Path 등록
sys.path.append(basePath)
pid = os.getpid()

from Common.Utils.Utils import getConfig
from Common.Logger.Logger import logger
from Common.Model.PredictModel import predictLoadModel
from Common.Process.Process import prcSendData, prcGetArgs, prcErrorData, prcClose, prcLogData
from Predict.MiniPredictor import miniPredictor
from Predict.AutoLabeling import autoLabeling
from Output.Output import sendMsg

log = logger("log")
srvIp, srvPort, logPath, tempPath, datasetPath, aiPath = getConfig()

server = 'http://{}:{}/api/'.format(srvIp, srvPort)
headers = {"Content-Type": "application/json; charset=utf-8"}

app = Flask(__name__)
api = Api(app)

model = None
modelName = None
inputShape = ()


def modelLoad(args):
    args = args
    OBJECT_TYPE = args["OBJECT_TYPE"]
    AI_CD = args["AI_CD"]
    MDL_PATH = args["MDL_PATH"]
    log.debug(args)

    global model
    global modelName
    global inputShape

    model, modelName, inputShape = predictLoadModel(OBJECT_TYPE, AI_CD, MDL_PATH)

    # print(model.summary())
    prcSendData(__file__, json.dumps({"TYPE": "LOG", "DATA": "Model Run", "AI_CD": AI_CD}))
    log.info("Model Run [{}]".format(MDL_PATH))
    return 1


# FlaskCheck URL
@app.route('/chekFlask', methods=['POST'])
def chekFlask():
    return jsonify({"STATUS": 1})

# runMiniPredictor
@app.route('/runMiniPredictor', methods=['POST'])
def runMiniPredictor():
    try:
        global model
        global modelName
        global inputShape

        req = request.json
        st = time.time()
        prcLogData(req)
        log.info("MiniPredice Start")
        result = miniPredictor(req, model, modelName, inputShape)
        log.info("MiniPredice Done")
        print("predict time : ", time.time() - st)
        print("result", result)
        return jsonify(result)
    except Exception as e:
        log.error(traceback.format_exc())
        prcErrorData(__file__, repr(e))


# runAutoLabeling
@app.route('/runAutoLabeling', methods=['POST'])
def runAutoLabeling():
    try:
        global model
        global modelName
        global inputShape
        req = request.json
        # req = req[0]
        print("REQ AUTOLABELING : ", req)
        st = time.time()
        result = autoLabeling(req, model, modelName, inputShape)
        # dmshin
        print("predict time : ", time.time() - st)
        print("result", result)
        return jsonify(result)

    except Exception as e:
        log.error(traceback.format_exc())
        prcErrorData(__file__, repr(e))


if __name__ == "__main__":
    modelData = None
    try:
        port = sys.argv[1]
        gpuIdx = sys.argv[2]
        # modelData = sys.argv[3])
        # print(modelData)
        modelData = json.loads(prcGetArgs(pid))
        # str = '{"IMAGES":[{"IMAGE_PATH":"/Users/upload/DataSets/SI200038/PennPed00093.png","DATASET_CD":"SI200038","DATA_CD":"00000000","COLOR":null,"RECT":null},{"IMAGE_PATH":"/Users/upload/DataSets/SI200038/PennPed000932.png","DATASET_CD":"SI200038","DATA_CD":"00000001","COLOR":null,"RECT":null}],"MODE":2,"URL":"prePredict","OBJECT_TYPE":"S","AI_CD":"deeplab","CLASS_CD":"BASE","MDL_PATH":"/Users/upload/models/segmentation/deeplab/","DATA_TYPE":"I","EPOCH":null}'
        # modelData = json.loads(str)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuIdx)
        if gpuIdx != "-1":
            gpus = tf.config.experimental.list_physical_devices('GPU')
            log.info("Check gpu. {}".format(gpus))
            log.info("Selected gpu index={}".format(gpuIdx))
            if gpus:
                # 텐서플로가 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
                try:
                    # tf.config.experimental.set_virtual_device_configuration(
                    #     gpus[0],
                    #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7266)])
                    tf.config.experimental.set_memory_growth(gpus[0], True)
                except RuntimeError as e:
                    out = {"AI_CD": None, "CODE": "120", "MSG": str(e)}
                    print(out)
                    log.error(traceback.format_exc())
                    req = sendMsg("http://0.0.0.0:5638/initChecker", out)
        modelLoad(modelData)
        port = int(port)
        app.run(host='0.0.0.0', port=port)

    except Exception as e:
        print(e)
        modelData["ERR"] = repr(e)
        prcErrorData(__file__,  json.dumps(modelData))
        log.error(traceback.format_exc())
        sys.stderr.flush(repr(e))
        pass

    prcClose(__file__)
