# -*- coding:utf-8 -*-
'''
Worker Flask
predictor model 미리 띄워놓거나, predict 하는 스크립트

1. runPredictModel() : 모델 미리 띄워놓는 스크립트
2. runMiniPredictor() : predictor
3. runAutoLabeling() : autoLabeling

'''
# etc lib
import os
import sys
import time

import tensorflow as tf

# flask lib
from flask import Flask
from flask_restful import Api
from flask import request, make_response
from flask import jsonify

import traceback

# Model 폴더 찾기 위한 방법
basePath = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# Model Path 등록
sys.path.append(basePath)
pid = os.getpid()

from Common.Utils.Utils import getConfig
from Common.Logger.Logger import logger
from Common.Model.Model import predictLoadModel
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

# runPredictModel
@app.route('/runPredictModel', methods=['POST'])
def runPredictModel():
    try:
        req = request.json
        req = req[0]
        objectType = req["OBJECT_TYPE"]
        inputShape = (512, 512, 3)
        aiCd = req["AI_CD"]
        mdlPath = req["MDL_PATH"]
        print(req)

        global model
        model = predictLoadModel(objectType, inputShape, aiCd, mdlPath)
        return make_response(jsonify({"STATE": 1}))

    except Exception as e:
        out = {"STATE": 0, "ERROR": e}
        log.error(out)
        log.error(traceback.format_exc())
        return jsonify(out)

# runMiniPredictor
@app.route('/runMiniPredictor', methods=['POST'])
def runMiniPredictor():
    global model
    req = request.json
    req = req[0]
    st = time.time()
    print("REQ : ", req)
    result = miniPredictor(req, model)
    print("predict time : ", time.time() - st)
    print("result", result)
    return jsonify(result)

# runAutoLabeling
@app.route('/runAutoLabeling', methods=['POST'])
def runAutoLabeling():
    req = request.json
    # req = req[0]
    print("REQ AUTOLABELING : ", req)
    st = time.time()
    result = autoLabeling(req)
    print("predict time : ", time.time() - st)
    # print("result", result)
    return jsonify(result)


def prcSendData(isErr, msg):
    print(f'#_%{pid}&G&{__file__}&G&{isErr}&G&{msg}')


def prcGetArgs():
    data = ''
    while True:
        char = sys.stdin.read(1)
        if char != '':
            data = data + char
        else:
            break
    prcSendData(False, f'{data}, "의 작업을 시작합니다."')
    return data


def modelLoad(args):
    print(args)
    prcSendData(False, args)
    return 1


if __name__ == "__main__":
    try:
        port = sys.argv[1]
        gpuIdx = sys.argv[2]
        modelData = prcGetArgs()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuIdx)
        if gpuIdx != "-1":
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                # 텐서플로가 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
                try:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpus[int(gpuIdx)],
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7266)])

                except RuntimeError as e:
                    out = {"AI_CD": None, "CODE": "120", "MSG": str(e)}
                    print(out)
                    req = sendMsg("http://0.0.0.0:5638/initChecker", out)

        # print("WORKER Port : ", port)
        # print("WORKER GPU IDX : ", gpuIdx)
        # print("finish")
        print("222")
        modelLoad(modelData)
        port = int(port)
        app.run(host='0.0.0.0', port=port)

    except Exception as e:
        # out = {"AI_CD": None, "CODE": "120", "MSG": str(e)}
        # print(out)
        # req = sendMsg("http://0.0.0.0:5638/initChecker", out)
        log.error(traceback.format_exc())
        prcSendData(True, repr(e))
        sys.stderr.flush(repr(e))
    finally:
        prcSendData(False, 'END')
        sys.exit()
