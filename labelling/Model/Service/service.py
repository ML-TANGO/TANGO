# etc lib
import os
import sys
import json
import glob
import joblib
import traceback
import importlib
import numpy as np
import torch
import tensorflow as tf
import time

# flask lib
from flask import Flask
from flask_restful import Api
from flask import request, make_response
from flask import jsonify

# request lib
import requests

# Model 폴더 찾기 위한 방법
pathA = os.path.join(os.path.dirname(__file__), os.path.pardir)
pathB = os.path.join(os.path.join(pathA))
basePath = os.path.abspath(pathB)
# Model Path 등록

sys.path.append(basePath)

from Common.Logger.Logger import logger
from Output.Output import sendMsg
from Output.PredictOutput import operator, operator2

log = logger("log")

app = Flask(__name__)
api = Api(app)
# input
'''
{
  "PORT" : 12345,
  "TARGET_CLASS": []
  "MODEL_INFO":
  {
    "MODEL_PATH": "abc/abc/abc",
    "WEIGHT_PATH": "AI_CD/MODEL_IDX/",
    "DATA_TYPE": "T",
    "OBJECT_TYPE": "C",
  },
  "SERVER_PARAM":
    {
      "AI_CD": 111111,
      "SRV_IP": "0.0.0.0",
      "SRV_PORT": 10236,
      "PREDICT_RESULT_URL": "/tab/aaaaa"
    }
}
'''

# output
'''
{
  "PID": 11111,
  "STATUS": True / False,
  "MSG": ""
}
'''

model = None
modulePath = None
labels = None
colums = None
predictResultUrl = None
AI_CD = None
headers = {"Content-Type": "application/json; charset=utf-8"}
targetCls = None
dataType = None

hyperParam = None


# loadModel
def modelLoad(param):
    global model
    global modulePath
    global labels
    global columns

    global predictResultUrl
    global serviceHealthUrl
    global AI_CD
    global IS_CD

    global targetCls
    global dataType

    global hyperParam

    try:
        modelLoadStatus = True
        saveMdlPath = param["MODEL_INFO"]["WEIGHT_PATH"]
        hyperParamJson = os.path.join(saveMdlPath, 'param.json')
        with open(hyperParamJson, "r") as jsonFile:
            hyperParam = json.load(jsonFile)

        modulePath = param["MODEL_INFO"]["MODEL_PATH"] if 'AUTOML' not in param["MODEL_INFO"]["MODEL_PATH"] else hyperParam["SELECT_MDL_PATH"]
        log.info(modulePath)

        modulePath = modulePath.replace("/", ".")
        predictResultUrl = "http://{}:{}/api{}".format(
            param["SERVER_PARAM"]["SRV_IP"],
            param["SERVER_PARAM"]["SRV_PORT"],
            param["SERVER_PARAM"]["PREDICT_RESULT_URL"]
        )
        serviceHealthUrl = "http://{}:{}/api{}".format(
            param["SERVER_PARAM"]["SRV_IP"],
            param["SERVER_PARAM"]["SRV_PORT"],
            param["SERVER_PARAM"]["SERVICE_HEALTH_URL"]
        )
        AI_CD = param["SERVER_PARAM"]["AI_CD"]
        IS_CD = param["SERVER_PARAM"]["IS_CD"]
        targetCls = param["TARGET_CLASS"]
        dataType = param["MODEL_INFO"]["DATA_TYPE"]

        modelModule = importlib.import_module(modulePath + '.model')

        labels = hyperParam["LABELS"]
        columns = hyperParam["COLUMNS"]
        model = modelModule.createModel(saveMdlPath=saveMdlPath)

        weightPath = glob.glob(os.path.join(param["MODEL_INFO"]["WEIGHT_PATH"], "weight.*"))
        if len(weightPath) != 0:
            weightPath = weightPath[0]

        # ML Model
        if '.pkl' in weightPath:
            model = joblib.load(weightPath)

        # pytorch Model
        elif '.pt' in weightPath:
            model.load_state_dict(torch.load(weightPath))

        # tensorflow Model
        elif '.index' in weightPath:
            weightPath = weightPath.replace('.index', '')
            model.load_weights(weightPath)

        # catboost Model
        elif '.cbm' in weightPath:
            model.load_model(weightPath)

        # keras model
        elif '.h5' in weightPath or len(weightPath) == 0:
            weightPath = os.path.join(param["MODEL_INFO"]["WEIGHT_PATH"], "weight")
            model = tf.keras.models.load_model(weightPath)

        return model, modelLoadStatus

    except Exception as e:
        log.error(traceback.format_exc())
        log.error(e)
        modelLoadStatus = False
        return str(e), modelLoadStatus


# runPredict
@app.route('/', methods=['POST'])
def runPredict():
    global model
    global modulePath
    global labels
    global columns

    global predictResultUrl
    global AI_CD
    global IS_CD

    global targetCls
    global dataType

    global hyperParam

    req = request.json[0]
    log.debug(req)

    header = req["HEADERS"]
    testData = req["X_TEST"]

    output = []
    predictModule = importlib.import_module(modulePath + '.predict')
    prdStart = time.time()
    for value in testData:
        try:
            xTest = list()
            for i in range(len(header)):
                xTest.append({
                    "HEADER": header[i],
                    "VALUE": value[i],
                })

            log.debug("predict start {}".format(prdStart))
            result = predictModule.runPredict(
                model,
                xTest=xTest,
                classes=labels,
                param=hyperParam,
                colNames=columns,
                flag=0
            )
            log.debug(result)
            prdTime = time.time() - prdStart
            log.debug("predict end {}".format(prdTime))
            
            resultData = []
            log.debug(targetCls)
            for target in targetCls:
                tmp = {}
                if result is not None:
                    if target["CLASS_NAME"] != str(result["label"]):
                        continue

                if target["TARGET"] == 'A':
                    tmp = operator(
                        accuracy=result["ACCURACY"],
                        accScope=target["ACC_SCOPE"],
                        color=None,
                        objectType=dataType,
                        className=result["label"]
                    )

                elif target["TARGET"] == 'V':
                    tmp = operator(
                        accuracy=result["label"],
                        accScope=target["ACC_SCOPE"],
                        color=None,
                        objectType=dataType,
                        className=None
                    )
                if result["MSG"] is not None:
                    tmp = {
                        "IS_CD": IS_CD,
                        "PRC_TIME": prdTime,
                        "MSG": result["MSG"],
                        "TEST": param["TARGET_CLASS"],
                        "DP_LABEL": target["DP_LABEL"],
                        "CLASS_CD": target["CLASS_CD"],
                        "OUT_CD": target["OUT_CD"],
                        "STATUS": 0,
                    }
                else:
                    if tmp is not None:
                        tmp["IS_CD"] = IS_CD
                        tmp["PRC_TIME"] = prdTime
                        tmp["DP_LABEL"] = target["DP_LABEL"]
                        tmp["CLASS_CD"] = target["CLASS_CD"]
                        tmp["OUT_CD"] = target["OUT_CD"]
                        tmp["STATUS"] = 1

                        if target["TARGET"] == 'A':
                            if str(target["CLASS_NAME"]) == str(result["label"]):
                                if operator2(float(result["ACCURACY"]), target["ACC_SCOPE"]):
                                    resultData.append(tmp)
                                else:
                                    tmp["STATUS"] = 2
                                    resultData.append(tmp)
                            else:
                                tmp["STATUS"] = 2
                                resultData.append(tmp)
                        else:
                            resultData.append(tmp)

            output.append(resultData)

        except Exception as e:
            log.error(traceback.format_exc())
            log.error(e)
            output.append([{
                # "MSG": str(e),
                "MSG": "IndexError: list index out of range",
                "STATUS": 0,
                "IS_CD": IS_CD
            }])
            log.error(output)

    # _ = sendMsg(predictResultUrl, json.dumps(result))
    log.debug(output)
    return jsonify(output)


# main
if __name__ == "__main__":

    # data = '{"PORT":12345,"MODEL_INFO":{"MODEL_PATH":"Network/Tabular/XGBOOST/XGB_REG","WEIGHT_PATH":"/Users/upload/AiModel/RT20210030/0","DATA_TYPE":"T","OBJECT_TYPE":"R"},"SERVER_PARAM":{"AI_CD":111111,"SRV_IP":"0.0.0.0","SRV_PORT":10236,"PREDICT_RESULT_URL":"/tab/aaaaa"},"TARGET_CLASS":[{"OUT_CD":129,"IS_CD":106,"CLASS_CD":1,"CLASS_NAME":"setosa","DP_LABEL":"setosa","COLOR":"#95f9d0","ACC_SCOPE":"gt,20,none,,","LOCATION":"S","HW_CD":null,"TARGET":"V"}]}'
    # data = '{"PORT":12345,"MODEL_INFO":{"MODEL_PATH":"Network/Tabular/AUTOKERAS/AUTOKERAS_REG","WEIGHT_PATH":"/Users/upload/AiModel/RT20210075/0","DATA_TYPE":"T","OBJECT_TYPE":"R"},"SERVER_PARAM":{"AI_CD":111111,"SRV_IP":"0.0.0.0","SRV_PORT":10236,"PREDICT_RESULT_URL":"/tab/aaaaa"},"TARGET_CLASS":[{"OUT_CD":129,"IS_CD":106,"CLASS_CD":1,"CLASS_NAME":"normal","DP_LABEL":"tttt","COLOR":"#95f9d0","ACC_SCOPE":"gt,0.01,none,,","LOCATION":"S","HW_CD":null,"TARGET":"V"}]}'
    # data = '{"PORT":12345,"TARGET_CLASS":[{"OUT_CD":174,"IS_CD":162,"CLASS_CD":0,"CLASS_NAME":"Ks","DP_LABEL":"KS1","COLOR":"#f7e7a0","ACC_SCOPE":"gt,0.1,none,,","LOCATION":"S","HW_CD":null,"TARGET":"V"},{"OUT_CD":175,"IS_CD":162,"CLASS_CD":0,"CLASS_NAME":"Ks","DP_LABEL":"KS2","COLOR":"#7be278","ACC_SCOPE":"gt,2,and,lt,200","LOCATION":"S","HW_CD":null,"TARGET":"V"}],"MODEL_INFO":{"MODEL_PATH":"Network/Tabular/XGBOOST/XGB_REG","WEIGHT_PATH":"/Users/upload/InputSources/162/model/0","DATA_TYPE":"T","OBJECT_TYPE":"R"},"SERVER_PARAM":{"AI_CD":"RT20210090","SRV_IP":"0.0.0.0","SRV_PORT":10236,"IS_CD":162,"PREDICT_RESULT_URL":"/tab/aaaaa"}}'

    # data = '{"PORT":12345,"TARGET_CLASS":[{"OUT_CD":174,"IS_CD":162,"CLASS_CD":0,"CLASS_NAME":"label","DP_LABEL":"label1","COLOR":"#f7e7a0","ACC_SCOPE":"gt,0.1,none,,","LOCATION":"S","HW_CD":null,"TARGET":"V"}],"MODEL_INFO":{"MODEL_PATH":"Network/Tabular/AUTOML/AUTOML_REG","WEIGHT_PATH":"/Users/gimminjong/Upload/RT210113/18","DATA_TYPE":"T","OBJECT_TYPE":"R"},"SERVER_PARAM":{"AI_CD":"RT20210090","SRV_IP":"0.0.0.0","SRV_PORT":10236,"IS_CD":162,"PREDICT_RESULT_URL":"/tab/aaaaa"}}'

    param = json.loads(sys.argv[1])

    # param = json.loads(data)
    port = int(param["PORT"])

    model, modelLoadStatus = modelLoad(param)

    if not modelLoadStatus:
        output = {
            "PID": None,
            "PARAM": param,
            "STATUS": False,
            "MSG": model
        }
        sys.exit()

    else:
        output = {
            "PID": os.getpid(),
            "PARAM": param,
            "STATUS": True,
            "MSG": "MODEL LOAD SUCCESS"
        }

    print(json.dumps(output))
    _ = sendMsg(serviceHealthUrl, output)

    try:
        # app.debug = True
        app.run(host='0.0.0.0', port=port)

    except Exception as e:
        log.error(traceback.format_exc())
        log.error(e)
