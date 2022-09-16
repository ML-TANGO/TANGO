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
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import label_binarize


# Model 폴더 찾기 위한 방법
pathA = os.path.join(os.path.dirname(__file__), os.path.pardir)
pathB = os.path.join(os.path.join(pathA))
basePath = os.path.abspath(pathB)
# Model Path 등록

sys.path.append(basePath)

from Common.Logger.Logger import logger
from Output.Output import sendMsg
from Output.GetGraph import graph
from DatasetLib import DatasetLib

log = logger("log")


model = None
modulePath = None
labels = None
columns = None
predictResultUrl = None
hyperParam = None

'''
Evaluation 

input : {"INPUT_DATA":{"START":10,"END":50,"TEST_PATH":["/Users/parksangmin/weda/2020/BluAI/tabular_dataset/clf/iris.csv"],"DB_INFO":{},"LABEL_COLUMN_NAME":"label"},"MODEL_INFO":{"MODEL_PATH":"Network/Tabular/TF/TabNetCLF","WEIGHT_PATH":"/Users/upload/AiModel/CT20210206/0","DATA_TYPE":"T","OBJECT_TYPE":"C"}}
output : Result
'''


# loadModel
def modelLoad(param):
    global model
    global modulePath
    global labels
    global columns

    global predictResultUrl
    global hyperParam

    try:
        modelLoadStatus = True
        saveMdlPath = param["MODEL_INFO"]["WEIGHT_PATH"]
        hyperParamJson = os.path.join(saveMdlPath, 'param.json')
        with open(hyperParamJson, "r") as jsonFile:
            hyperParam = json.load(jsonFile)

        modulePath = param["MODEL_INFO"]["MODEL_PATH"] if 'AUTOML' not in param["MODEL_INFO"]["MODEL_PATH"] else hyperParam["SELECT_MDL_PATH"]

        modulePath = modulePath.replace("/", ".")
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

        return modelLoadStatus

    except Exception as e:
        print(e, file=sys.stderr)
        log.error(traceback.format_exc())
        log.error(e)
        modelLoadStatus = False
        return str(e), modelLoadStatus


# run predict
def runPredict(param):
    global model
    global modulePath
    global labels
    global columns

    dfColumns = columns

    predictModule = importlib.import_module(modulePath + '.predict')
    prdStart = time.time()

    output = {}
    yPred = []

    start = int(param["INPUT_DATA"]["START"])
    end = int(param["INPUT_DATA"]["END"])

    try:
        if param["INPUT_DATA"]["UPLOAD_TYPE"] == "FILE":
            dfColumns.append(param["INPUT_DATA"]["LABEL_COLUMN_NAME"])
            testDf = pd.read_csv(
                param["INPUT_DATA"]["TEST_PATH"][0],
                skiprows=start,
                nrows=end - start + 1,
                header=0,
                encoding="utf-8-sig"
            )

            testDf.columns = dfColumns

            dataDf = testDf.drop(param["INPUT_DATA"]["LABEL_COLUMN_NAME"], axis=1)
            labelDf = testDf[param["INPUT_DATA"]["LABEL_COLUMN_NAME"]]

            predictDf = pd.concat([dataDf, labelDf], axis=1)

            for column in dfColumns:
                if column in hyperParam["ENCODER_DATA"]:
                    predictDf[column].fillna("None", inplace = True)
                else:
                    predictDf[column].fillna(0)

        elif param["INPUT_DATA"]["UPLOAD_TYPE"] == "DB":
            dataLib = DatasetLib.DatasetLib()
            testDf, dbOutput = dataLib.getDbData(param["INPUT_DATA"]["DB_INFO"])

            if dbOutput["STATUS"] == 0:
                return dbOutput
            else:
                testDf = testDf[start:end]
            
            dataDf = testDf.drop(param["INPUT_DATA"]["LABEL_COLUMN_NAME"], axis=1)
            labelDf = testDf[param["INPUT_DATA"]["LABEL_COLUMN_NAME"]]

            predictDf = pd.concat([dataDf, labelDf], axis=1)

            for column in columns:
                if column in hyperParam["ENCODER_DATA"]:
                    predictDf[column].fillna("None", inplace = True)
                else:
                    predictDf[column].fillna(0)

        labelColumnName = param["INPUT_DATA"]["LABEL_COLUMN_NAME"]
        resultData = []

        for tmp in predictDf.values:
            xTest = []
            for j in range(len(columns)):
                if columns[j] == labelColumnName:
                    continue
                xTest.append({
                    "HEADER": columns[j],
                    "VALUE": tmp[j]
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

            prdTime = time.time() - prdStart
            log.debug("predict end {}".format(prdTime))

            if result["MSG"] is not None:
                tmp = {
                    "PRC_TIME": prdTime,
                    "MSG": result["MSG"],
                    "DP_LABEL": None,
                    "STATUS": 0,
                }
                # resultData.append(tmp)
            else:
                tmp = {
                    "PRC_TIME": prdTime,
                    "DP_LABEL": result['label'],
                    "ACCURACY": result['ACCURACY'] if "ACCURACY" in result else None,
                    "STATUS": 1
                }
                yPred.append(result['label'])
                # resultData.append(tmp)

            resultData.append(tmp)
        yTest = predictDf[labelColumnName]

        g = graph(param, labels)

        graphData = []

        if param["MODEL_INFO"]["OBJECT_TYPE"] == "R":

            yTest = np.array(yTest.astype(float))
            yPred = np.array(yPred)

            # REG Plot
            regPlotOutput = g.regPlot(yTest, yPred)
            graphData.append(regPlotOutput)

            # distribution Plot
            distributionPlotOutput = g.distributionPlot(yTest, yPred)
            graphData.append(distributionPlotOutput)

        elif param["MODEL_INFO"]["OBJECT_TYPE"] == "C":
            yTest = np.array(yTest)
            if yTest.dtype == object and type(yTest[0]) == int:
                yTest = np.array(yTest, dtype=np.int64)

            yPred = np.array(yPred)

            yTestBinary = label_binarize(yTest, classes=labels)
            yPredBinary = label_binarize(yPred, classes=labels)

            # ROC Curve
            rocOutput = g.roc(yTestBinary, yPredBinary)
            graphData.append(rocOutput)

            # Precision Recall Curve
            preRecallOutput = g.precisionRecall(yTestBinary, yPredBinary)
            graphData.append(preRecallOutput)

            # Confusion Matrix
            confMat = g.confusionMatrix(yTest, yPred, labels)
            graphData.append(confMat)

        output = {
            "GRAPH": graphData,
            "RESULT": resultData
        }

    except Exception as e:
        output = [{
            "MSG": str(e),
            "STATUS": 0
        }]

    return output


# main
if __name__ == "__main__":
    try:
        # data = '{"INPUT_DATA":{"START":4,"END":5,"TEST_PATH":["/Users/gimminjong/Documents/data/iris2.csv"],"DB_INFO":{},"LABEL_COLUMN_NAME":"label"},"MODEL_INFO":{"MODEL_PATH":"Network/Tabular/XGBOOST/XGB_CLF","WEIGHT_PATH":"/Users/gimminjong/Upload/RT210113/19","DATA_TYPE":"T","OBJECT_TYPE":"C"}}'

        # data = '{"INPUT_DATA":{"START":10,"END":500,"TEST_PATH":["/Users/upload/DataSets/RT210051/DTR_FINAL.csv"],"DB_INFO":{},"LABEL_COLUMN_NAME":"Ks"},"MODEL_INFO":{"MODEL_PATH":"Network/Tabular/XGBOOST/XGB_REG","WEIGHT_PATH":"/Users/upload/InputSources/162/model/0","DATA_TYPE":"T","OBJECT_TYPE":"R"},"SERVER_PARAM":{"AI_CD":"RT20210090","SRV_IP":"0.0.0.0","SRV_PORT":10236,"IS_CD":162,"PREDICT_RESULT_URL":"/tab/aaaaa"}}'
        # data = '{"INPUT_DATA":{"START":10,"END":50,"TEST_PATH":["/Users/parksangmin/weda/2020/BluAI/tabular_dataset/clf/iris.csv"],"DB_INFO":{},"LABEL_COLUMN_NAME":"label"},"MODEL_INFO":{"MODEL_PATH":"Network/Tabular/TF/TabNetCLF","WEIGHT_PATH":"/Users/upload/AiModel/CT20210206/0","DATA_TYPE":"T","OBJECT_TYPE":"C"}}'
        # param = json.loads(data)
        
        param = json.loads(sys.argv[1])
        modelStatus = modelLoad(param)
        output = runPredict(param)

        print(json.dumps(output))

    except Exception as e:
        log.error(traceback.format_exc())
        log.error(e)
        output = {
            "MSG": str(e),
            "STATUS": 0
        }

        print(json.dumps(output))
