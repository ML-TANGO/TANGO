import json
import os
import sys
from subprocess import Popen, PIPE
from queue import Queue
from threading import Thread
import time
import psutil
import signal

import traceback

# Model 폴더 찾기 위한 방법
basePath = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# Model Path 등록
sys.path.append(basePath)

from Common.Process.Process import prcGetArgs, prcSendData
from Common.Logger.Logger import logger
from DatasetLib.DatasetLib import DatasetLib
from Output.Output import sendMsg

log = logger("log")

pids = []
tmpPids = pids

# 서버에 전달해줘야 할 데이터
# Train Manager
# AI_CD, Train Name(xgb_clf, xgb_reg, ...), STATUS, PID(Train.py), train.py idx

# each train.py
# 1. train.py 상태 : AI_CD, Train Name(xgb_clf, xgb_reg, ...), MSG, Train Type(ML, DL), train.py idx
# 2. 학습 상태 : AI_CD, Train Name, Train Type, output(datasetlib set output), train.py idx


def signal_term_handler(sig, frame):
    global pids
    aiCd = None
    url = None
    for idx, pidData in enumerate(pids):
        # 5초마다 Train alive 체크
        log.info(pidData)
        aiCd = pidData["AI_CD"]
        url = pidData["URL"]
        pid = pidData["PID"]
        if psutil.pid_exists(pid):
            os.kill(pid, signal.SIGTERM)

    tmStatus = {
        "AI_CD": aiCd,
        "STATUS": False,
        "MSG": "Train Manager kill",
        "PID": os.getpid()
    }

    res = sendMsg(url, tmStatus)
    sys.exit(0)


# start Train.py
# input : modelPath, args(jsonData)
# output = message
def startTrain(mdlPath, args, url, AI_CD, idx, mdlName):
    global pids
    try:
        item = []
        item.append("python")
        item.append(mdlPath)
        item.append(args)

        proc = Popen(item)
        pidData = {
            "STATUS": True,
            "AI_CD": AI_CD,
            "PID": proc.pid,
            "MDL_IDX": idx,
            "MODEL_NAME": mdlName,
            "URL": url
        }
        pids.append(pidData)
    except Exception as e:
        log.error(str(e))


# run train model Thread
# input : mdlPath, args(jsonData)
def runProcess(mdlPath, args, url, AI_CD, idx, mdlName):
    try:
        proc = Thread(target=startTrain, args=(mdlPath, args, url, AI_CD, idx, mdlName))
        proc.start()
        proc.join()

    except Exception as e:
        log.error(str(e))

    # return output


# 쓰레드로 돌린 pid 들 생존 체크
# input : X
# output : X
# 각 train.py status 보내기?
def checkTrainAlive(url, AI_CD, mdlNames):
    global tmpPids
    global pids

    output = {}
    try:
        for idx, pidData in enumerate(pids):
            # 5초마다 Train alive 체크
            pid = pidData["PID"]
            mdlIdx = pidData["MDL_IDX"]
            mdlName = pidData["MODEL_NAME"]
            if psutil.pid_exists(pid):
                p = psutil.Process(pid).as_dict()
                procStatus = p["status"]
                if procStatus == 'zombie':
                    os.kill(pid, signal.SIGTERM)
                    pids.remove(pidData)

                else:
                    output["PID"] = pid
                    output["MODEL_NAME"] = mdlName,
                    output["MDL_IDX"] = mdlIdx,
                    output["STATUS"] = True
                    # _ = sendMsg(url, output)
                    continue

            else:
                if pid in pids:
                    output["PID"] = None
                    output["MODEL_NAME"] = mdlName,
                    output["MDL_IDX"] = mdlIdx,
                    output["STATUS"] = False
                    output["MSG"] = None
                    # _ = sendMsg(url, output)
                    os.kill(pid, signal.SIGTERM)
                    pids.remove(pidData)

                else:
                    continue

            # _ = sendMsg(url, output)

    except Exception as e:
        # prcSendData(__file__, json.dumps({"TYPE": "LOG", "DATA": "Trainer Run Fail"}))
        log.error(str(e))
        log.error(str(traceback.format_exc()))



if __name__ == "__main__":
    mdlNames = []
    mdlIdxs = []
    try:
        # trainDatas = json.loads(prcGetArgs(0))
        # data = '{"INPUT_DATA":{"FILE_PATH":"/Users/parksangmin/Downloads/","TRAIN_FILE_NAMES":["smallkdd.csv"],"DELIMITER":",","SPLIT_RATIO":0.25,"LABEL_COLUMN_NAME":"label","MAPING_INFO":{"aaa":"aaa"}},"SERVER_PARAM":{"AI_CD":"TD000001","SRV_IP":"192.168.0.6","SRV_PORT":5000,"TRAIN_RESULT_URL":"trainBinLog","TRAIN_STATE_URL":"trainStatusLog"},"MODEL_INFO":[{"MODEL_PATH":"Network/Tabular/TF/TabNetCLF","HYPER_PARAM":{"num_decision_steps":7,"relaxation_factor":1.5,"sparsity_coefficient":1e-05,"batch_momentum":0.98},"MODEL_PARAM":{"BATCH_SIZE":128,"EPOCH":10}}]}'
        # log.info(sys.argv[1])

        # data = '{"INPUT_DATA":{"TRAIN_PATH":[],"DELIMITER":",","SPLIT_YN":"Y","DATASET_SPLIT":10,"TEST_DATASET_CD":null,"TEST_PATH":null,"LABEL_COLUMN_NAME":"label","MAPING_INFO":[{"DATASET_CD":"CT210114","COLUMN_NM":"sepal_length","COLUMN_ALIAS":null,"DEFAULT_VALUE":null,"IS_CLASS":0,"COLUMN_IDX":0,"checked":1},{"DATASET_CD":"CT210114","COLUMN_NM":"sepal_width","COLUMN_ALIAS":null,"DEFAULT_VALUE":null,"IS_CLASS":0,"COLUMN_IDX":1,"checked":1},{"DATASET_CD":"CT210114","COLUMN_NM":"petal_length","COLUMN_ALIAS":null,"DEFAULT_VALUE":null,"IS_CLASS":0,"COLUMN_IDX":2,"checked":1},{"DATASET_CD":"CT210114","COLUMN_NM":"petal_width","COLUMN_ALIAS":null,"DEFAULT_VALUE":null,"IS_CLASS":0,"COLUMN_IDX":3,"checked":1},{"DATASET_CD":"CT210114","COLUMN_NM":"label","COLUMN_ALIAS":null,"DEFAULT_VALUE":null,"IS_CLASS":1,"COLUMN_IDX":4,"checked":1}],"DB_INFO":{"DATASET_CD":"CT210114","DB_SEQ":0,"CLIENT":"mysql","ADDRESS":"www.wedalab.com","PORT":9156,"DBNAME":"BLUAI_AI","USER":"bluai","PASSWORD":"WEDA_BLUAI_0717","QUERY":"select * from BLUAI_AI.TEST_IRIS","IS_TEST":0,"LIMIT":5}},"SERVER_PARAM":{"AI_CD":"CT20210198","AI_PATH":"/Users/upload/AiModel/CT20210198","SRV_IP":"127.0.0.1","SRV_PORT":10236,"TRAIN_RESULT_URL":"/tab/binary/trainResultLog","TRAIN_STATE_URL":"/tab/binary/binaryStatusLog","TRAINING_INFO_URL":"/tab/binary/trainInfoLog"},"MODEL_INFO":[{"DATA_TYPE":"T","OBJECT_TYPE":"C","MODEL_NAME":"XGBClassifier","MODEL_TYPE":"ML","MDL_ALIAS":"XGBClassifier_0","MDL_IDX":0,"n_estimators":"100","max_depth":"6","min_child_weight":"1","gamma":"0","colsample_bytree":"1","colsample_bylevel":"1","colsample_bynode":"1","subsample":"1","learning_rate":"0.3","early_stopping":"TRUE","monitor":"accuracy","mode":"auto","MDL_PATH":"Network/Tabular/XGBOOST/XGB_CLF"},{"DATA_TYPE":"T","OBJECT_TYPE":"C","MODEL_NAME":"RandomForestClassifier","MODEL_TYPE":"ML","MDL_ALIAS":"RandomForestClassifier_1","MDL_IDX":1,"n_estimators":"100","max_depth":"None","min_samples_split":"2","min_samples_leaf":"1","max_leaf_nodes":"None","max_features":"auto","early_stopping":"TRUE","monitor":"accuracy","mode":"auto","MDL_PATH":"Network/Tabular/SCIKIT/RF_CLF"}]}'

        # trainDatas = json.loads(data)
        trainDatas = json.loads(sys.argv[1])
        signal.signal(signal.SIGTERM, signal_term_handler)

        # prcSendData(__file__, json.dumps({"TYPE": "LOG", "DATA": "Trainer Run"}))
        tmStatus = {
            "AI_CD": trainDatas["SERVER_PARAM"]["AI_CD"],
            "STATUS": True,
            "MSG": "Train Manager Run",
            "PID": os.getpid()
        }
        url2 = 'http://{}:{}/api{}'.format(
            trainDatas["SERVER_PARAM"]["SRV_IP"],
            trainDatas["SERVER_PARAM"]["SRV_PORT"],
            "/tab/binary/trainmanagerinfo"
        )
        url = 'http://{}:{}/api{}'.format(
            trainDatas["SERVER_PARAM"]["SRV_IP"],
            trainDatas["SERVER_PARAM"]["SRV_PORT"],
            trainDatas["SERVER_PARAM"]["TRAIN_STATE_URL"],
        )
        res = sendMsg(url2, tmStatus)
        log.info("Train Manager Run")

        for idx, trainData in enumerate(trainDatas["MODEL_INFO"]):
            mdlName = trainData["MODEL_NAME"]
            mdlIdx = trainData["MDL_IDX"]
            mdlNames.append(mdlName)

            mdlIdxs.append(mdlIdx)
            # trainData.update({'MDL_IDX': mdlIdx})
            # trainData.update({"MODEL_NAME": mdlName})

            params = {
                "INPUT_DATA": trainDatas["INPUT_DATA"],
                "SERVER_PARAM": trainDatas["SERVER_PARAM"],
                "MODEL_INFO": trainData
            }
            AI_CD = trainDatas["SERVER_PARAM"]["AI_CD"]
            mdlPath = os.path.join(basePath, trainData["MDL_PATH"], "train.py")
            params = json.dumps(params)

            runProcess(mdlPath, params, url2, AI_CD, mdlIdx, mdlName)
            signal.signal(signal.SIGTERM, signal_term_handler)

        while True:
            signal.signal(signal.SIGTERM, signal_term_handler)
            checkTrainAlive(url, AI_CD, mdlNames)
            if len(pids) <= 0:
                break

        log.info("train manager done")
        tmStatus = {
            "AI_CD": trainDatas["SERVER_PARAM"]["AI_CD"],
            "STATUS": False,
            "MSG": "Train Manager done",
            "PID": os.getpid()
        }
        res = sendMsg(url2, tmStatus)
        # os.kill(os.getpid(), signal.SIGTERM)
        # sys.exit()

    except Exception as e:
        log.error("Train Manager Run Fail")
        log.error(str(e))
        tmStatus = {
            "AI_CD": trainDatas["SERVER_PARAM"]["AI_CD"],
            "STATUS": False,
            "MSG": str(e),
            "PID": None
        }
        res = sendMsg(url2, tmStatus)
        log.error(str(traceback.format_exc()))