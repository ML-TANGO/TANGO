# -*- coding:utf-8 -*-
'''
RealTime Predict 스크립트
'''

import sys
import os
import cv2
import json
import time
import traceback
import tensorflow as tf

import numpy as np

from threading import Thread

from datetime import datetime

# Model 폴더 찾기 위한 방법
basePath = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# Model Path 등록
sys.path.append(basePath)

from Common.Logger.Logger import logger
from Common.Process.Process import prcSendData, prcGetArgs, prcErrorData, prcClose, prcLogData
from Common.Model.PredictModel import predictLoadModel
from Dataset.ImageDataSet import predictDataset
from Predict.AutoLabeling import classification, detection, segmentation
from Common.Utils.Utils import getClasses, makeDir, getConfig
from Output.Output import sendMsg
from Output.PredictOutput import operator


log = logger("log")
svrIp, svrPort, logPath, tempPath, datasetPath, aiPath = getConfig()

currFrame = None
isError = False
errMsg = None
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def runPredictor(AI_CD, IS_CD, modelInfo, usrScript):
    global currFrame
    global isError

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    try:
        MDL_PATH = modelInfo["MDL_PATH"]
        OBJECT_TYPE = modelInfo["OBJECT_TYPE"]
        
        if AI_CD == 'YOLOV3' or AI_CD == 'EFFICIENTDET' or AI_CD == 'YOLOV4' or AI_CD == 'DEEP-LAB' or AI_CD == 'efficientnet':
            saveResultPath = os.path.abspath(os.path.join(aiPath, "../InputSources", str(IS_CD), "data/result"))
        else:
            saveResultPath = os.path.abspath(os.path.join(MDL_PATH, "../../", "data/result"))

       
        _ = makeDir(saveResultPath)
        model, modelName, inputShape = predictLoadModel(OBJECT_TYPE, IS_CD, MDL_PATH)

        log.info(MDL_PATH)
        if AI_CD == 'YOLOV3' or AI_CD == 'EFFICIENTDET' or AI_CD == 'YOLOV4' or AI_CD == 'DEEP-LAB' or AI_CD == 'efficientnet':
            classes = getClasses(AI_CD, OBJECT_TYPE, MDL_PATH)
        else:
            classes = getClasses(IS_CD, OBJECT_TYPE, MDL_PATH)

        TARGET_CLASS = modelInfo["TARGET_CLASS"] if "TARGET_CLASS" in modelInfo else None

        # COLOR = TARGET_CLASS["COLOR"] if TARGET_CLASS is not None else None
        COLOR = []
        CLASS_CD = []
        CLASS_NAME = []
        DP_LABEL = []
        LOCATION = []
        OUT_CD = []
        TARGET_CLASS.sort(key=lambda x: x["CLASS_CD"])
        for targetData in TARGET_CLASS:
            COLOR.append(targetData["COLOR"])
            CLASS_CD.append(targetData["CLASS_CD"])
            CLASS_NAME.append(targetData["CLASS_NAME"])
            DP_LABEL.append(targetData["DP_LABEL"])
            LOCATION.append(targetData["LOCATION"])
            OUT_CD.append(targetData["OUT_CD"])

        # 여기에 모델 로드하는 부분 추가
        print("[{}] Model Load {}".format(IS_CD, MDL_PATH))
        # 여기부터 프레딕트 와일
        dataType = 'V'
        while True:
            seq = 0
            # 여기에 모델 프레딕트 하는 부분 추가
            if currFrame is not None:
                labels = []
                targetFrame = currFrame
                saveFrame = currFrame
                targetFrame, oriH, oriW, resizeH, resizeW = predictDataset(dataType, modelName, OBJECT_TYPE, targetFrame, inputShape)

                # log.info("[{}] Start detection".format(IS_CD))
                if OBJECT_TYPE == "C":
                    classes.sort()
                    predict = model.predict(targetFrame)
                    log.debug(predict)
                    maxIdx = np.argmax(predict[0])
                    classDbNm = classes[maxIdx]
                    classAcc = float(predict[0][maxIdx])

                    for targetData in TARGET_CLASS:
                        targetClassName = targetData["CLASS_NAME"]
                        accScope = targetData["ACC_SCOPE"]
                        outCd = targetData["OUT_CD"]
                        log.debug(
                            "PREDICT : {}, CLASS : {}, targetClassName : {}, ACCURACY : {}".format(
                                predict, classDbNm, targetClassName, classAcc))

                        if targetClassName == classDbNm:
                            label = operator(
                                classAcc,
                                accScope,
                                COLOR[maxIdx],
                                "C",
                                targetClassName,
                                imgW=None, imgH=None, boxes=None
                            )
                            if label is not None:
                                label["FRAME_NUMBER"] = 0
                                labels.append(label)
                                break
                    # tmp = []
                    # for idx, acc in enumerate(predict[0]):
                    #     tmp.append([idx, acc])
                    # tmp = sorted(tmp, key=lambda x: -x[1])
                    # classification(tmp, labels, classes, COLOR, TARGET_CLASS, OBJECT_TYPE, 0)

                elif OBJECT_TYPE == "D":
                    detection(
                        "I",
                        model,
                        classes,
                        targetFrame,
                        oriH, oriW,
                        COLOR,
                        0,
                        labels,
                        0,
                        modelName,
                        TARGET_CLASS,
                        OBJECT_TYPE
                    )
                elif OBJECT_TYPE == "S":
                    segmentation(
                        "I",
                        model,
                        classes,
                        targetFrame,
                        oriH, oriW,
                        COLOR,
                        0,
                        labels,
                        modelName,
                        TARGET_CLASS
                    )
                today = datetime.today()
                capturedTime = "{}-{}-{} {}:{}:{}.{}".format(
                    today.year,
                    today.month,
                    today.day,
                    today.hour,
                    today.minute,
                    today.second,
                    today.microsecond
                )
                OBJECTS = []
                ticTime = time.time()
                saveImgPath = os.path.join(saveResultPath, "{}.jpg".format(ticTime))
                saveDataPath = os.path.join(saveResultPath, "{}.dat".format(ticTime))

                for labelData in labels:
                    if labelData is not None:
                        for idx, colorData in enumerate(COLOR):
                            if labelData["COLOR"] == colorData:
                                resultData = {
                                    "CLASS_CD": CLASS_CD[idx],
                                    "CLASS_NAME": CLASS_NAME[idx],
                                    "COLOR": COLOR[idx],
                                    "DP_LABEL": DP_LABEL[idx],
                                    "LOCATION": LOCATION[idx],
                                    "ACCURACY": labelData["ACCURACY"] if OBJECT_TYPE != "S" else 0,
                                    "POSITION": labelData["POSITION"] if OBJECT_TYPE != "C" else [],
                                    "RAW_TIME": capturedTime,
                                    "SEQ": seq,
                                    "VALUE": labelData["VALUE"] if OBJECT_TYPE != "C" else 0,
                                    "OUT_CD": OUT_CD[idx]
                                }
                                seq += 1
                                OBJECTS.append(resultData)
                sendServerData = {
                    "IS_CD": IS_CD,
                    "IS_TYPE": "R",
                    "RAW_TIME": capturedTime,
                    "OBJECT_TYPE": OBJECT_TYPE,
                    "POLYGON_DATA": [],
                    "RESULT_PATH": saveDataPath,
                    "OUTPUT_PATH": saveImgPath,
                    "USER_SCRIPT": usrScript
                }
                if getCenter(saveFrame):
                    cv2.imwrite(saveImgPath, saveFrame)
                    sendServerData["POLYGON_DATA"] = OBJECTS
                # log.debug(sendServerData)
                result = []
                if len(sendServerData["POLYGON_DATA"]) != 0:
                    result.append(sendServerData)
                    resultOutput = {"POLYGON_DATA": sendServerData["POLYGON_DATA"]}
                    log.debug(sendServerData)
                    with open(sendServerData["RESULT_PATH"], "w") as f:
                        json.dump(resultOutput, f)
                    _ = sendMsg('http://{}:{}/api/QI/report/predictResult'.format(svrIp, svrPort), result)
            else:
                continue
        return

    except Exception as e:
        errMsg = str(e)
        isError = True
        out = {"status": 0, "IS_CD": IS_CD, "MSG": errMsg}
        _ = sendMsg('http://{}:{}/api/QI/report/stateCheck'.format(svrIp, svrPort), out)
        log.debug("EXCEPTION {}".format(traceback.format_exc()))
        return


def getCenter(img):
    isCenter = False
    ratio = 0.4
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    bigArea = -1
    bigIdx = -1
    if len(contours) == 0:
        return isCenter
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > bigArea:
            bigArea = area
            bigIdx = i
    contourX, contourY, contourW, contourH = cv2.boundingRect(contours[bigIdx])
    _, width = img.shape[:2]
    cx = int((contourX + (contourX + contourW)) / 2)

    isCenter = True if (width * ratio) < cx < (width * (1 - ratio)) else False
    return isCenter


def getVideoFeed(HW_INFO):
    FEED_URL = HW_INFO["FEED_URL"]
    global currFrame
    global isError

    isOpened = False
    retryCnt = 0
    log.info("Connect VidoeFeed {}".format(FEED_URL))
    vc = cv2.VideoCapture(FEED_URL)

    try:
        while True:
            if retryCnt < 20:
                if not vc.isOpened():
                    retryCnt = retryCnt + 1
                    log.debug("Retry Connect {}".format(retryCnt))
                    vc = cv2.VideoCapture(FEED_URL)
                    time.sleep(1)
                else:
                    isOpened = True
                    break
            else:
                isError = True
                errMsg = "VC Not opened"
                raise Exception("VC Not opened")

        if isOpened:
            log.debug("Connect VidoeFeed {}".format(FEED_URL))
            while True:
                if isError:
                    errMsg = "VC Not opened"
                    break
                ret, frame = vc.read()
                currFrame = frame
        return

    except Exception as e:
        errMsg = str(e)
        isError = True
        out = {"status": 0, "IS_CD": IS_CD, "MSG": errMsg}
        _ = sendMsg('http://{}:{}/api/QI/report/stateCheck'.format(svrIp, svrPort), out)
        log.debug("EXCEPTION {}".format(traceback.format_exc()))


if __name__ == "__main__":
    # data = sys.argv[1]
    data = json.loads(prcGetArgs(os.getpid()))
    prcSendData(__file__, json.dumps({"TYPE": "LOG", "DATA": "RealTime Predictor Run"}))

    HW_INFO = data["HW_INFO"]
    MDL_INFO = data["MDL_INFO"]
    IS_CD = HW_INFO["IS_CD"]
    AI_CD = data["AI_CD"]
    usrScript = data["USER_SCRIPT"]
    log.debug(data)

    try:
        thList = []
        for model in MDL_INFO:
            IS_CD = model["IS_CD"]
            th = Thread(target=runPredictor, args=(AI_CD, IS_CD, model, usrScript,))
            th.start()
            thList.append(th)

        getVideoFeed(HW_INFO)
        if isError:
            raise Exception(errMsg)

    except Exception as e:
        out = {"status": 0, "IS_CD": IS_CD, "MSG": str(e)}
        _ = sendMsg('http://{}:{}/api/QI/report/stateCheck'.format(svrIp, svrPort), out)
        log.debug("EXCEPTION {}".format(traceback.format_exc()))
