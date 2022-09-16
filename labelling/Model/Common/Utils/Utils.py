# -*- coding: utf-8 -*-
'''
    각종 유틸 function 을 모아놓은 스크립트
    1. getConfig() : Server/config/server.json 파일의 config 내용을 가져오는 함수(srvIP, srvPort, logPath, tempPath, datasetPath, aiPath)
    2. getFps() : 영상 fps 추출
    3. hex2rgb(color) : 16진수 컬러 타입 -> opencv color type으로 변경하는 함수
'''

import cv2
import json
import os
import sys
from six import class_types
import tensorflow as tf
import numpy as np
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import tensorflow.keras.backend as tfback
sys.stderr = stderr

import traceback

import requests

try:
    from pypylon import pylon
except Exception as e:
    pass

import GPUtil
# Model 폴더 찾기 위한 방법
basePath = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir, "../")))

readConfig = os.path.join(basePath, "../Server/src/config/devServer.json")
if not os.path.isfile(readConfig):
    readConfig = os.path.join(basePath, "../Server/dist/config/server.json")

# readConfig = None

# Model Path 등록
sys.path.append(basePath)

from Common.Process.Process import prcSendData, prcErrorData

def setConfig(path):
    global readConfig
    readConfig = path


def getLogLevel():
    global readConfig
    logLevel = "INFO"
    if readConfig is None:
        return logLevel
    try:
        configPath = os.path.abspath(os.path.join(basePath, readConfig))
        if os.path.isfile(configPath):
            with open(configPath) as jsonFile:
                config = json.load(jsonFile)
                logLevel = config["Log"]["level"].upper()

            return logLevel

    except Exception as e:
        prcErrorData(__file__, e)
        print(traceback.format_exc())


def getConfig():
    global readConfig
    srvIp, srvPort = None, None
    logPath, tempPath, datasetPath, aiPath = None, None, None, None
    if readConfig is None:
        return srvIp, srvPort, logPath, tempPath, datasetPath, aiPath
    try:
        configPath = os.path.abspath(os.path.join(basePath, readConfig))

        if os.path.isfile(configPath):
            with open(configPath) as jsonFile:
                config = json.load(jsonFile)

                srvIp = config["ip"]
                srvPort = config["port"]
                logPath = config["Log"]["dirPath"]
                tempPath = config["tempPath"]
                datasetPath = config["datasetPath"]
                aiPath = config["aiPath"]

            return srvIp, srvPort, logPath, tempPath, datasetPath, aiPath

    except Exception as e:
        prcErrorData(__file__, e)
        print(traceback.format_exc())


def getFps(filePath):
    vc = cv2.VideoCapture(filePath)
    fps = vc.get(cv2.CAP_PROP_FPS)
    return fps


# hex color to rgb color
def hex2rgb(color):
    color = color.lstrip('#')
    lv = len(color)
    return tuple(int(color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def makeDir(dirPath):
    if not os.path.isdir(dirPath):
        os.makedirs(dirPath, exist_ok=True)
    return True


def getGpus():
    # global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]


# segmentation colormap
def createLabelColorMap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
      A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def createMask(predMask):
    predMask = tf.argmax(predMask, axis=-1)
    return predMask[0]


def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train


def label2ColorImage(label):
    """Adds color defined by the dataset colormap to the label.
    Args:
      label: A 2D array with integer type, storing the segmentation label.
    Returns:
      result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.
    Raises:
      ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    # if label.ndim != 2:
    #     raise ValueError('Expect 2-D input label')

    colormap = createLabelColorMap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label], colormap


def getClasses(aiCd, objectType, mdlPath):
    srvIp, srvPort, logPath, tempPath, datasetPath, aiPath = getConfig()
    classFPath = None

    newMdlPath = os.path.abspath(os.path.join(mdlPath, '..'))

    if aiCd is not None:
        if aiCd == "YOLOV3" or aiCd == "EFFICIENTDET" or aiCd == "YOLOV4":
            classFPath = os.path.join(basePath, "Common/Classes/cocoClasses.txt")
            classes = [c.strip() for c in open(classFPath, encoding='utf-8').readlines()]
        elif aiCd == "DEEP-LAB":
            classFPath = os.path.join(basePath, "Common/Classes/vocClasses.txt")
            classes = [c.strip() for c in open(classFPath, encoding='utf-8').readlines()]
        elif aiCd == "eifficientnet":
            classFPath = os.path.join(basePath, "Common/Classes/imageNetClasses.txt")
            classes = [c.strip() for c in open(classFPath, encoding='utf-8').readlines()]
        else:
            classFPath = os.path.join(newMdlPath, "classes.json")
            with open(classFPath, "r") as f:
                data = json.load(f)
            classTmp = []

            for tmp in data["CLASS_INFO"]:
                classTmp.append((tmp["CLASS_CD"], tmp["CLASS_NAME"]))

            classTmp.sort(key=lambda x: x[0])
            classes = [c[1] for c in classTmp]

    else:
        classFPath = os.path.join(newMdlPath, "classe.json")
        classTmp = []
        with open(classFPath, "r") as f:
            data = json.load(f)

        for tmp in data["CLASS_INFO"]:
            classTmp.append((tmp["CLASS_CD"], tmp["CLASS_NAME"]))

        classTmp.sort(key=lambda x: x["CLASS_CD"])
        classes = [c[1] for c in classTmp]

    return classes


def connectCamera(serial, width, height, exposureTime):
    err = None
    try:
        '''
        # basler camera bind
        tlFactory = pylon.TlFactory.GetInstance()
        devInfo = pylon.DeviceInfo()
        devInfo.SetSerialNumber("{}".format(serial))

        camera = pylon.InstantCamera(tlFactory.CreateFirstDevice(devInfo))
        camera.Open()
        '''
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

        # Grabing Continusely (video) with minimal delay
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        camera.Open()
        # camera settings
        camera.ExposureTime.SetValue(exposureTime)
        # camera.ExposureAuto.SetValue("Once")
        # camera.Width = width
        # camera.Height = height

        camera.GainAuto.SetValue("Once")
        camera.AcquisitionFrameRate.SetValue(100.0)
        return camera, err

    except Exception as e:
        err = e
        return None, err


def getGpuState():
    gpus = GPUtil.getGPUs()
    output = []

    if len(gpus) == 0:
        output = []

    else:
        for gpu in gpus:
            # print(gpu.memoryUtil)
            output.append({"GPU_NAME": gpu.name, "MEM_TOTAL": gpu.memoryTotal, "MEM_USED": gpu.memoryUsed})

    return output


def getUsableGPU(gpuState):
    for idx, gpu in enumerate(gpuState):
        memTotal = gpu["MEM_TOTAL"]
        memUsed = gpu["MEM_USED"]

        gpuId = idx
        if memUsed == 0.0:
            return gpuId
            break
        else:
            memUsagePer = (float(memUsed / memTotal) * 100)
            if memUsagePer >= 90.0:
                continue
            else:
                return gpuId
                break
    return -1


def getTotalUsableGPU():
    gpuState = getGpuState()
    gpuUsable = []
    for idx, gpu in enumerate(gpuState):
        memTotal = float(gpu["MEM_TOTAL"])
        memUsed = float(gpu["MEM_USED"])
        memUsable = float(memTotal - memUsed) / memTotal * 100
        gpuUsable.append({"GPU_ID": idx, "MEM_USED": memUsed, "MEM_TOTAL": memTotal, "MEM_USABLE": memUsable})
    return gpuUsable


# PIPE get data
def getStdInData():
    s = ''
    while True:
        r = sys.stdin.read(1024)
        if r != '':
            s = s + r
        else:
            break
    return json.loads(s)


# set parameter
def setParameter(cls, data):
    for key in data.keys():
        cls[key] = data[key]
    cls["GPU_RATE"] = float(data["GPU_RATE"] / 100)
    cls["GPU_STATUS"] = json.loads(sys.argv[1])
    cls["START_EPOCH"] = int(sys.argv[2])
    return cls


# get encode data
def getEnData(colNames, xTest, param):
    encodeData = []
    for col in colNames:
        for i in range(len(xTest)):
            tmp = xTest[i]
            if tmp["HEADER"] == col:
                if type(tmp["VALUE"]) == str:
                    encodeData.append(param["ENCODER_DATA"][tmp["HEADER"]][tmp["VALUE"]])
                else:
                    encodeData.append(tmp["VALUE"])

    return encodeData

# get Dict values to list for gridSearchCV -> ex) "epoch" : 5 -> [5]
def getDictData(dictData):
    for k, v in dictData.items():
        if k == "n_estimators":
            dictData[k] = range(1, v+1)
        elif k == "max_iter":
            dictData[k] = range(1, v+1)
        elif k == "n_neighbors":
            dictData[k] = range(1, v+1)
        else:
            dictData[k] = [v]

    return dictData
