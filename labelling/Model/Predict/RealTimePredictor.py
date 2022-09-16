# -*- coding:utf-8 -*-
'''
RealTime Predict 스크립트
'''

import io
import sys
import os
import requests
import cv2
import numpy as np
import json
import time
from PIL import Image
import traceback

from datetime import datetime
import math

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Model 폴더 찾기 위한 방법
basePath = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# Model Path 등록
sys.path.append(basePath)

from Common.Logger.Logger import logger
from Common.Utils.Utils import getConfig, label2ColorImage

import Network.KERAS.DeepLab.deeplab as deeplab
from Network.TF.YOLOv3.models import YoloV3
from Network.TF.YOLOv4.models import YoloV4
from Network.TF.TinyYOLOv3.models import YoloV3Tiny
from Network.TF.YOLOv3.dataset import transform_images


from Network.KERAS.RetinaNet import models
from Output.PredictOutput import express
from Output.Output import sendMsg

log = logger("log")
svrIp, svrPort, logPath, tempPath, datasetPath, aiPath = getConfig()


class MjpegReader():
    def __init__(self, url: str):
        self._url = url

    def iter_content(self):
        """
        Raises:
            RuntimeError
        """
        r = requests.get(self._url, stream=True)

        # parse boundary
        content_type = r.headers['content-type']
        index = content_type.rfind("boundary=")
        assert index != 1
        boundary = content_type[index + len("boundary="):] + "\r\n"
        boundary = boundary.encode('utf-8')

        rd = io.BufferedReader(r.raw)
        while True:
            self._skip_to_boundary(rd, boundary)
            length, data = self._parse_length(rd)
            yield rd.read(length), data

    def _parse_length(self, rd) -> int:
        length = 0
        data = None
        while True:
            line = rd.readline()
            if line == b'\r\n':
                return length, data
            if line.startswith(b"Content-Length"):
                length = int(line.decode('utf-8').split(": ")[1])
                assert length > 0
            if line.startswith(b"Data"):
                data = str(line.decode('utf-8').split("Data: ")[1])

    def _skip_to_boundary(self, rd, boundary: bytes):
        for _ in range(10):
            if boundary in rd.readline():
                break
        else:
            raise RuntimeError("Boundary not detected:", boundary)


# model load
def modelLoad(isCD, mdlPath, numClasses):
    try:
        model = None
        modelType = None
        mdlPathList = os.listdir(mdlPath)

        if "yolov3_tiny.tf.index" in mdlPathList:
            model = YoloV3Tiny(classes=numClasses)
            model.load_weights(os.path.join(mdlPath, "yolov3_tiny.tf")).expect_partial()
            modelType = "yolo"

        elif "yolov3.tf.index" in mdlPathList:
            model = YoloV3(classes=numClasses)
            model.load_weights(os.path.join(mdlPath, "yolov3.tf"))
            modelType = "yolo"

        elif "yolov4.tf.index" in mdlPathList:
            model = YoloV4(classes=numClasses)
            model.load_weights(os.path.join(mdlPath, "yolov4.tf"))
            modelType = "yolo"

        elif "predict.h5" in mdlPathList:
            model = models.load_model(os.path.join(mdlPath, "predict.h5"), backbone_name='resnet50')
            modelType = "retina"

        elif "deeplab.h5" in mdlPathList:
            model = deeplab.createModel(num_classes=numClasses)
            model.load_weights(os.path.join(mdlPath, "deeplab.h5"))
            modelType = "deeplab"

    except Exception as e:
        out = {"IS_CD": isCD, "CODE": "131", "MSG": str(e)}
        log.error(out)
        log.error(traceback.format_exc())
        _ = sendMsg('http://{}:{}/api/QI/report/stateCheck'.format(svrIp, svrPort), out)

    # unet, pspnet 추가

    return model, modelType


# hex color to rgb color
def hex2rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def yolo(model, imgPath, classes, target, resultPath, oriBoxes, isCD, isType, capturedTime):
    image = None
    saveImage = None
    classCD = None
    className = None
    con1, con2 = None, None
    accu1, accu2 = None, None
    logical = None
    tag, dpLabel = None, None
    location = None
    color = None
    targets = []

    for tmp in target:
        classCD = tmp["CLASS_CD"]
        className = tmp["CLASS_NAME"]
        dpLabel = tmp["DP_LABEL"]
        location = tmp["LOCATION"]
        color = tmp["COLOR"]
        accScope = tmp["ACC_SCOPE"].split(',')

        con1 = accScope[0]
        accu1 = accScope[1]
        logical = accScope[2]
        con2 = accScope[3]
        accu2 = accScope[4]

        targets.append([classCD, className, con1, accu1, logical, con2, accu2, tag, dpLabel, location, color])

    image = imgPath
    saveImage = imgPath

    img = tf.expand_dims(image, 0)
    hh, ww = image.shape[0:2]
    img = transform_images(img, 416)
    output = {"IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime, "OBJECTS": None}
    boxes, scores, detectClasses, nums = model(img)
    # print("ptime:{}".format(time.time() - st))
    resultData = {}
    objects = []
    seq = 0
    tmpTime = time.time()
    saveImageName = os.path.join(resultPath, "{}.jpg".format(tmpTime))
    resultName = os.path.join(resultPath, "{}.dat".format(tmpTime))

    for i in range(nums[0]):
        print(i)
        for j, target in enumerate(targets):
            classCD, className, con1, accu1, logical, con2, accu2, tag, dpLabel, location, color = target
            if classes[int(detectClasses[0][i])].lower() == className.lower():
                resultData, saveImage = express(boxes, logical, con1, accu1, con2, accu2, scores,
                                                i, ww, hh, target, saveImage, resultPath, oriBoxes, capturedTime)
                print("===============================================")
                print(resultData)
                print("===============================================")

                resultData["SEQ"] = seq
                seq += 1
                if resultData["RECT"] is not None:
                    objects.append(resultData)
                    cv2.imwrite(saveImageName, saveImage)
                else:
                    print("resultData['RECT']:", resultData["RECT"])
                    continue
                break
            else:
                continue
            # objects.append(resultData)

    output["RESULT_PATH"] = resultName
    output["OUTPUT_PATH"] = saveImageName
    output["OBJECT_TYPE"] = "D"
    output["OBJECTS"] = objects

    return output


# segmentation
def deeplabModel(model, imagePath, classes, target, resultPath, oriBoxes, isCD, isType, capturedTime):
    trained_image_width = 512
    mean_subtraction_value = 127.5
    targets = []
    image = None
    contours = None
    saveImage = None
    h, w = None, None
    resultData = {}
    objects = []

    image = imagePath
    saveImage = image

    output = {"IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime, "OBJECTS": None}

    for tmp in target:
        classCD = tmp["CLASS_CD"]
        className = tmp["CLASS_NAME"]
        tag = None
        dpLabel = tmp["DP_LABEL"]
        location = tmp["LOCATION"]
        color = tmp["COLOR"]
        accScope = tmp["ACC_SCOPE"].split(',')

        con1 = accScope[0]
        accu1 = accScope[1]
        logical = accScope[2]
        con2 = accScope[3]
        accu2 = accScope[4]

        targets.append([classCD, className, con1, accu1, logical, con2, accu2, tag, dpLabel, location, color])

    w, h = image.shape[:2]
    # resize to max dimension of images from training dataset
    ratio = float(trained_image_width) / np.max([h, w])
    resizedImage = np.array(Image.fromarray(image.astype('uint8')).resize((int(ratio * h), int(ratio * w))))
    resizedImage = (resizedImage / mean_subtraction_value) - 1

    # pad array to square image to match training images
    padX = int(trained_image_width - resizedImage.shape[0])
    padY = int(trained_image_width - resizedImage.shape[1])
    resizedImage = np.pad(resizedImage, ((0, padX), (0, padY), (0, 0)), mode='constant')

    # run prediction
    res = model.predict(np.expand_dims(resizedImage, 0))
    labels = np.argmax(res.squeeze(), -1)

    # remove padding and resize back to original image
    if padX > 0:
        labels = labels[:-padX]
    if padY > 0:
        labels = labels[:, :-padY]

    segmap, colorMapTmp = label2ColorImage(labels)
    segmap = np.array(Image.fromarray(segmap.astype('uint8')).resize((h, w)))

    seq = 0
    labelsCp = labels
    lh, lw = labels.shape[:2]
    idx = []
    printColor = [0 for _ in range(1000)]
    printClassName = [0 for _ in range(1000)]
    printClassCd = [0 for _ in range(1000)]
    printdpLabel = [0 for _ in range(1000)]
    printLocation = [0 for _ in range(1000)]
    colorMap = np.empty((0, 3), int)

    for targetData in targets:
        classCD, className, con1, accu1, logical, con2, accu2, tag, dpLabel, location, color = targetData
        cvtColor = hex2rgb(color)
        cvtColor = np.asarray(cvtColor)
        colorMap = np.append(colorMap, np.array([cvtColor]), axis=0)

        className.lower()
        labelFlag = False
        if className in classes:
            i = classes.index(className)
            for yy in range(lh):
                for xx in range(lw):
                    if labelsCp[yy, xx] == i:
                        printColor.insert(i, color)
                        printClassName.insert(i, className)
                        printClassCd.insert(i, classCD)
                        printdpLabel.insert(i, dpLabel)
                        printLocation.insert(i, location)
                        idx.append(i)
                        labelFlag = True
                        break

                if labelFlag:
                    break

    imgTmp = segmap
    imgTmpH, imgTmpW = imgTmp.shape[:2]
    imgTmpHR, imgTmpWR = int(imgTmpH / 4), int(imgTmpW / 4)
    imgTmp = cv2.resize(imgTmp, dsize=(imgTmpWR, imgTmpHR), interpolation=cv2.INTER_AREA)
    tmpTime = time.time()
    saveImageName = os.path.join(resultPath, "{}.jpg".format(tmpTime))
    resultName = os.path.join(resultPath, "{}.dat".format(tmpTime))
    for i in range(len(idx)):
        thresh = np.zeros((imgTmpHR, imgTmpWR, 3), np.uint8)
        for yy in range(imgTmpHR):
            for xx in range(imgTmpWR):
                if imgTmp[yy, xx, 0] == colorMapTmp[idx[i]][0] and \
                   imgTmp[yy, xx, 1] == colorMapTmp[idx[i]][1] and \
                   imgTmp[yy, xx, 2] == colorMapTmp[idx[i]][2]:

                    thresh[yy, xx, 0] = colorMap[i][0]
                    thresh[yy, xx, 1] = colorMap[i][1]
                    thresh[yy, xx, 2] = colorMap[i][2]

        thresh = cv2.resize(thresh, dsize=(imgTmpW, imgTmpH), interpolation=cv2.INTER_CUBIC)
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        try:
            thresholdValue = np.min(thresh[thresh > 10])
        except ValueError:
            thresholdValue = 127
        ret, thresh = cv2.threshold(thresh, thresholdValue, 255, cv2.THRESH_BINARY)
        contours, hierachy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if oriBoxes is None:
            oriBoxes = [0, 0, 0, 0]

        for contour in contours:
            contourX = []
            contourY = []
            contourPt = []
            x1, y1, x2, y2 = cv2.boundingRect(contour)
            boxSize = math.sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2))

            for pt in contour:
                contourX.append(float(pt[0][0]))
                contourY.append(float(pt[0][1]))

            for j in range(len(contourX)):
                contourPt.append({"X": contourX[j] + oriBoxes[0], "Y": contourY[j] + oriBoxes[1]})

            resultData = {"CLASS_CD": printClassCd[idx[i]], "CLASS_NAME": printClassName[idx[i]], "COLOR": printColor[idx[i]],
                          "DP_LABEL": printdpLabel[idx[i]], "LOCATION": printLocation[idx[i]], "ACCURACY": 0,
                          "POSITION": contourPt, "RAW_TIME": capturedTime, "SEQ": seq, "VALUE": boxSize}

            seq += 1
            objects.append(resultData)
        cv2.imwrite(saveImageName, saveImage)

    output["RESULT_PATH"] = resultName
    output["OUTPUT_PATH"] = saveImageName
    output["OBJECT_TYPE"] = "S"

    output["OBJECTS"] = objects
    return output


# get model file path and class file path
def getModelFilePath(mdlPath):
    modelList = os.listdir(mdlPath)
    modelFilePath = None
    classPath = None

    for path in modelList:
        if "__MACOSX" in path:
            continue

        if os.path.isdir(os.path.join(mdlPath, path)):
            modelFilePath = os.path.join(mdlPath, path)

    modelList = os.listdir(modelFilePath)
    clssDirList = os.listdir(modelFilePath)
    classPath = modelFilePath

    for path in modelList:
        if "__MACOSX" in path:
            continue
        if os.path.isdir(os.path.join(modelFilePath, path)):
            modelFilePath = os.path.join(modelFilePath, path)

    for tmp in clssDirList:
        if "._" in tmp:
            continue
        elif ".txt" in tmp or ".csv" in tmp:
            classPath = os.path.join(classPath, tmp)

    return modelFilePath, classPath


def detection(predictType, mdlPath, target, resultPath, server, isCD, isType, frameRatio):
    # model path
    modelFilePath, classPath = getModelFilePath(mdlPath)

    with open(classPath, 'r') as f:
        numClasses = (len(f.readlines()))
        classesTmp = [c.strip() for c in open(classPath).readlines()]
        classes = []

        for tmp in classesTmp:
            tmp = tmp.split(",")
            classes.append(tmp[1].lower())

    # model load
    model, modelType = modelLoad(isCD, modelFilePath, numClasses)
    time.sleep(3)

    # mr = MjpegReader(server)
    boxes = [0, 0, 0, 0]
    cnt = 0
    resultData = {"IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": None, "OBJECTS": []}
    vc = cv2.VideoCapture(server)
    while True:
        # for content, data in mr.iter_content():
        # data = json.loads(data)
        # frame = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        ret, frame = vc.read()
        today = datetime.today()
        capturedTime = "{}-{}-{} {}:{}:{}.{}".format(today.year, today.month, today.day,
                                                     today.hour, today.minute, today.second, today.microsecond)
        h, w = frame.shape[:2]
        result = []
        if modelType == "yolo":
            resultData = yolo(model, frame, classes, target, resultPath, boxes, isCD, isType, capturedTime)

        elif modelType == 'deeplab':
            resultData = deeplabModel(model, frame, classes, target, resultPath, boxes, isCD, isType, capturedTime)

        if len(resultData["OBJECTS"]) != 0:
            result.append(resultData)
            resultOutput = {"OBJECTS": resultData["OBJECTS"]}
            with open(resultData["RESULT_PATH"], "w") as f:
                json.dump(resultOutput, f)

            _ = sendMsg('http://{}:{}/api/QI/report/predictResult'.format(svrIp, svrPort), result)
        cnt += 1
        # if data is not None  :
        #     if predictType == 'F' :
        #         isRect = data["isRect"]
        #         capturedTime = data["RAW_TIME"]
        #         if isRect is not None:
        #             if isRect is not False :
        #                 boxes = data["BOXES"]
        #                 print(boxes)
        #                 x = int(boxes[0] * w)
        #                 y = int(boxes[1] * h)
        #                 w = int(boxes[2] * w)
        #                 h = int(boxes[3] * h)
        #                 roiImage = frame[y:h , x:w]
        #                 if modelType == "yolo" :
        #                     resultData = yolo(model, roiImage, classes, target, resultPath, boxes, isCD, isType, capturedTime)

        #                 elif modelType == 'deeplab':
        #                     resultData = deeplabModel(model, roiImage, classes, target,
        #                                               resultPath, boxes, isCD, isType, capturedTime)

        #             else :
        #                 if modelType == "yolo" :
        #                     resultData = yolo(model, frame, classes, target, resultPath, boxes, isCD,  isType, capturedTime)

        #                 elif modelType == 'deeplab':
        #                     resultData = deeplabModel(model, frame, classes, target,
        #                                               resultPath, boxes, isCD,  isType, capturedTime)

        #     elif predictType == 'R' :
        #         isRect = data["isRect"]
        #         if isRect is not None:
        #             boxes = data["BOXES"]
        #             if isRect is not False :
        #                 roiImage = frame[boxes[1]:boxes[3], boxes[0]:boxes[2]]
        #                 if modelType == "yolo" :
        #                     resultData = yolo(model, roiImage, classes, target,
        #                                       resultPath, boxes, isCD,  isType, capturedTime)

        #                 elif modelType == 'deeplab':
        #                     resultData = deeplabModel(model, roiImage, classes, target,
        #                                               resultPath, boxes, isCD,  isType, capturedTime)

        #             else :
        #                 if modelType == "yolo" :
        #                     resultData = yolo(model, frame, classes, target, resultPath, boxes, isCD,  isType, capturedTime)

        #                 elif modelType == 'deeplab':
        #                     resultData = deeplabModel(model, frame, classes, target,
        #                                               resultPath, boxes, isCD, isType, capturedTime)
        #     # print(resultData)
        #     if len(resultData["OBJECTS"]) is not 0 :
        #         result.append(resultData)
        #         resultOutput={"OBJECTS":resultData["OBJECTS"]}
        #         with open(resultData["RESULT_PATH"], "w") as f:
        #             json.dump(resultOutput, f)

        #         res = requests.post(sendServer, headers=headers, data=json.dumps(result))
        # cnt += 1


def makeDir(dirPath):
    if not os.path.isdir(dirPath):
        os.makedirs(dirPath, exist_ok=True)
    return True


if __name__ == "__main__":
    try:
        string = sys.argv[1]
        gpuIdx = sys.argv[2]
        # while True:
        #     pipeData = sys.stdin.read(1024)
        #     if pipeData is not '':
        #         string = string + pipeData
        #     else :
        #         break
        #     # print(string)
        data = json.loads(string)

        if gpuIdx != "-1":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuIdx)
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                # 텐서플로가 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
                try:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpus[int(gpuIdx)],
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7266)])
                except Exception as e:
                    log.error(traceback.format_exc())
                    print(e)

        predictType = data["TYPE"]
        isCD = data["IS_CD"]
        isType = data["IS_TYPE"]
        mdlPath = data["MDL_PATH"]
        target = data["TARGET_CLASS"]
        HW_IP = data["HW_IP"]
        port = data["HW_PORT"]
        frameRatio = int(data["FRAME_RATIO"])


        
        print("pid: ", os.getpid())

        isMakeDir = False

        server = 'http://{}:{}/video_feed'.format(HW_IP, port)

        mdlPathList = os.listdir(os.path.join(mdlPath, "model"))
        resultPath = os.path.join(mdlPath, "data", "result")

        # node server send
        # output = {"RESULT_PATH":resultPath, "IS_CD":isCD, "PORT":port}
        # res = requests.post(sendServer, headers=headers, data=json.dumps(output))

        isMakeDir = makeDir(resultPath)

        out = {"IS_CD": isCD, "CODE": "100", "MSG": None, "SRV_PID": os.getpid()}
        _ = sendMsg('http://{}:{}/api/QI/report/stateCheck'.format(svrIp, svrPort), out)
        detection(predictType, mdlPath, target, resultPath, server, isCD, isType, frameRatio)

    except Exception as e:
        out = {"IS_CD": isCD, "CODE": "130", "MSG": str(e)}
        print(e)
        log.error(out)
        log.error(traceback.format_exc())
        _ = sendMsg('http://{}:{}/api/QI/report/stateCheck'.format(svrIp, svrPort), out)
