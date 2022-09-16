# -*- coding:utf-8 -*-
'''
Static Data Predict 스크립트
'''

# image lib
import cv2
from PIL import Image

# etc lib
import os
import sys
import json
import numpy as np
import math
from datetime import datetime
import traceback

# request lib
import requests

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Model 폴더 찾기 위한 방법
basePath = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir, '../')))

# Model Path 등록
sys.path.append(basePath)


from Common.Logger.Logger import logger
from Common.Utils.Utils import getConfig, hex2rgb, label2ColorImage, makeDir
import Network.KERAS.DeepLab.deeplab as deeplab
from Network.TF.YOLOv3.models import YoloV3
from Network.TF.TinyYOLOv3.models import YoloV3Tiny
from Network.TF.YOLOv3.dataset import transform_images
from Network.KERAS.RetinaNet import models
from Output.PredictOutput import express
from Output.Output import sendMsg


log = logger("log")
svrIp, svrPort, logPath, tempPath, datasetPath, aiPath = getConfig()


sendServer = 'http://{}:{}/api/QI/report/predictResult'.format(svrIp, svrPort)
sendServer2 = 'http://{}:{}/api/QI/report/stateCheck'.format(svrIp, svrPort)
headers = {"Content-Type": "application/json; charset=utf-8"}
IMAGE_SHAPE = (512, 512, 3)


# # 수정해야함. - smpark
# def getResultRetina(objectType, cnt , boxes, scores, ww, hh, target, saveImage, resultPath) :
#     resultData = {}
#     classCD, className, con1, accu1, logical, con2, accu2, tag, dpLabel, location, color = target
#     cvColor = hex2rgb(color)

#     x = float(np.array(boxes[0][i][0]) * ww)
#     y = float(np.array(boxes[0][i][1]) * hh)
#     w = float(np.array(boxes[0][i][2]) * ww)
#     h = float(np.array(boxes[0][i][3]) * hh)

#     saveImage = cv2.rectangle(saveImage, (int(x), int(y)), (int(w), int(h)), cvColor, 2)

#     # resultData = {"CLASS_CD": None, "CLASS_NAME": None, "TAG": None, "COLOR": None, "DP_LABEL": None,
#                     "LOCATION": None, "accuracy": None, "rect": None, "RESULT_PATH": None}
#     if objectType == 'v' or objectType == 'V' :
#         resultData["FRAME_NO"] = cnt
#     else :
#         resultData["FRAME_NO"] = None
#     resultData["CLASS_CD"] = classCD
#     resultData["CLASS_NAME"] = className
#     resultData["TAG"] = tag
#     resultData["COLOR"] = color
#     resultData["DP_LABEL"] = dpLabel
#     resultData["LOCATION"] = location
#     resultData["ACCURACY"] = float(np.array(scores[0][i]))
#     resultData["RECT"] =  {"X1": x, "Y1": y, "X2": w, "Y2": h}
#     resultData["RESULT_PATH"] = resultPath

#     return resultData, saveImage


# Image read
def readImage(filePath, IMAGE_SHAPE):
    img = cv2.imread(filePath, cv2.IMREAD_COLOR)  # cv2.IMREAD_GRAYSCALE
    return cv2.resize(
        img, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]), interpolation=cv2.INTER_CUBIC
    )


def prepData(images, IMAGE_SHAPE):
    image = readImage(images, IMAGE_SHAPE)
    data = image
    return data.reshape(1, IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2])


# keras retinanet for train model -> for predict model change
def cvtModel(modelPath):
    modelName = os.path.join(modelPath, "retinanet.h5")
    model = models.load_model(modelName, backbone_name="resnet50")
    models.check_training_model(model)
    model = models.convert_model(
        model, nms=False, class_specific_filter=False, anchor_params=None
    )

    model.save(os.path.join(modelPath, "predict.h5"))


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


# model load
def modelLoad(isCD, mdlPath, numClasses):
    model = None
    modelType = None
    mdlPathList = os.listdir(mdlPath)
    print(mdlPath, numClasses)
    try:
        if "yolov3_tiny.tf.index" in mdlPathList:
            model = YoloV3Tiny(classes=numClasses)
            model.load_weights(os.path.join(mdlPath, "yolov3_tiny.tf")).expect_partial()
            modelType = "yolo"

        elif "yolov3.tf.index" in mdlPathList:
            model = YoloV3(classes=numClasses)
            model.load_weights(os.path.join(mdlPath, "yolov3.tf"))
            modelType = "yolo"

        elif "predict.h5" in mdlPathList:
            model = models.load_model(os.path.join(mdlPath, "predict.h5"), backbone_name='resnet50')
            modelType = "retina"

        elif "deeplab.h5" in mdlPathList:
            model = deeplab.createModel(num_classes=numClasses)
            model.load_weights(os.path.join(mdlPath, "deeplab.h5"))
            modelType = "deeplab"

    except Exception as e:
        out = {"IS_CD": isCD, "CODE": "141", "MSG": str(e)}
        log.error(traceback.format_exc())
        _ = sendMsg('http://{}:{}/api/QI/report/stateCheck'.format(svrIp, svrPort), out)

    # unet, pspnet 추가

    return model, modelType


# Detection Model
# Save result image 추가 - smpark
def yolov3(model, imgPath, classes, target, resultPath, isCD, isType, fileSEQ):
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
    resultData = {}
    objects = []
    seq = 0
    oriBoxes = [0, 0, 0, 0]
    foundClasses = []
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

    # video인 경우 frame
    if isType == 'v' or isType == "V":
        image = imgPath
        saveImage = imgPath
        output = {}

    # image 인 경우 image Path
    elif isType == 'i' or isType == 'I':
        try:
            image = tf.image.decode_image(open(imgPath, "rb").read(), channels=3)
            saveImage = cv2.imread(imgPath)

        except Exception as e:
            out = {"IS_CD": isCD, "CODE": "141", "MSG": str(e)}
            log.error(traceback.format_exc())
            _ = sendMsg('http://{}:{}/api/QI/report/stateCheck'.format(svrIp, svrPort), out)

        output = {"IS_CD": isCD, "IS_TYPE": isType, "FILE_INFO": None, "OBJECT_TYPE": "D"}

    img = tf.expand_dims(image, 0)
    hh, ww = image.shape[0:2]
    img = transform_images(img, 416)

    today = datetime.today()
    capturedTime = "{}-{}-{} {}:{}:{}.{}".format(today.year, today.month, today.day, today.hour, today.minute,
                                                 today.second, today.microsecond)

    boxes, scores, detectClasses, nums = model(img)
    for j, target in enumerate(targets):
        classCD, className, con1, accu1, logical, con2, accu2, tag, dpLabel, location, color = target
        classCnt = 0
        avgAcc = 0.0
        classInfo = {"CLASS_CD": classCD, "CLASS_NAME": className, "COLOR": color, "DP_LABEL": dpLabel, "LOCATION": location,
                     "ACCURACY": None, "CLASS_CNT": None}
        for i in range(nums[0]):
            if classes[int(detectClasses[0][i])].lower() == className.lower():
                resultData, saveImage = express(boxes, logical, con1, accu1, con2, accu2, scores, i, ww, hh, target, saveImage,
                                                resultPath, oriBoxes, capturedTime)
                classCnt += 1
                avgAcc += float(resultData["ACCURACY"])

                resultData["SEQ"] = seq
                del resultData['RAW_TIME']
                seq += 1
                if resultData["RECT"] is not None:
                    objects.append(resultData)
                else:
                    continue

        if classCnt != 0:
            avgAcc /= classCnt
        else:
            avgAcc = 0
        classInfo["ACCURACY"] = avgAcc
        classInfo["CLASS_CNT"] = classCnt
        foundClasses.append(classInfo)

    # saveObj["OBJECTS"] = objects
    output["FOUND_CLASSES"] = foundClasses
    output["OBJECTS"] = objects
    # return saveObj, output
    return output


# segmentation
def deeplabModel(model, imagePath, classes, target, resultPath, isCD, isType, fileSEQ):
    trainedImageWidth = 512
    meanSubtractionValue = 127
    targets = []
    image = None
    contours = None
    h, w = None, None
    resultData = {}
    objects = []

    foundClasses = []

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

    if isType == 'V' or isType == 'v':
        image = imagePath
        output = {}
    elif isType == 'I' or isType == 'i':
        try:
            image = np.array(Image.open(imagePath))
        except Exception as e:
            out = {"IS_CD": isCD, "CODE": "141", "MSG": str(e)}
            log.error(traceback.format_exc())
            _ = sendMsg('http://{}:{}/api/QI/report/stateCheck'.format(svrIp, svrPort), out)

        output = {"IS_CD": isCD, "IS_TYPE": isType, "FILE_INFO": None, "OBJECT_TYPE": "S"}

    w, h = image.shape[:2]
    # resize to max dimension of images from training dataset
    ratio = float(trainedImageWidth) / np.max([h, w])
    resizedImage = np.array(Image.fromarray(image.astype('uint8')).resize((int(ratio * h), int(ratio * w))))
    resizedImage = (resizedImage / meanSubtractionValue) - 1

    # pad array to square image to match training images
    padX = int(trainedImageWidth - resizedImage.shape[0])
    padY = int(trainedImageWidth - resizedImage.shape[1])
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
    for i in range(len(idx)):
        thresh = np.zeros((imgTmpHR, imgTmpWR, 3), np.uint8)
        classCnt = 0
        classInfo = {"CLASS_CD": printClassCd[idx[i]], "CLASS_NAME": printClassName[idx[i]], "COLOR": printColor[idx[i]],
                     "DP_LABEL": printdpLabel[idx[i]], "LOCATION": printLocation[idx[i]], "ACCURACY": None, "CLASS_CNT": None}

        for yy in range(imgTmpHR):
            for xx in range(imgTmpWR):
                if imgTmp[yy, xx, 0] == colorMapTmp[idx[i]][0] and\
                   imgTmp[yy, xx, 1] == colorMapTmp[idx[i]][1] and\
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
        # print(thresholdValue)
        ret, thresh = cv2.threshold(thresh, thresholdValue, 255, cv2.THRESH_BINARY)
        # cv2.imshow("thresh", thresh)
        # cv2.waitKey(0)
        contours, hierachy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            contourX = []
            contourY = []
            contourPt = []
            # if cv2.contourArea(contour) <= 10: continue

            x1, y1, x2, y2 = cv2.boundingRect(contour)
            boxSize = math.sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2))

            for pt in contour:
                contourX.append(float(pt[0][0]))
                contourY.append(float(pt[0][1]))

            for j in range(len(contourX)):
                contourPt.append({"X": contourX[j], "Y": contourY[j]})

            resultData = {"CLASS_CD": printClassCd[idx[i]], "CLASS_NAME": printClassName[idx[i]], "COLOR": printColor[idx[i]],
                          "DP_LABEL": printdpLabel[idx[i]], "LOCATION": printLocation[idx[i]], "ACCURACY": None,
                          "POSITION": contourPt, "SEQ": seq, "VALUE": boxSize}

            seq += 1
            classCnt += 1
            objects.append(resultData)

        classInfo["CLASS_CNT"] = classCnt
        foundClasses.append(classInfo)

    # saveObj["OBJECTS"] = objects
    output["FOUND_CLASSES"] = foundClasses
    output["OBJECTS"] = objects
    # return saveObj, output
    return output


# unet, pspnet 개발 필요
def unet():
    return 0


def pspnet():
    return 0


def detection(objectType, mdlPath, dataPath, target, resultPath, isCD, isType):
    resultData = None
    # model path
    modelFilePath, classPath = getModelFilePath(mdlPath)
    targetLen = len(target)

    output = []
    with open(classPath, 'r') as f:
        numClasses = (len(f.readlines()))
        classesTmp = [c.strip() for c in open(classPath).readlines()]
        classes = []

        for tmp in classesTmp:
            tmp = tmp.split(",")
            classes.append(tmp[1])

    # model load
    model, modelType = modelLoad(isCD, modelFilePath, numClasses)
    print(modelType)

    # input data is Video
    if isType == 'V' or isType == 'v':
        # fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        serverOutput = []
        for tmp in dataPath:
            videoPath = tmp["FILE_PATH"]
            print(videoPath)
            fileSEQ = tmp["FILE_SEQ"]
            try:
                vc = cv2.VideoCapture(videoPath)
            except Exception as e:
                out = {"IS_CD": isCD, "CODE": "142", "MSG": str(e)}
                log.error(traceback.format_exc())
                _ = sendMsg('http://{}:{}/api/QI/report/stateCheck'.format(svrIp, svrPort), out)

            if vc.isOpened():
                # fps = vc.get(cv2.CAP_PROP_FPS)
                # width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
                # height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
                # out = cv2.VideoWriter(os.path.join(resultPath, fileName), fcc, fps, (width, height))
                frameCount = 0
                frames = []
                totalFNumber = vc.get(cv2.CAP_PROP_FRAME_COUNT)
                fps = vc.get(cv2.CAP_PROP_FPS)
                saveDataName = os.path.join(resultPath, "{}.dat".format(fileSEQ))
                fileInfo = {"FILE_SEQ": fileSEQ, "FILE_PATH": videoPath, "RESULT_PATH": saveDataName,
                            "TOTAL_FRAME_NUMBER": totalFNumber, "FPS": fps, "TOTAL_FRAME_TIME": None}

                classCnt = [0 for _ in(range(targetLen))]
                avgAcc = [0 for _ in(range(targetLen))]

                foundClasses = [0 for _ in(range(targetLen))]

                while True:
                    ret, frame = vc.read()
                    if frame is None:
                        break

                    if ret:
                        if objectType == "D" or objectType == "d":
                            resultData = yolov3(model, frame, classes, target, resultPath, isCD, isType, fileSEQ)

                        elif objectType == "S" or objectType == "s":
                            resultData = deeplabModel(model, frame, classes, target, resultPath, isCD, isType, fileSEQ)

                    resultData["FRAME_NUMBER"] = frameCount
                    resultData["FRAME_TIME"] = vc.get(cv2.CAP_PROP_POS_MSEC)

                    foundClassTmp = resultData["FOUND_CLASSES"]
                    del resultData["FOUND_CLASSES"]

                    for i, foundClass in enumerate(foundClassTmp):
                        classCnt[i] += foundClass["CLASS_CNT"]
                        foundClass["CLASS_CNT"] = classCnt[i]
                        if objectType == "S" or objectType == "s":
                            foundClass["ACCURACY"] = None

                        elif objectType == "D" or objectType == "d":
                            avgAcc[i] += foundClass["ACCURACY"]
                            foundClass["ACCURACY"] = avgAcc[i]

                        foundClasses.insert(i, foundClass)

                    frameCount += 1
                    frames.append(resultData)

                for i, foundClass in enumerate(foundClassTmp):
                    if objectType == "S" or objectType == "s":
                        foundClass["ACCURACY"] = None

                    elif objectType == "D" or objectType == "d":
                        foundClass["ACCURACY"] /= foundClass["CLASS_CNT"]

                fileInfo["TOTAL_FRAME_TIME"] = frameCount
                output = {"IS_CD": isCD, "IS_TYPE": isType, "FILE_INFO": fileInfo, "OBJECT_TYPE": objectType, "FRAMES": None}

                output["FOUND_CLASSES"] = foundClassTmp
                output["FRAMES"] = frames
                serverOutput.append(output)
                with open(saveDataName, "w") as saveJson:
                    json.dump(output, saveJson)
                vc.release()
        return serverOutput

    # input data is Image
    elif isType == 'I' or isType == 'i':
        for tmp in dataPath:
            print(tmp)
            imagePath = tmp["FILE_PATH"]
            fileSEQ = tmp["FILE_SEQ"]
            saveDataName = os.path.join(resultPath, "{}.dat".format(fileSEQ))

            if modelType == "yolo":
                resultData = yolov3(model, imagePath, classes, target, resultPath, isCD, isType, fileSEQ)

            elif modelType == 'deeplab':
                resultData = deeplabModel(model, imagePath, classes, target, resultPath, isCD, isType, fileSEQ)

            resultData["FILE_INFO"] = {"FILE_SEQ": fileSEQ, "FILE_PATH": imagePath, "RESULT_PATH": saveDataName}

            output.append(resultData)
            with open(saveDataName, "w") as saveJson:
                json.dump(resultData, saveJson)

        return output

    else:
        print("obj type error")
        return 0


if __name__ == "__main__":
    try:
        # input: [{"DATA_TYPE":"I/V", "OBJECT_TYPE": "D/S", "IS_CD": "IS000001", "MDL_PATH":"MODEL_PATH","STATE":"START/STOP",
        #          "TARGET_CLASS": "tvmonitor"}]
        # data = json.loads(sys.argv[1])
        string = sys.argv[1]
        gpuIdx = sys.argv[2]
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
                    log.error(e)

        objectType = data["OBJECT_TYPE"]
        isCD = data["IS_CD"]
        mdlPath = data["MDL_PATH"]
        target = data["TARGET_CLASS"]
        dataPath = data["DATA_PATH"]
        isCD = data["IS_CD"]
        isType = data["IS_TYPE"]
        print(os.getpid())
        output = []

        # objectType = "S"
        # mdlPath = "/Users/upload/InputSources/27"
        resultPath = os.path.join(mdlPath, "data", "result")
        isMakeDir = makeDir(resultPath)
        output = detection(objectType, mdlPath, dataPath, target, resultPath, isCD, isType)

        # print(json.dumps(output))
        res = requests.post(sendServer, headers=headers, data=json.dumps(output))

    except Exception as e:
        out = {"IS_CD": isCD, "CODE": "140", "MSG": str(e)}
        log.error(traceback.format_exc())
        res = requests.post(sendServer2, headers=headers, data=json.dumps(out))
