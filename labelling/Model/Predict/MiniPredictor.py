# -*- coding:utf-8 -*-
'''
minipredictor 스크립트
'''
import os
import sys
import json
import cv2
from PIL import Image
import numpy as np
import traceback
import random

import tensorflow as tf

# Model 폴더 찾기 위한 방법
basePath = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir, "../")))
# Model Path 등록
sys.path.append(basePath)

from Common.Logger.Logger import logger, getConfig
from Common.Utils.Utils import getClasses, hex2rgb, label2ColorImage, createMask, transform_images
from Dataset.ImageDataSet import predictDataset
from Common.Process.Process import prcSendData, prcErrorData

_, _, _, _, _, aiPath = getConfig()

log = logger("log")


def getThreshImg(newLabelsCalH, newLabelsCalW, newLabelsCal, colorMap, thresh):
    for yy in range(newLabelsCalH):
        for xx in range(newLabelsCalW):
            if newLabelsCal[yy, xx, 0] == colorMap[0][2] and\
               newLabelsCal[yy, xx, 1] == colorMap[0][1] and\
               newLabelsCal[yy, xx, 2] == colorMap[0][0]:

                thresh.itemset(yy, xx, 0, 255)
                thresh.itemset(yy, xx, 1, 255)
                thresh.itemset(yy, xx, 2, 255)
            else:
                thresh.itemset(yy, xx, 0, 0)
                thresh.itemset(yy, xx, 1, 0)
                thresh.itemset(yy, xx, 2, 0)
    return thresh


def getSegOutput(contours, newLabelsCalW, newLabelsCalH, rect, w, h,
                 imgPath, datasetCd, dataCd, CLASS_DB_NM, outputColor, annoData, frameCnt):

    contourRatio1 = (newLabelsCalH * newLabelsCalW) * 0.01
    contourRatio2 = (newLabelsCalH * newLabelsCalW) * 0.95
  
    for contour in contours:
        if cv2.contourArea(contour) <= contourRatio1:
            continue
        if cv2.contourArea(contour) >= contourRatio2:
            continue

        contourX = []
        contourY = []
        contourPt = []
        for pt in contour:
            contourX.append(float(pt[0][0] / newLabelsCalW))
            contourY.append(float(pt[0][1] / newLabelsCalH))

        for i in range(0, len(contourX)):
            if rect is not None:
                contourPt.append({
                    "X": (contourX[i] * w) + rect[0],
                    "Y": (contourY[i] * h) + rect[1]
                })
            else:
                contourPt.append({
                    "X": contourX[i] * w,
                    "Y": contourY[i] * h
                })

        if len(contourPt) > 0:
            annoInfo = {"FRAME_NUMBER": frameCnt, "IMAGE_PATH": imgPath, "DATASET_CD": datasetCd, "DATA_CD": dataCd,
                        "CLASS_DB_NM": CLASS_DB_NM, "POSITION": contourPt, "ACCURACY": None,
                        "COLOR": outputColor, "CURSOR": "isPolygon"}

        if len(annoInfo) > 0:
            annoData.append(annoInfo)
    return annoData


def miniPredictor(data, model, modelName, inputShape):
    try:
        log.debug("======================================")
        log.debug(json.dumps(data))
        log.debug("======================================")
        # get params
        DATA_TYPE = data["DATA_TYPE"]
        OBJECT_TYPE = data["OBJECT_TYPE"]
        IMAGES = data["IMAGES"]
        AI_CD = data["AI_CD"]

        IS_TEST = data["IS_TEST"]
        # IS_TEST = False
        # get Classes
        mdlPath = data["MDL_PATH"]
        classes = getClasses(AI_CD, OBJECT_TYPE, mdlPath)
        lenClasses = len(classes)
        # log.debug("========================================")
        # log.debug(classes)
        # log.debug("========================================")

        # log.debug("===========================")
        # log.debug(inputShape)
        # log.debug("===========================")

        annoData = []
        annoInfo = {}
        output = []
        x1, x2, y1, y2 = None, None, None, None

        if DATA_TYPE == "I":
            imgCnt = 0
            imgTotal = len(IMAGES)
            for imgInfo in IMAGES:
                CLASS_DB_NM = data["CLASS_DB_NM"].lower() if "CLASS_DB_NM" in data else None
                imgCnt += 1
                imgPath = imgInfo["IMAGE_PATH"]
                datasetCd = imgInfo["DATASET_CD"]
                dataCd = imgInfo["DATA_CD"]
                colors = imgInfo["COLOR"] if not IS_TEST else [
                    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(lenClasses)]

                if imgInfo["RECT"] is not None:
                    tmp = imgInfo["RECT"]
                    tmp = list(tmp)
                    rect = (int(tmp[0]["X"]), int(tmp[0]["Y"]), int(tmp[1]["X"]), int(tmp[1]["Y"]))
                else:
                    rect = None

                img, oriH, oriW, resizeH, resizeW = predictDataset(DATA_TYPE, modelName, OBJECT_TYPE, imgPath, inputShape)
                if OBJECT_TYPE == "C":
                    classes.sort()
                    log.info("[{}] Predict ({}/{})".format(AI_CD, imgCnt, imgTotal))
                    predict = model.predict_on_batch(img)
                    log.debug(predict)
                    maxIdx = np.argmax(predict[0])
                    classDbNm = classes[maxIdx]
                    classAcc = predict[0][maxIdx]
                    annoInfo = {"IMAGE_PATH": imgPath, "DATASET_CD": datasetCd, "DATA_CD": dataCd,
                                "CLASS_DB_NM": classDbNm, "POSITION": [], "ACCURACY": float(classAcc), "COLOR": colors[0]}
                    annoData.append(annoInfo)

                elif OBJECT_TYPE == "D":
                    outCLASS_DB_NM = None
                    if modelName == "YOLOV3" or modelName == "YOLOV4":
                        boxes, scores, detectClasses, nums = model(img)
                        annoInfo = {}
                        log.debug("{} / {}".format(int(nums[0]), scores[0]))
                        for i in range(nums[0]):
                            if CLASS_DB_NM is None:
                                if np.array(scores[0][i]) > 0.01:
                                    classTmp = classes[int(detectClasses[0][i])].split(",")
                                    # classTmp = classTmp[len(classTmp) - 1]
                                    CLASS_DB_NM = classTmp
                                    outCLASS_DB_NM = CLASS_DB_NM
                                    color = colors[int(detectClasses[0][i])] if type(colors) is not str else colors
                                    x1 = float(np.array(boxes[0][i][0]) * oriW)
                                    y1 = float(np.array(boxes[0][i][1]) * oriH)
                                    x2 = float(np.array(boxes[0][i][2]) * oriW)
                                    y2 = float(np.array(boxes[0][i][3]) * oriH)
                                    acc = np.array(scores[0][i])
                                    CLASS_DB_NM = None

                            elif CLASS_DB_NM is not None:
                                classTmp = classes[int(detectClasses[0][i])].split(",")
                                classTmp = classTmp[len(classTmp) - 1]
                                if classTmp == CLASS_DB_NM:
                                    if np.array(scores[0][i]) > 0.01:
                                        outCLASS_DB_NM = CLASS_DB_NM
                                        color = colors[int(detectClasses[0][i])] if type(colors) is not str else colors
                                        x1 = float(np.array(boxes[0][i][0]) * oriW)
                                        y1 = float(np.array(boxes[0][i][1]) * oriH)
                                        x2 = float(np.array(boxes[0][i][2]) * oriW)
                                        y2 = float(np.array(boxes[0][i][3]) * oriH)
                                        acc = np.array(scores[0][i])

                            if x1 is not None:
                                annoInfo = {
                                    "IMAGE_PATH": imgPath,
                                    "DATASET_CD": datasetCd,
                                    "DATA_CD": dataCd,
                                    "CLASS_DB_NM": outCLASS_DB_NM,
                                    "POSITION": [{"X": x1, "Y": y1}, {"X": x2, "Y": y2}],
                                    "ACCURACY": float(acc),
                                    "COLOR": color
                                }

                            if len(annoInfo) > 0:
                                annoData.append(annoInfo)
                    
                    elif modelName == "EFFICIENTDET":
                        outCLASS_DB_NM = None
                        annoInfo = {}
                        boxes, scores, detectClass, valid_len = model.f(img)

                        for i in range(len(img)):
                            length = valid_len[i]
                            if length == 0:
                                annoInfo = {}
                                break
                            box = boxes[i].numpy()[:length]
                            classIdx = detectClass[i].numpy().astype(np.int)[:length]
                            score = scores[i].numpy()[:length]
                            if CLASS_DB_NM is None:
                                if score[0] > 0.8:
                                    classTmp = classes[int(classIdx[0])].split(",")
                                    CLASS_DB_NM = classTmp
                                    outCLASS_DB_NM = CLASS_DB_NM
                                    color = colors[int(classIdx[0])] if type(colors) is not str else colors

                                    x1 = float(np.array(box[0][0]))
                                    y1 = float(np.array(box[0][1]))
                                    x2 = float(np.array(box[0][2]))
                                    y2 = float(np.array(box[0][3]))
                                    acc = np.array(score[0])

                                    CLASS_DB_NM = None

                            elif CLASS_DB_NM is not None:
                                classTmp = classes[int(classIdx[0])].split(",")
                                classTmp = classTmp[len(classTmp) - 1]
                                if CLASS_DB_NM == classTmp:
                                    CLASS_DB_NM = classTmp
                                    color = colors[int(classIdx[0])] if type(colors) is not str else colors
                                    x1 = float(np.array(box[0][0]))
                                    y1 = float(np.array(box[0][1]))
                                    x2 = float(np.array(box[0][2]))
                                    y2 = float(np.array(box[0][3]))
                                    acc = np.array(score[0])

                            if x1 is not None:
                                annoInfo = {
                                    "IMAGE_PATH": imgPath,
                                    "DATASET_CD": datasetCd,
                                    "DATA_CD": dataCd,
                                    "CLASS_DB_NM": outCLASS_DB_NM,
                                    "POSITION": [{"X": x1, "Y": y1}, {"X": x2, "Y": y2}],
                                    "ACCURACY": float(acc),
                                    "COLOR": color
                                }

                            if len(annoInfo) > 0:
                                annoData.append(annoInfo)

                elif OBJECT_TYPE == "S":
                    labels = None

                    if modelName == "DEEP-LAB":
                        trainedImageWidth = 512
                        meanSubtractionValue = 127.5

                        if rect:
                            img = img[rect[1]:rect[3], rect[0]:rect[2]]

                        h, w = oriH, oriW
                        resizedImage = np.array(Image.fromarray(img.astype('uint8')).resize((trainedImageWidth, trainedImageWidth)))
                        resizedImage = (resizedImage / meanSubtractionValue) - 1

                        # pad array to square image to match training images

                        pad_x = int(trainedImageWidth - resizedImage.shape[0])
                        pad_y = int(trainedImageWidth - resizedImage.shape[1])
                        resized_image = np.pad(resizedImage, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')

                        # run prediction
                        res = model.predict(np.expand_dims(resized_image, 0))
                        labels = np.argmax(res.squeeze(), -1)

                        # remove padding and resize back to original image
                        if pad_x > 0:
                            labels = labels[:-pad_x]
                        if pad_y > 0:
                            labels = labels[:, :-pad_y]

                    elif modelName == "EFFICIENTDET-SEG":
                        if rect:
                            img = img[rect[1]:rect[3], rect[0]:rect[2]]

                        h, w = oriH, oriW
                        log.debug("H : {}, W : {}".format(h, w))

                        img = tf.expand_dims(img, 0)
                        img = transform_images(img, 512)

                        labels = createMask(model(img, False))
                        labels = labels.numpy()[0]

                    else:
                        if rect:
                            img = img[rect[1]:rect[3], rect[0]:rect[2]]

                        h, w = oriH, oriW

                        res = model.predict(img)
                        labels = tf.argmax(res[0], -1).numpy()

                    labelsTmp = labels.ravel()
                    labelsTmp = list(set(labelsTmp))

                    if modelName == "U-NET":
                        newLabelTmp = []
                        for tmp in labelsTmp:
                            tmp = tmp - 1
                            newLabelTmp.append(tmp)
                        labelsTmp = []
                        labelsTmp = newLabelTmp

                    labelsTmp.remove(0)

                    lh, lw = labels.shape
                    if IS_TEST:
                        labels, colorMap = label2ColorImage(labels)
                        newLabelsCal = np.array(Image.fromarray(labels.astype('uint8')))
                        newLabelsCal = cv2.cvtColor(newLabelsCal, cv2.COLOR_RGB2BGR)
                        newLabelsCalH, newLabelsCalW, _ = newLabelsCal.shape

                        for idx, colorTmp in enumerate(colorMap[labelsTmp]):
                            CLASS_DB_NM = classes[labelsTmp[idx]]
                            thresh = np.zeros((newLabelsCalH, newLabelsCalW, 3), np.uint8)
                            outputColor = colors[idx]
                            thresh = getThreshImg(newLabelsCalH, newLabelsCalW, newLabelsCal, colorMap, thresh)

                            thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
                            contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                            annoData = getSegOutput(
                                contours,
                                newLabelsCalW,
                                newLabelsCalH,
                                rect,
                                w, h,
                                imgPath,
                                datasetCd,
                                dataCd,
                                CLASS_DB_NM,
                                outputColor,
                                annoData,
                                0
                            )

                    else:

                        cvtColor = hex2rgb(colors)
                        cvtColor = np.asarray(cvtColor)
                        colorMap = np.empty((0, 3), int)
                        colorMap = np.append(colorMap, np.array([cvtColor]), axis=0)
                        colorMap = np.append(colorMap, np.array([[0, 0, 0]]), axis=0)
                        if CLASS_DB_NM:
                            if CLASS_DB_NM in classes:
                                idx = classes.index(CLASS_DB_NM)
                                for yy in range(lh):
                                    for xx in range(lw):
                                        if labels[yy, xx] == idx:
                                            labels[yy, xx] = 0
                                        else:
                                            labels[yy, xx] = 1

                        labels = colorMap[labels]
                        newLabelsCal = np.array(Image.fromarray(labels.astype('uint8')))
                        newLabelsCal = cv2.cvtColor(newLabelsCal, cv2.COLOR_RGB2BGR)
                        newLabelsCalH, newLabelsCalW, _ = newLabelsCal.shape
                        thresh = np.zeros((newLabelsCalH, newLabelsCalW, 3), np.uint8)

                        thresh = getThreshImg(newLabelsCalH, newLabelsCalW, newLabelsCal, colorMap, thresh)
                        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
                        contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        annoData = getSegOutput(
                            contours,
                            newLabelsCalW,
                            newLabelsCalH,
                            rect,
                            w, h,
                            imgPath,
                            datasetCd,
                            dataCd,
                            CLASS_DB_NM,
                            colors,
                            annoData,
                            0
                        )

            outImgPath = imgPath

            if IS_TEST:
                outImgPath = os.path.join(aiPath, AI_CD)
                outImgPath = os.path.join(outImgPath, "test.dat")

            filename, fileExtension = os.path.splitext(outImgPath)
            saveJsonPath = outImgPath.replace(fileExtension, ".dat")

            with open(saveJsonPath, "w", encoding='euc-kr') as f:
                json.dump({"POLYGON_DATA": annoData}, f)

            outputData = {"AI_CD": AI_CD, "DATA_TYPE": DATA_TYPE,
                          "OBJECT_TYPE": OBJECT_TYPE, "ANNO_DATA": annoData, "IMAGE_PATH": imgPath, "TEST_PATH": outImgPath}
            output.append(outputData)

            log.debug("======================================")
            log.debug(json.dumps(output))
            log.debug("======================================")

            log.info("Predict Done.")
            return output

        elif DATA_TYPE == "V":
            for imgInfo in IMAGES:
                CLASS_DB_NM = data["CLASS_DB_NM"].lower() if "CLASS_DB_NM" in data else None
                imgPath = imgInfo["IMAGE_PATH"]
                datasetCd = imgInfo["DATASET_CD"]
                startFrame = imgInfo["START_FRAME"]
                if startFrame is None:
                    startFrame = 0

                endFrame = imgInfo["END_FRAME"]
                dataCd = imgInfo["DATA_CD"]
                colors = imgInfo["COLOR"] if not IS_TEST else [
                    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(lenClasses)]

                color = colors
                if imgInfo["RECT"] is not None:
                    tmp = imgInfo["RECT"]
                    tmp = list(tmp)
                    rect = (int(tmp[0]["X"]), int(tmp[0]["Y"]), int(tmp[1]["X"]), int(tmp[1]["Y"]))
                else:
                    rect = None
                vc = cv2.VideoCapture(imgPath)
                if not vc.isOpened():
                    raise Exception

                vc.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
                frameCnt = startFrame

                while True:
                    ret, img = vc.read()
                    img, oriH, oriW, resizeH, resizeW = predictDataset(DATA_TYPE, modelName, OBJECT_TYPE, img, inputShape)
                    if OBJECT_TYPE == "D":
                        outCLASS_DB_NM = None
                        if modelName == "YOLOV3" or modelName == "YOLOV4":
                            boxes, scores, detectClasses, nums = model.predict(img)
                            annoInfo = {}
                            for i in range(nums[0]):
                                if CLASS_DB_NM is None:
                                    if np.array(scores[0][i]) > 0.01:
                                        classTmp = classes[int(detectClasses[0][i])].split(",")
                                        # classTmp = classTmp[len(classTmp) - 1]
                                        CLASS_DB_NM = classTmp
                                        outCLASS_DB_NM = CLASS_DB_NM
                                        color = colors[int(detectClasses[0][i])] if type(colors) is not str else colors
                                        x1 = float(np.array(boxes[0][i][0]) * oriW)
                                        y1 = float(np.array(boxes[0][i][1]) * oriH)
                                        x2 = float(np.array(boxes[0][i][2]) * oriW)
                                        y2 = float(np.array(boxes[0][i][3]) * oriH)
                                        acc = np.array(scores[0][i])
                                        CLASS_DB_NM = None

                                elif CLASS_DB_NM is not None:
                                    classTmp = classes[int(detectClasses[0][i])].split(",")
                                    classTmp = classTmp[len(classTmp) - 1]
                                    if classTmp == CLASS_DB_NM:
                                        if np.array(scores[0][i]) > 0.01:
                                            # classTmp = classTmp[len(classTmp) - 1]
                                            outCLASS_DB_NM = CLASS_DB_NM
                                            color = colors[int(detectClasses[0][i])] if type(colors) is not str else colors
                                            x1 = float(np.array(boxes[0][i][0]) * oriW)
                                            y1 = float(np.array(boxes[0][i][1]) * oriH)
                                            x2 = float(np.array(boxes[0][i][2]) * oriW)
                                            y2 = float(np.array(boxes[0][i][3]) * oriH)
                                            acc = np.array(scores[0][i])

                                if x1 is not None:
                                    annoInfo = {
                                        "IMAGE_PATH": imgPath,
                                        "DATASET_CD": datasetCd,
                                        "DATA_CD": dataCd,
                                        "CLASS_DB_NM": outCLASS_DB_NM,
                                        "POSITION": [{"X": x1, "Y": y1}, {"X": x2, "Y": y2}],
                                        "ACCURACY": float(acc),
                                        "COLOR": color,
                                        "FRAME_NUMBER": frameCnt
                                    }

                                if len(annoInfo) > 0:
                                    annoData.append(annoInfo)

                        elif modelName == "EFFICIENTDET":
                            outCLASS_DB_NM = None
                            annoInfo = {}
                            boxes, scores, detectClass, valid_len = model.f(img)

                            for i in range(len(img)):
                                length = valid_len[i]
                                if length == 0:
                                    annoInfo = {}
                                    break
                                box = boxes[i].numpy()[:length]
                                classIdx = detectClass[i].numpy().astype(np.int)[:length]
                                score = scores[i].numpy()[:length]
                                if CLASS_DB_NM is None:
                                    if score[0] > 0.01:
                                        classTmp = classes[int(classIdx[0])].split(",")
                                        CLASS_DB_NM = classTmp
                                        outCLASS_DB_NM = CLASS_DB_NM
                                        color = colors[int(classIdx[0])] if type(colors) is not str else colors
                                        x1 = float(np.array(box[0][0]))
                                        y1 = float(np.array(box[0][1]))
                                        x2 = float(np.array(box[0][2]))
                                        y2 = float(np.array(box[0][3]))
                                        acc = np.array(score[0])

                                        CLASS_DB_NM = None

                                elif CLASS_DB_NM is not None:
                                    classTmp = classes[int(classIdx[0])].split(",")
                                    classTmp = classTmp[len(classTmp) - 1]
                                    if CLASS_DB_NM == classTmp:
                                        CLASS_DB_NM = classTmp
                                        color = colors[int(classIdx[0])] if type(colors) is not str else colors
                                        x1 = float(np.array(box[0][0]))
                                        y1 = float(np.array(box[0][1]))
                                        x2 = float(np.array(box[0][2]))
                                        y2 = float(np.array(box[0][3]))
                                        acc = np.array(score[0])

                                if x1 is not None:
                                    annoInfo = {
                                        "FRAME_NUMBER": frameCnt,
                                        "IMAGE_PATH": imgPath,
                                        "DATASET_CD": datasetCd,
                                        "DATA_CD": dataCd,
                                        "CLASS_DB_NM": outCLASS_DB_NM,
                                        "POSITION": [{"X": x1, "Y": y1}, {"X": x2, "Y": y2}],
                                        "ACCURACY": float(acc),
                                        "COLOR": color,
                                        "FRAME_NUMBER": frameCnt
                                    }

                                if len(annoInfo) > 0:
                                    annoData.append(annoInfo)

                    elif OBJECT_TYPE == 'S':
                        annoInfo = {}
                        if modelName == "DEEP-LAB":
                            trainedImageWidth = 512
                            meanSubtractionValue = 127.5

                            if rect:
                                img = img[rect[1]:rect[3], rect[0]:rect[2]]

                            h, w = img.shape[:2]
                            log.debug("H : {}, W : {}".format(h, w))

                            resizedImage = np.array(Image.fromarray(img.astype('uint8')).resize((trainedImageWidth, trainedImageWidth)))
                            resizedImage = (resizedImage / meanSubtractionValue) - 1

                            # pad array to square image to match training images

                            pad_x = int(trainedImageWidth - resizedImage.shape[0])
                            pad_y = int(trainedImageWidth - resizedImage.shape[1])
                            resized_image = np.pad(resizedImage, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')

                            # run prediction
                            res = model.predict(np.expand_dims(resized_image, 0))
                            labels = np.argmax(res.squeeze(), -1)

                            # remove padding and resize back to original image
                            if pad_x > 0:
                                labels = labels[:-pad_x]
                            if pad_y > 0:
                                labels = labels[:, :-pad_y]

                        elif modelName == "EFFICIENTDET-SEG":
                            if rect:
                                img = img[rect[1]:rect[3], rect[0]:rect[2]]

                            h, w = img.shape[:2]
                            log.debug("H : {}, W : {}".format(h, w))

                            img = tf.expand_dims(img, 0)
                            img = transform_images(img, 512)

                            labels = createMask(model(img, False))
                            labels = labels.numpy()[0]

                        else:
                            if rect:
                                img = img[rect[1]:rect[3], rect[0]:rect[2]]

                            h, w = oriH, oriW

                            res = model.predict(img)
                            labels = tf.argmax(res[0], -1).numpy()

                        labelsTmp = labels.ravel()
                        labelsTmp = list(set(labelsTmp))

                        if modelName == "U-NET":
                            newLabelTmp = []
                            for tmp in labelsTmp:
                                tmp = tmp - 1
                                newLabelTmp.append(tmp)
                            labelsTmp = []
                            labelsTmp = newLabelTmp

                        labelsTmp.remove(0)
                        lh, lw = labels.shape

                        if IS_TEST:
                            labels, colorMap = label2ColorImage(labels)
                            newLabelsCal = np.array(Image.fromarray(labels.astype('uint8')))
                            newLabelsCal = cv2.cvtColor(newLabelsCal, cv2.COLOR_RGB2BGR)
                            newLabelsCalH, newLabelsCalW, _ = newLabelsCal.shape

                            for idx, colorTmp in enumerate(colorMap[labelsTmp]):
                                CLASS_DB_NM = classes[labelsTmp[idx]]
                                thresh = np.zeros((newLabelsCalH, newLabelsCalW, 3), np.uint8)
                                outputColor = color[idx]
                                thresh = getThreshImg(newLabelsCalH, newLabelsCalW, newLabelsCal, colorMap, thresh)

                                thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
                                contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                                annoData = getSegOutput(
                                    contours,
                                    newLabelsCalW,
                                    newLabelsCalH,
                                    rect,
                                    w, h,
                                    imgPath,
                                    datasetCd,
                                    dataCd,
                                    CLASS_DB_NM,
                                    outputColor,
                                    annoData,
                                    frameCnt
                                )

                        else:
                            cvtColor = hex2rgb(color)
                            cvtColor = np.asarray(cvtColor)
                            colorMap = np.empty((0, 3), int)
                            colorMap = np.append(colorMap, np.array([cvtColor]), axis=0)
                            colorMap = np.append(colorMap, np.array([[0, 0, 0]]), axis=0)
                            if CLASS_DB_NM:
                                if CLASS_DB_NM in classes:
                                    idx = classes.index(CLASS_DB_NM)
                                    for yy in range(lh):
                                        for xx in range(lw):
                                            if labels[yy, xx] == idx:
                                                labels[yy, xx] = 0
                                            else:
                                                labels[yy, xx] = 1

                            labels = colorMap[labels]
                            newLabelsCal = np.array(Image.fromarray(labels.astype('uint8')))
                            newLabelsCal = cv2.cvtColor(newLabelsCal, cv2.COLOR_RGB2BGR)
                            newLabelsCalH, newLabelsCalW, _ = newLabelsCal.shape
                            thresh = np.zeros((newLabelsCalH, newLabelsCalW, 3), np.uint8)

                            thresh = getThreshImg(newLabelsCalH, newLabelsCalW, newLabelsCal, colorMap, thresh)
                            thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
                            contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                            annoData = getSegOutput(
                                contours,
                                newLabelsCalW,
                                newLabelsCalH,
                                rect,
                                w, h,
                                imgPath,
                                datasetCd,
                                dataCd,
                                CLASS_DB_NM,
                                color,
                                annoData,
                                frameCnt
                            )
                    frameCnt += 1
                    log.info("framCNT : {}, START : {}, END : {}".format(frameCnt, startFrame, endFrame))
                    if frameCnt >= endFrame:
                        break

                else:
                    raise Exception
                
                outImgPath = imgPath

                if IS_TEST:
                    outImgPath = os.path.join(aiPath, AI_CD)
                    outImgPath = os.path.join(outImgPath, "test.dat")

                filename, fileExtension = os.path.splitext(outImgPath)
                saveJsonPath = outImgPath.replace(fileExtension, ".dat")

                with open(saveJsonPath, "w", encoding='euc-kr') as f:
                    json.dump({"POLYGON_DATA": annoData}, f)

                outputData = {
                    "AI_CD": AI_CD,
                    "DATA_TYPE": DATA_TYPE,
                    "OBJECT_TYPE": OBJECT_TYPE,
                    "ANNO_DATA": annoData,
                    "IMAGE_PATH": imgPath,
                    "TEST_PATH": outImgPath}

                output.append(outputData)
                log.debug("======================================")
                log.debug(json.dumps(output))
                log.debug("======================================")

                log.info("Predict Done.")
                return output

        else:
            raise Exception

    except Exception as e:
        log.error(traceback.format_exc())
        prcErrorData(__file__, e)