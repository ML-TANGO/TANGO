# -*- coding:utf-8 -*-
'''
autoLabeling 스크립트
'''
import os
import sys
import cv2
from PIL import Image
import numpy as np
import random
import json
import traceback
import tensorflow as tf

# Model 폴더 찾기 위한 방법
basePath = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# Model Path 등록
sys.path.append(basePath)

from Common.Logger.Logger import logger
from Common.Utils.Utils import label2ColorImage, getClasses, createMask, transform_images
from Dataset.ImageDataSet import predictDataset
from Common.Process.Process import prcErrorData

from Output.PredictOutput import operator

log = logger("log")


def saveJson(imgPath, labels, totalFrame):
    filename, fileExtension = os.path.splitext(imgPath)
    saveJsonPath = imgPath.replace(fileExtension, ".dat")
    newLabels = []
    if totalFrame == 0:
        newLabels = labels
    else:
        for idx in range(totalFrame):
            line = []
            for labelInfo in labels:
                if labelInfo is None:
                    continue
                frameNumber = labelInfo["FRAME_NUMBER"] if "FRAME_NUMBER" in labelInfo else None
                if idx == frameNumber:
                    line.append(labelInfo)
                else:
                    continue
            newLabels.append(line)
    with open(saveJsonPath, "w") as f:
        output = {"POLYGON_DATA": newLabels}
        json.dump(output, f)


def setDectOutput(dataType, className, acc, cursor, position, color, frameCnt):
    if dataType == 'I':
        result = {
            "label": className,
            "ACCURACY": float(acc),
            "CURSOR": cursor,
            "POSITION": position,
            "COLOR": color
        }
    else:
        result = {
            "FRAME_NUMBER": frameCnt,
            "label": className,
            "ACCURACY": float(acc),
            "CURSOR": cursor,
            "POSITION": position,
            "COLOR": color
        }
    return result


def setSegOutput(dataType, className, cursor, contourPt, color, frameCnt):
    if dataType == 'I':
        result = {
            "label": className,
            "ACCURACY": None,
            "POSITION": contourPt,
            "CURSOR": cursor,
            "COLOR": color
        }
    else:
        result = {
            "FRAME_NUMBER": frameCnt,
            "label": className,
            "ACCURACY": None,
            "POSITION": contourPt,
            "CURSOR": cursor,
            "COLOR": color
        }
    return result


def classification(tmp, labels, classes, colors, targetClass, objectType, frameCnt):
    classes.sort()
    if targetClass is None:
        for i in range(0, len(tmp)):
            if i > 10:
                break
            if float(tmp[i][1]) > 0.0:
                labels.append({
                    "label": classes[tmp[i][0]],
                    "ACCURACY": float(tmp[i][1]),
                    "COLOR": colors[0],
                    "FRAME_NUMBER": frameCnt
                })
    else:
        label = {}
        for targetData in targetClass:
            targetClassName = targetData["CLASS_NAME"]
            accScope = targetData["ACC_SCOPE"]
            for i in range(0, len(tmp)):
                if targetClassName == classes[tmp[i][0]]:
                    accuracy = float(tmp[i][1])
                    label = operator(
                        accuracy,
                        accScope,
                        colors[0],
                        objectType,
                        targetClassName,
                        imgW=None, imgH=None, boxes=None
                    )
                if label is not None:
                    label["FRAME_NUMBER"] = frameCnt
                    break
            labels.append(label)
    return labels


def detection(dataType, model, classes, img, oriH, oriW, colors, frameCnt, labels, autoAcc, modelName, targetClass, objectType):
    if modelName == "YOLOV3" or modelName == "YOLOV4":
        label = {}
        boxes, scores, detectClasses, nums = model.predict(img)
        if targetClass is None:
            for i in range(nums[0]):
                if np.array(scores[0][i]) > autoAcc:
                    className = classes[int(detectClasses[0][i])]
                    acc = np.array(scores[0][i])
                    x1 = float(np.array(boxes[0][i][0]) * oriW)
                    y1 = float(np.array(boxes[0][i][1]) * oriH)
                    x2 = float(np.array(boxes[0][i][2]) * oriW)
                    y2 = float(np.array(boxes[0][i][3]) * oriH)
                    cursor = "isRect"
                    position = [{"X": x1, "Y": y1}, {"X": x2, "Y": y2}]
                    color = colors[int(detectClasses[0][i])]
                    result = setDectOutput(dataType, className, acc, cursor, position, color, frameCnt)
                    labels.append(result)
        else:
            for targetData in targetClass:
                targetClassName = targetData["CLASS_NAME"]
                accScope = targetData["ACC_SCOPE"]
                color = targetData["COLOR"]

                for i in range(nums[0]):
                    className = classes[int(detectClasses[0][i])]
                    if targetClassName == className:
                        accuracy = np.array(scores[0][i])
                        label = operator(
                            accuracy,
                            accScope,
                            color,
                            objectType,
                            className,
                            imgW=oriW,
                            imgH=oriH,
                            boxes=np.array(boxes[0][i]))

                        label["FRAME_NUMBER"] = frameCnt
                        labels.append(label)

    elif modelName == "EFFICIENTDET":
        label = {}
        boxes, scores, detectClass, valid_len = model.f(img)
        if targetClass is None:
            for i in range(len(img)):
                length = valid_len[i]
                if length == 0:
                    label = {}
                    labels.append(label)
                    break

                box = boxes[i].numpy()[:length]
                classIdx = detectClass[i].numpy().astype(np.int)[:length]
                score = scores[i].numpy()[:length]

                className = classes[int(classIdx[0])]
                acc = score[0]
                if acc > autoAcc:
                    x1 = float(np.array(box[0][0]))
                    y1 = float(np.array(box[0][1]))
                    x2 = float(np.array(box[0][2]))
                    y2 = float(np.array(box[0][3]))

                    cursor = "isRect"
                    position = [{"X": x1, "Y": y1}, {"X": x2, "Y": y2}]
                    color = colors[int(classIdx[0])]
                    result = setDectOutput(dataType, className, acc, cursor, position, color, frameCnt)
                    labels.append(result)

        else:
            for targetData in targetClass:
                targetClassName = targetData["CLASS_NAME"]
                accScope = targetData["ACC_SCOPE"]
                color = targetData["COLOR"]
                for i in range(len(img)):
                    length = valid_len[i]
                    if length == 0:
                        label = {}
                        labels.append(label)
                        break
                    box = boxes[i].numpy()[:length]
                    classIdx = detectClass[i].numpy().astype(np.int)[:length]
                    score = scores[i].numpy()[:length]

                    className = classes[int(classIdx[0])]
                    if targetClassName == className:
                        accuracy = score[0]
                        label = operator(
                            accuracy,
                            accScope,
                            color,
                            objectType,
                            className,
                            imgW=1,
                            imgH=1,
                            boxes=np.array(box[0]))

                        label["FRAME_NUMBER"] = frameCnt
                        labels.append(label)


def segmentation(dataType, model, classes, img, oriH, oriW, colors, frameCnt, labels, modelName, inputShape):
    if modelName == "DEEP-LAB":
        trainedImageWidth = inputShape[0]
        meanSubtractionValue = 127.5

        h, w = img.shape[:2]
        resizedImage = np.array(Image.fromarray(img.astype('uint8')).resize((trainedImageWidth, trainedImageWidth)))
        resizedImage = (resizedImage / meanSubtractionValue) - 1

        pad_x = int(trainedImageWidth - resizedImage.shape[0])
        pad_y = int(trainedImageWidth - resizedImage.shape[1])
        resized_image = np.pad(resizedImage, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')

        # run prediction
        res = model.predict(np.expand_dims(resized_image, 0))
        labelsResult = np.argmax(res.squeeze(), -1)

        # remove padding and resize back to original image
        if pad_x > 0:
            labelsResult = labelsResult[:-pad_x]
        if pad_y > 0:
            labelsResult = labelsResult[:, :-pad_y]

    elif modelName == "EFFICIENTDET-SEG":
        trainedImageWidth = inputShape[0]
        img = tf.expand_dims(img, 0)
        img = transform_images(img, trainedImageWidth)

        labelsResult = createMask(model(img, False))
        labelsResult = labelsResult.numpy()[0]
    
    labelsTmp = labelsResult.ravel()
    labelsTmp = list(set(labelsTmp))
    labelsTmp.remove(0)

    newLabels, colorMap = label2ColorImage(labelsResult)
    newLabelsCal = np.array(Image.fromarray(newLabels.astype('uint8')))
    newLabelsCal = cv2.cvtColor(newLabelsCal, cv2.COLOR_RGB2BGR)
    newLabelsCalH, newLabelsCalW, _ = newLabelsCal.shape
    contourRatio = (newLabelsCalH * newLabelsCalW) * 0.1

    for idx, colorTmp in enumerate(colorMap[labelsTmp]):
        className = classes[labelsTmp[idx]]
        thresh = np.zeros((newLabelsCalH, newLabelsCalW, 3), np.uint8)
        for yy in range(newLabelsCalH):
            for xx in range(newLabelsCalW):
                if newLabelsCal[yy, xx, 0] == colorTmp[2] and\
                   newLabelsCal[yy, xx, 1] == colorTmp[1] and\
                   newLabelsCal[yy, xx, 2] == colorTmp[0]:

                    thresh.itemset(yy, xx, 0, 255)
                    thresh.itemset(yy, xx, 1, 255)
                    thresh.itemset(yy, xx, 2, 255)
                else:
                    thresh.itemset(yy, xx, 0, 0)
                    thresh.itemset(yy, xx, 1, 0)
                    thresh.itemset(yy, xx, 2, 0)

        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        contours, hieracy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) <= contourRatio:
                continue

            contourX = []
            contourY = []
            contourPt = []
            for pt in contour:
                contourX.append(float(pt[0][0] / newLabelsCalW))
                contourY.append(float(pt[0][1] / newLabelsCalH))
            for i in range(len(contourX)):
                contourPt.append({
                    "X": contourX[i] * w,
                    "Y": contourY[i] * h
                })
            cursor = "isPolygon"
            color = '#%02x%02x%02x' % tuple(colorTmp)
            result = setSegOutput(dataType, className, cursor, contourPt, color, frameCnt)

            labels.append(result)


def autoLabeling(data, model, modelName, inputShape):
    try:
        # data = json.loads(data)
        dataType = data["DATA_TYPE"]
        objectType = data["OBJECT_TYPE"]

        # COLOR, CLASS_DB_NM, RECT(null일 수 있음.), IMAGE_PATH, DATASET_CD, DATA_CD
        images = data["IMAGES"]
        aiCd = data["AI_CD"]
        autoAcc = data["AUTO_ACC"]
        mdlPath = data["MDL_PATH"]
        targetClass = None

        if "TYPE" in data:
            predictType = data["TYPE"]
            if predictType == "STATIC_PREDICT":
                targetClass = data["TARGET_CLASS"]

        classes = getClasses(aiCd, objectType, mdlPath)

        output = []
        imgCnt = 0
        imglen = len(images)

        if dataType == "I":
            resultInfo = {}
            colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(80)]
            for imgInfo in images:
                imgCnt += 1
                labels = []
                imgPath = imgInfo["IMAGE_PATH"]
                datasetCd = imgInfo["DATASET_CD"]
                dataCd = imgInfo["DATA_CD"]
                img, oriH, oriW, resizeH, resizeW = predictDataset(dataType, modelName, objectType, imgPath, inputShape)

                if objectType == "C":
                    predict = model.predict(img)
                    tmp = []
                    for idx, acc in enumerate(predict[0]):
                        tmp.append([idx, acc])
                    tmp = sorted(tmp, key=lambda x: -x[1])

                    classification(tmp, labels, classes, colors, targetClass, objectType, 0)

                elif objectType == "D":
                    log.info("[{}] Start detection".format(datasetCd))
                    detection(
                        dataType,
                        model,
                        classes,
                        img,
                        oriH, oriW,
                        colors,
                        0,
                        labels,
                        autoAcc,
                        modelName,
                        targetClass,
                        objectType
                    )
                    log.info("[{}] End detection".format(datasetCd))

                elif objectType == "S":
                    segmentation(
                        dataType,
                        model,
                        classes,
                        img,
                        oriH, oriW,
                        colors,
                        0,
                        labels,
                        modelName,
                        inputShape
                    )

                resultInfo = {"IMAGE_PATH": imgPath, "DATASET_CD": datasetCd, "DATA_CD": dataCd,
                              "LABELS": labels, "TOTAL_FRAME": 0}

                log.info("[{}] Save result.".format(datasetCd))
                saveJson(imgPath, labels, 0)
                output.append(resultInfo)
                log.info("[{}], image:{} ({}/{})".format(datasetCd, imgPath, imgCnt, imglen))
                log.debug(resultInfo)

        elif dataType == "V":
            resultInfo = {}
            colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])for i in range(80)]
            for imgInfo in images:
                labels = []
                imgPath = imgInfo["IMAGE_PATH"]
                datasetCd = imgInfo["DATASET_CD"]
                dataCd = imgInfo["DATA_CD"]
                vc = cv2.VideoCapture(imgPath)
                frameCnt = 0

                while True:
                    ret, img = vc.read()
                    if ret is not True:
                        break
                    img, oriH, oriW, resizeH, resizeW = predictDataset(dataType, modelName, objectType, img, inputShape)

                    if objectType == "C":
                        predict = model.predict(img)
                        tmp = []
                        for idx, acc in enumerate(predict[0]):
                            tmp.append([idx, acc])
                        tmp = sorted(tmp, key=lambda x: -x[1])

                        classification(tmp, labels, classes, colors, targetClass, objectType, frameCnt)

                    elif objectType == "D":
                        detection(
                            dataType,
                            model,
                            classes,
                            img,
                            oriH, oriW,
                            colors,
                            frameCnt,
                            labels,
                            autoAcc,
                            modelName,
                            targetClass,
                            objectType
                        )
                        log.debug("[{}] End detection. frame={}".format(datasetCd, frameCnt + 1))

                    elif objectType == "S":
                        segmentation(
                            dataType,
                            model,
                            classes,
                            img,
                            oriH, oriW,
                            colors,
                            frameCnt,
                            labels,
                            modelName,
                            inputShape
                        )

                        log.debug("[{}] End detection. frame={}".format(datasetCd, frameCnt + 1))

                    frameCnt += 1

                resultInfo = {"IMAGE_PATH": imgPath, "DATASET_CD": datasetCd, "DATA_CD": dataCd,
                              "LABELS": labels, "TOTAL_FRAME": frameCnt}

                saveJson(imgPath, labels, frameCnt)
                # polygonData [][][][]< framecount
                log.info("[{}] Save result.".format(datasetCd))
                output.append(resultInfo)
                log.info("[{}] Save result. video path={}, {} frames.".format(datasetCd, imgPath, frameCnt))
                log.debug(resultInfo)

        if os.path.isfile(os.path.join(basePath, 'Manager/tmp.jpg')):
            os.remove(os.path.join(basePath, 'Manager/tmp.jpg'))
        return output

    except Exception as e:
        log.error(traceback.format_exc())
        prcErrorData(__file__, str(e))
        # output = {"STATUS": 0, "code": str(e)}
