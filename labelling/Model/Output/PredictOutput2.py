# -*- coding:utf-8 -*-
'''
형식에 따른 출력 함수
1. getResultYolo() : yolo 출력 방식
2. express() : logical, con1, accu1, con2, accu2 에 따른 출력 방식
'''
import os
import sys
import numpy as np
import operator as op
from flask import make_response, jsonify
import traceback
import math

# Model 폴더 찾기 위한 방법
basePath = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir, "../")))

# Model Path 등록
sys.path.append(basePath)

from Common.Logger.Logger import logger
log = logger("log")


def getResultYolo(boxes, scores, i, ww, hh, target, saveImage, resultPath, oriBoxes, resultData):
    try:
        boxData = []
        accuracy = []
        boxSize = None

        if oriBoxes is not None:
            x = float(np.array(boxes[0][i][0]) * ww) + oriBoxes[0]
            y = float(np.array(boxes[0][i][1]) * hh) + oriBoxes[1]
            w = float(np.array(boxes[0][i][2]) * ww) + oriBoxes[0]
            h = float(np.array(boxes[0][i][3]) * hh) + oriBoxes[1]

        else:
            x = float(np.array(boxes[0][i][0]) * ww)
            y = float(np.array(boxes[0][i][1]) * hh)
            w = float(np.array(boxes[0][i][2]) * ww)
            h = float(np.array(boxes[0][i][3]) * hh)

        boxData = {"X1": x, "Y1": y, "X2": w, "Y2": h}
        accuracy = float(np.array(scores[0][i]))

        # saveImage = cv2.rectangle(saveImage, (int(x), int(y)), (int(w), int(h)), cvColor, 2)

        boxSize = math.sqrt(pow((x - w), 2) + pow((y - h), 2))
        resultData["ACCURACY"] = accuracy
        resultData["RECT"] = boxData
        resultData["VALUE"] = boxSize

        return resultData

    except Exception as e:
        log.error(e)
        log.error(traceback.format_exc())
        return make_response(jsonify({"STATE": 0, "MSG": str(e)}))


def express(boxes, logical, con1, accu1, con2, accu2, scores, i, ww, hh, target, saveImage, resultPath, oriBoxes, capturedTime):
    classCD, className, con1, accu1, logical, con2, accu2, tag, dpLabel, location, color = target
    resultData = {"CLASS_CD": classCD, "CLASS_NAME": className, "COLOR": color, "DP_LABEL": dpLabel, "LOCATION": location,
                  "ACCURACY": None, "RECT": None, "RAW_TIME": capturedTime}
    if logical == 'none':
        # score > accu1
        if con1 == 'gt':
            if op.gt(np.array(scores[0][i]), float(accu1)):
                resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                        target, saveImage, resultPath, oriBoxes, resultData)

        # score >= accu1
        elif con1 == 'gteq':
            if op.ge(np.array(scores[0][i]), float(accu1)):
                resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                        target, saveImage, resultPath, oriBoxes, resultData)
        # score < accu1
        elif con1 == 'lt':
            if op.lt(np.array(scores[0][i]), float(accu1)):
                resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                        target, saveImage, resultPath, oriBoxes, resultData)
        # score <= accu1
        elif con1 == 'lteq':
            if op.le(np.array(scores[0][i]), float(accu1)):
                resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                        target, saveImage, resultPath, oriBoxes, resultData)
    elif logical == 'and':
        # score > accu1
        if con1 == 'gt':
            # score > accu2
            if con2 == 'gt':
                if op.gt(np.array(scores[0][i]), float(accu1)) and op.gt(np.array(scores[0][i]), float(accu2)):
                    resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                            target, saveImage, resultPath, oriBoxes, resultData)

                # score >= accu2
                if con2 == 'gteq':
                    if op.gt(np.array(scores[0][i]), float(accu1)) and op.ge(np.array(scores[0][i]), float(accu2)):
                        resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                                target, saveImage, resultPath,
                                                                                oriBoxes, resultData)

                # score < accu2
                if con2 == 'lt':
                    if op.gt(np.array(scores[0][i]), float(accu1)) and op.lt(np.array(scores[0][i]), float(accu2)):
                        resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                                target, saveImage, resultPath, oriBoxes,
                                                                                resultData)
                # score <= accu2
                if con2 == 'lteq':
                    if op.gt(np.array(scores[0][i]), float(accu1)) and op.le(np.array(scores[0][i]), float(accu2)):
                        resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                                target, saveImage, resultPath, oriBoxes,
                                                                                resultData)
        # score >= accu1
        elif con1 == 'gteq':
            # score > accu2
            if con2 == 'gt':
                if op.gt(np.array(scores[0][i]), float(accu1)) and op.gt(np.array(scores[0][i]), float(accu2)):
                    resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                            target, saveImage, resultPath, oriBoxes, resultData)
                # score >= accu2
                if con2 == 'gteq':
                    if op.gt(np.array(scores[0][i]), float(accu1)) and op.ge(np.array(scores[0][i]), float(accu2)):
                        resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                                target, saveImage, resultPath, oriBoxes,
                                                                                resultData)
                # score < accu2
                if con2 == 'lt':
                    if op.gt(np.array(scores[0][i]), float(accu1)) and op.lt(np.array(scores[0][i]), float(accu2)):
                        resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                                target, saveImage, resultPath, oriBoxes,
                                                                                resultData)
                # score <= accu2
                if con2 == 'lteq':
                    if op.gt(np.array(scores[0][i]), float(accu1)) and op.le(np.array(scores[0][i]), float(accu2)):
                        resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                                target, saveImage, resultPath, oriBoxes,
                                                                                resultData)
        # score < accu1
        elif con1 == 'lt':
            # score > accu2
            if con2 == 'gt':
                if op.gt(np.array(scores[0][i]), float(accu1)) and op.gt(np.array(scores[0][i]), float(accu2)):
                    resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                            target, saveImage, resultPath, oriBoxes, resultData)
                # score >= accu2
                if con2 == 'gteq':
                    if op.gt(np.array(scores[0][i]), float(accu1)) and op.ge(np.array(scores[0][i]), float(accu2)):
                        resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                                target, saveImage, resultPath, oriBoxes,
                                                                                resultData)
                # score < accu2
                if con2 == 'lt':
                    if op.gt(np.array(scores[0][i]), float(accu1)) and op.lt(np.array(scores[0][i]), float(accu2)):
                        resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                                target, saveImage, resultPath, oriBoxes,
                                                                                resultData)
                # score <= accu2
                if con2 == 'lteq':
                    if op.gt(np.array(scores[0][i]), float(accu1)) and op.le(np.array(scores[0][i]), float(accu2)):
                        resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                                target, saveImage, resultPath, oriBoxes,
                                                                                resultData)
        # score <= accu1
        elif con1 == 'lteq':
            # score > accu2
            if con2 == 'gt':
                if op.gt(np.array(scores[0][i]), float(accu1)) and op.gt(np.array(scores[0][i]), float(accu2)):
                    resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                            target, saveImage, resultPath, oriBoxes, resultData)

            # score >= accu2
            if con2 == 'gteq':
                if op.gt(np.array(scores[0][i]), float(accu1)) and op.ge(np.array(scores[0][i]), float(accu2)):
                    resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                            target, saveImage, resultPath, oriBoxes, resultData)

            # score < accu2
            if con2 == 'lt':
                if op.gt(np.array(scores[0][i]), float(accu1)) and op.lt(np.array(scores[0][i]), float(accu2)):
                    resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                            target, saveImage, resultPath, oriBoxes, resultData)

            # score <= accu2
            if con2 == 'lteq':
                if op.gt(np.array(scores[0][i]), float(accu1)) and op.le(np.array(scores[0][i]), float(accu2)):
                    resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                            target, saveImage, resultPath, oriBoxes, resultData)

    elif logical == 'or':
        # score > accu1
        if con1 == 'gt':
            # score > accu2
            if con2 == 'gt':
                if op.gt(np.array(scores[0][i]), float(accu1)) or op.gt(np.array(scores[0][i]), float(accu2)):
                    resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                            target, saveImage, resultPath, oriBoxes, resultData)

            # score >= accu2
            if con2 == 'gteq':
                if op.gt(np.array(scores[0][i]), float(accu1)) or op.ge(np.array(scores[0][i]), float(accu2)):
                    resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                            target, saveImage, resultPath, oriBoxes, resultData)
            # score < accu2
            if con2 == 'lt':
                if op.gt(np.array(scores[0][i]), float(accu1)) or op.lt(np.array(scores[0][i]), float(accu2)):
                    resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                            target, saveImage, resultPath, oriBoxes, resultData)

            # score <= accu2
            if con2 == 'lteq':
                if op.gt(np.array(scores[0][i]), float(accu1)) or op.le(np.array(scores[0][i]), float(accu2)):
                    resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                            target, saveImage, resultPath, oriBoxes, resultData)

        # score >= accu1
        elif con1 == 'gteq':
            # score > accu2
            if con2 == 'gt':
                if op.gt(np.array(scores[0][i]), float(accu1)) or op.gt(np.array(scores[0][i]), float(accu2)):
                    resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                            target, saveImage, resultPath, oriBoxes, resultData)

            # score >= accu2
            if con2 == 'gteq':
                if op.gt(np.array(scores[0][i]), float(accu1)) or op.ge(np.array(scores[0][i]), float(accu2)):
                    resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                            target, saveImage, resultPath, oriBoxes, resultData)

            # score < accu2
            if con2 == 'lt':
                if op.gt(np.array(scores[0][i]), float(accu1)) or op.lt(np.array(scores[0][i]), float(accu2)):
                    resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                            target, saveImage, resultPath, oriBoxes, resultData)

            # score <= accu2
            if con2 == 'lteq':
                if op.gt(np.array(scores[0][i]), float(accu1)) or op.le(np.array(scores[0][i]), float(accu2)):
                    resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                            target, saveImage, resultPath, oriBoxes, resultData)
        # score < accu1
        elif con1 == 'lt':
            # score > accu2
            if con2 == 'gt':
                if op.gt(np.array(scores[0][i]), float(accu1)) or op.gt(np.array(scores[0][i]), float(accu2)):
                    resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                            target, saveImage, resultPath, oriBoxes, resultData)
            # score >= accu2
            if con2 == 'gteq':
                if op.gt(np.array(scores[0][i]), float(accu1)) or op.ge(np.array(scores[0][i]), float(accu2)):
                    resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                            target, saveImage, resultPath, oriBoxes, resultData)

            # score < accu2
            if con2 == 'lt':
                if op.gt(np.array(scores[0][i]), float(accu1)) or op.lt(np.array(scores[0][i]), float(accu2)):
                    resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                            target, saveImage, resultPath, oriBoxes, resultData)

            # score <= accu2
            if con2 == 'lteq':
                if op.gt(np.array(scores[0][i]), float(accu1)) or op.le(np.array(scores[0][i]), float(accu2)):
                    resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                            target, saveImage, resultPath, oriBoxes, resultData)

        # score <= accu1
        elif con1 == 'lteq':
            # score > accu2
            if con2 == 'gt':
                if op.gt(np.array(scores[0][i]), float(accu1)) or op.gt(np.array(scores[0][i]), float(accu2)):
                    resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                            target, saveImage, resultPath, oriBoxes, resultData)

            # score >= accu2
            if con2 == 'gteq':
                if op.gt(np.array(scores[0][i]), float(accu1)) or op.ge(np.array(scores[0][i]), float(accu2)):
                    resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                            target, saveImage, resultPath, oriBoxes, resultData)

            # score < accu2
            if con2 == 'lt':
                if op.gt(np.array(scores[0][i]), float(accu1)) or op.lt(np.array(scores[0][i]), float(accu2)):
                    resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                            target, saveImage, resultPath, oriBoxes, resultData)

            # score <= accu2
            if con2 == 'lteq':
                if op.gt(np.array(scores[0][i]), float(accu1)) or op.le(np.array(scores[0][i]), float(accu2)):
                    resultData = boxData, accuracy, boxSize = getResultYolo(boxes, scores, i, ww, hh,
                                                                            target, saveImage, resultPath, oriBoxes, resultData)
    return resultData, saveImage


def roughExpress(boxes, accuracy, i, target, capturedTime, hwType, classes, isCD, isType):
    classCD, className, con1, accu1, logical, con2, accu2, tag, location, color = target
    out = {"HW_TYPE": hwType, "BOXES": None, "ACC": None, "CLASS": None, "isRect": None,
           "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}
    if logical == 'none':
        # score > accu1
        if con1 == 'gt':
            if op.gt(accuracy, float(accu1)):
                out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                       "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}

        # score >= accu1
        elif con1 == 'gteq':
            if op.ge(accuracy, float(accu1)):
                out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                       "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}

        # score < accu1
        elif con1 == 'lt':
            if op.lt(accuracy, float(accu1)):
                out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                       "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}

        # score <= accu1
        elif con1 == 'lteq':
            if op.le(accuracy, float(accu1)):
                out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                       "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}

    elif logical == 'and':
        # score > accu1
        if con1 == 'gt':
            # score > accu2
            if con2 == 'gt':
                if op.gt(accuracy, float(accu1)) and op.gt(accuracy, float(accu2)):
                    out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                           "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}

            # score >= accu2
            if con2 == 'gteq':
                if op.gt(accuracy, float(accu1)) and op.ge(accuracy, float(accu2)):
                    out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                           "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}

            # score < accu2
            if con2 == 'lt':
                if op.gt(accuracy, float(accu1)) and op.lt(accuracy, float(accu2)):
                    out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                           "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}

            # score <= accu2
            if con2 == 'lteq':
                if op.gt(accuracy, float(accu1)) and op.le(accuracy, float(accu2)):
                    out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                           "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}

        # score >= accu1
        elif con1 == 'gteq':
            # score > accu2
            if con2 == 'gt':
                if op.gt(accuracy, float(accu1)) and op.gt(accuracy, float(accu2)):
                    out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                           "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}

            # score >= accu2
            if con2 == 'gteq':
                if op.gt(accuracy, float(accu1)) and op.ge(accuracy, float(accu2)):
                    out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                           "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}

            # score < accu2
            if con2 == 'lt':
                if op.gt(accuracy, float(accu1)) and op.lt(accuracy, float(accu2)):
                    out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                           "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}

            # score <= accu2
            if con2 == 'lteq':
                if op.gt(accuracy, float(accu1)) and op.le(accuracy, float(accu2)):
                    out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                           "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}
        # score < accu1
        elif con1 == 'lt':
            # score > accu2
            if con2 == 'gt':
                if op.gt(accuracy, float(accu1)) and op.gt(accuracy, float(accu2)):
                    out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                           "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}

            # score >= accu2
            if con2 == 'gteq':
                if op.gt(accuracy, float(accu1)) and op.ge(accuracy, float(accu2)):
                    out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                           "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}

            # score < accu2
            if con2 == 'lt':
                if op.gt(accuracy, float(accu1)) and op.lt(accuracy, float(accu2)):
                    out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                           "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}

            # score <= accu2
            if con2 == 'lteq':
                if op.gt(accuracy, float(accu1)) and op.le(accuracy, float(accu2)):
                    out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                           "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}

        # score <= accu1
        elif con1 == 'lteq':
            # score > accu2
            if con2 == 'gt':
                if op.gt(accuracy, float(accu1)) and op.gt(accuracy, float(accu2)):
                    out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                           "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}

            # score >= accu2
            if con2 == 'gteq':
                if op.gt(accuracy, float(accu1)) and op.ge(accuracy, float(accu2)):
                    out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                           "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}

            # score < accu2
            if con2 == 'lt':
                if op.gt(accuracy, float(accu1)) and op.lt(accuracy, float(accu2)):
                    out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                           "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}
            # score <= accu2
            if con2 == 'lteq':
                if op.gt(accuracy, float(accu1)) and op.le(accuracy, float(accu2)):
                    out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                           "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}

    elif logical == 'or':
        # score > accu1
        if con1 == 'gt':
            # score > accu2
            if con2 == 'gt':
                if op.gt(accuracy, float(accu1)) or op.gt(accuracy, float(accu2)):
                    out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                           "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}

            # score >= accu2
            if con2 == 'gteq':
                if op.gt(accuracy, float(accu1)) or op.ge(accuracy, float(accu2)):
                    out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                           "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}

            # score < accu2
            if con2 == 'lt':
                if op.gt(accuracy, float(accu1)) or op.lt(accuracy, float(accu2)):
                    out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                           "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}

            # score <= accu2
            if con2 == 'lteq':
                if op.gt(accuracy, float(accu1)) or op.le(accuracy, float(accu2)):
                    out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                           "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}

        # score >= accu1
        elif con1 == 'gteq':
            # score > accu2
            if con2 == 'gt':
                if op.gt(accuracy, float(accu1)) or op.gt(accuracy, float(accu2)):
                    out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                           "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}

            # score >= accu2
            if con2 == 'gteq':
                if op.gt(accuracy, float(accu1)) or op.ge(accuracy, float(accu2)):
                    out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                           "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}

            # score < accu2
            if con2 == 'lt':
                if op.gt(accuracy, float(accu1)) or op.lt(accuracy, float(accu2)):
                    out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                           "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}

            # score <= accu2
            if con2 == 'lteq':
                if op.gt(accuracy, float(accu1)) or op.le(accuracy, float(accu2)):
                    out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                           "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}

        # score < accu1
        elif con1 == 'lt':
            # score > accu2
            if con2 == 'gt':
                if op.gt(accuracy, float(accu1)) or op.gt(accuracy, float(accu2)):
                    out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                           "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}

            # score >= accu2
            if con2 == 'gteq':
                if op.gt(accuracy, float(accu1)) or op.ge(accuracy, float(accu2)):
                    out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                           "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}

            # score < accu2
            if con2 == 'lt':
                if op.gt(accuracy, float(accu1)) or op.lt(accuracy, float(accu2)):
                    out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                           "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}

            # score <= accu2
            if con2 == 'lteq':
                if op.gt(accuracy, float(accu1)) or op.le(accuracy, float(accu2)):
                    out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                           "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}

        # score <= accu1
        elif con1 == 'lteq':
            # score > accu2
            if con2 == 'gt':
                if op.gt(accuracy, float(accu1)) or op.gt(accuracy, float(accu2)):
                    out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                           "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}

            # score >= accu2
            if con2 == 'gteq':
                if op.gt(accuracy, float(accu1)) or op.ge(accuracy, float(accu2)):
                    out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                           "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}

            # score < accu2
            if con2 == 'lt':
                if op.gt(accuracy, float(accu1)) or op.lt(accuracy, float(accu2)):
                    out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                           "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}

            # score <= accu2
            if con2 == 'lteq':
                if op.gt(accuracy, float(accu1)) or op.le(accuracy, float(accu2)):
                    out = {"HW_TYPE": hwType, "BOXES": boxes[0], "ACC": accuracy, "CLASS": classes[i], "isRect": True,
                           "IS_CD": isCD, "IS_TYPE": isType, "RAW_TIME": capturedTime}

    return out
