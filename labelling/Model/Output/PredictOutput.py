import numpy as np
import operator as op
import traceback
import math


def classificationOutput(className, color, accuracy):
    labels = {
        "label": className,
        "COLOR": color,
        "ACCURACY": accuracy,
    }
    return labels


def detectionOutput(className, color, accuracy, imgW, imgH, boxes):
    x1 = float(np.array(boxes[0]) * imgW)
    y1 = float(np.array(boxes[1]) * imgH)
    x2 = float(np.array(boxes[2]) * imgW)
    y2 = float(np.array(boxes[3]) * imgH)

    boxSize = math.sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2))
    labels = {
        "label": className,
        "COLOR": color,
        "ACCURACY": float(accuracy),
        "POSITION": [{"X": x1, "Y": y1}, {"X": x2, "Y": y2}],
        "VALUE": float(boxSize)
    }
    return labels


def segmentationOutput(className, color, accuracy):
    labels = {
        "label": className,
        "COLOR": color,
        "ACCURACY": accuracy
    }
    return labels


def tabularOutput(className, accuracy):
    if className is not None:
        labels = {
            "label": className,
            "ACCURACY": accuracy
        }
    else:
        labels = {
            "label": accuracy,
        }
    return labels


def getLabels(objectType, className, color, accuracy, imgW=None, imgH=None, boxes=None):
    if objectType == "C":
        labels = classificationOutput(className, color, accuracy)
    elif objectType == "D":
        labels = detectionOutput(className, color, accuracy, imgW, imgH, boxes)
    elif objectType == "S":
        labels = segmentationOutput(className, color, accuracy)
    elif objectType == "T":
        labels = tabularOutput(className, accuracy)
    return labels


operatorMapper = {
    "gt": op.gt,
    "gteq": op.ge,
    "lt": op.lt,
    "lteq": op.le
}


def callFunc(x, y, func):
    try:
        return operatorMapper[func](x, y)
    except:
        return "Invalid function"


def operator(accuracy, accScope, color, objectType, className, imgW=None, imgH=None, boxes=None):
    accScope = accScope.split(',')
    con1 = accScope[0] if len(accScope[0]) > 0 else 0.01
    accu1 = accScope[1]
    logical = accScope[2]
    con2 = accScope[3]
    accu2 = accScope[4]
    labels = None

    if logical == 'none':
        if callFunc(accuracy, float(accu1), con1):
            labels = getLabels(objectType, className, color, accuracy, imgW, imgH, boxes)

    elif logical == 'and':
        if callFunc(accuracy, float(accu1), con1) and callFunc(accuracy, float(accu2), con2):
            labels = getLabels(objectType, className, color, accuracy, imgW, imgH, boxes)

    elif logical == 'or':
        if callFunc(accuracy, float(accu1), con1) or callFunc(accuracy, float(accu2), con2):
            labels = getLabels(objectType, className, color, accuracy, imgW, imgH, boxes)

    return labels


def operator2(accuracy, accScope):
        accScope = accScope.split(',')
        con1 = accScope[0] if len(accScope[0]) > 0 else 0.01
        accu1 = accScope[1]
        logical = accScope[2]
        con2 = accScope[3]
        accu2 = accScope[4]

        if logical == 'none':
            return callFunc(accuracy, float(accu1), con1)

        elif logical == 'and':
            return callFunc(accuracy, float(accu1), con1) and callFunc(accuracy, float(accu2), con2)

        elif logical == 'or':
            return callFunc(accuracy, float(accu1), con1) or callFunc(accuracy, float(accu2), con2)

