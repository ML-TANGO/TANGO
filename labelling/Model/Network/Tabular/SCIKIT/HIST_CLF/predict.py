
import os
import sys
import traceback
import numpy as np

# Model 폴더 찾기 위한 방법
pathA = os.path.join(os.path.dirname(__file__), os.path.pardir, "../../../")
pathB = os.path.join(os.path.join(pathA))
basePath = os.path.abspath(pathB)
# Model Path 등록

sys.path.append(basePath)

from Common.Logger.Logger import logger
from Output.GetGraph import graph
from Common.Utils.Utils import getEnData

from sklearn.preprocessing import label_binarize
from sklearn import metrics

log = logger("log")


# predict func
# input : model, yTrain, xTest, yTest, flag
# output : result
def runPredict(model, param=None, xTest=None, yTest=None, colNames=None, classes=None, flag=0):
    g = graph(param, classes)
    output = []

    try:
        # flag == 1 : 각 train.py 에서 실행시키는 경우(evaluate)
        if flag == 1:
            if xTest is None:
                raise Exception("xTest is None!")
            elif yTest is None:
                raise Exception("yTest is None!")
            else:
                if classes is None:
                    raise Exception("classes is None!")
                else:
                    # ACC SCORE
                    yPred = model.predict(xTest)
                    score = metrics.accuracy_score(yTest, yPred)

                    # ROC Curve
                    yTestBinary = label_binarize(yTest, classes=range(0, len(classes)))
                    yPredBinary = label_binarize(yPred, classes=range(0, len(classes)))
                    # yPredProb = model.predict_proba(xTest)
                    rocOutput = g.roc(yTestBinary, yPredBinary)
                    output.append(rocOutput)

                    # Precision Recall Curve
                    preRecallOutput = g.precisionRecall(yTestBinary, yPredBinary)
                    output.append(preRecallOutput)

                    # Confusion Matrix
                    confMat = g.confusionMatrix(yTest, yPred, classes)
                    output.append(confMat)

                    # Feature Importance
                    pImp = g.permutation_fi(model, xTest, yTest, colNames, n_repeats=30)
                    output.append(pImp)

                return score, output

        elif flag == 0:
            try:
                if xTest is None:
                    raise Exception("xTest is None!")
                else:
                    encodeData = getEnData(colNames, xTest, param)
                    xTest = np.array(encodeData)
                    xTest = xTest.reshape((1, -1))
                    try:
                        predict = model.predict(xTest)
                        log.info("PRE {}".format(predict))
                        score = model.predict_proba(xTest)
                        predict = classes[predict[0]]
                        score = np.max(score)

                        predictResult = {
                            "label": predict,
                            "ACCURACY": float(score),
                            "MSG": None
                        }
                    except ValueError as e:
                        log.error(traceback.format_exc())

                        predictResult = {
                            "label": None,
                            "ACCURACY": None,
                            "MSG": str(e)
                        }
                return predictResult

            except Exception as e:
                log.error(traceback.format_exc())

                predictResult = {
                    "label": None,
                    "ACCURACY": None,
                    "MSG": str(e)
                }

    except Exception as e:
        log.error(str(e))
        log.error(traceback.format_exc())