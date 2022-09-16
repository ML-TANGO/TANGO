import os
import sys
import traceback
import numpy as np
from sklearn import metrics

# Model 폴더 찾기 위한 방법
pathA = os.path.join(os.path.dirname(__file__), os.path.pardir, "../../../")
pathB = os.path.join(os.path.join(pathA))
basePath = os.path.abspath(pathB)
# Model Path 등록

sys.path.append(basePath)

from Common.Logger.Logger import logger
from Common.Utils.Utils import getEnData

from Output.GetGraph import graph

log = logger("log")


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
                # ACC SCORE
                yPred = model.predict(xTest)
                newYPred = yPred.flatten()

                # SCORE
                score = metrics.r2_score(yTest, newYPred)

                # REG Plot
                regPlotOutput = g.regPlot(yTest, newYPred)
                output.append(regPlotOutput)

                # distribution Plot
                distributionPlotOutput = g.distributionPlot(yTest, newYPred)
                output.append(distributionPlotOutput)

                return score, output

        # flag == 0 : predictMother에서 실행시키는 경우(predict)
        elif flag == 0:
            try:
                if xTest is None:
                    raise Exception("xTest is None!")
                else:
                    encodeData = getEnData(colNames, xTest, param)

                    xTest = np.array(encodeData)
                    xTest = xTest.reshape((1, -1))
                    xTest = xTest.astype(np.unicode)

                    try:
                        predict = model.predict(xTest)
                        predict = predict[0]
                        predictResult = {
                            "label": float(predict),
                            "MSG": None
                        }
                    except ValueError as e:
                        log.error(str(e))
                        predictResult = {
                            "label": None,
                            "MSG": str(e)
                        }
                return predictResult

            except Exception as e:
                log.error(str(e))
                predictResult = {
                    "label": None,
                    "MSG": str(e)
                }

    except Exception as e:
        log.error(str(e))
        log.error(traceback.format_exc())
