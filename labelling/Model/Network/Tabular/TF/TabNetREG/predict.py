
import os
import sys
import traceback
import numpy as np
import tensorflow as tf
import re


# Model 폴더 찾기 위한 방법
pathA = os.path.join(os.path.dirname(__file__), os.path.pardir, "../../../")
pathB = os.path.join(os.path.join(pathA))
basePath = os.path.abspath(pathB)
# Model Path 등록

sys.path.append(basePath)

from Common.Logger.Logger import logger
from Output.GetGraph import graph
from Common.Utils.Utils import getEnData

from sklearn import metrics

log = logger("log")

columns = None
# dataset 변형 함수
# input : dataset
# output : x, y


def transform(ds):
    features = tf.unstack(ds['features'])
    labels = ds['label']

    x = dict(zip(columns, features))
    y = tf.one_hot(labels, 1)
    return x, y


# predict func
# input : model, yTrain, xTest, yTest, flag
# output : result
def runPredict(model, param=None, testDs=None, colNames=None, classes=None, flag=0, xTest=None):
    g = graph(param, classes)
    output = []
    try:
        # flag == 1 : 각 train.py 에서 실행시키는 경우(evaluate)
        if flag == 1:
            if testDs is None:
                raise Exception("test Dataset is None!")
            else:
                if classes is None:
                    raise Exception("classes is None!")
                else:
                    xTest, yTest = next(iter(testDs))
                    yScore = model(xTest)

                    yScore = yScore.numpy()
                    yTest = yTest.numpy()

                    yTest2 = []
                    yScore2 = []
                    for data in yTest:
                        data = data.tolist()
                        yTest2.append(data[0])

                    for data in yScore:
                        data = data.tolist()
                        yScore2.append(data[0])

                    newYScore = []
                    for tmp in yScore:
                        maxScore = np.max(tmp)
                        maxIdx = np.where(tmp == maxScore)
                        data = [0.0 for i in range(len(tmp))]
                        data[maxIdx[0][0]] = 1.0
                        data = np.array(data)

                        newYScore.append(data)

                    newYScore = np.array(newYScore)
                    # ACC SCORE
                    score = metrics.accuracy_score(yTest, newYScore)

                    # REG Plot
                    regPlotOutput = g.regPlot(yTest2, yScore2)
                    output.append(regPlotOutput)

                    # distribution Plot
                    distributionPlotOutput = g.distributionPlot(yTest2, yScore2)
                    output.append(distributionPlotOutput)

                return score, output

        # flag == 0 : predictMother에서 실행시키는 경우(predict)
        # flag == 0 : predictMother에서 실행시키는 경우(predict)
        elif flag == 0:
            try:
                if xTest is None:
                    raise Exception("xTest is None!")
                else:
                    try:
                        encodeData = list()
                        for i in range(len(xTest)):
                            tmp = xTest[i]

                            if re.search("^[A-Za-z0-9_.\\-/>]*$", tmp["COLUMN_NAMES"]):
                                pass
                            else:
                                tmp["COLUMN_NAMES"] = tmp["COLUMN_NAMES"].replace("(", "_")
                                tmp["COLUMN_NAMES"] = tmp["COLUMN_NAMES"].replace(")", "_")
                                tmp["COLUMN_NAMES"] = tmp["COLUMN_NAMES"].replace(" ", "_")

                        encodeData = getEnData(colNames, xTest, param)
                      
                        xTest = np.array(encodeData)
                        xTest = xTest.reshape((1, -1))

                        global columns
                        columns = colNames

                        dataset = tf.data.Dataset.from_tensor_slices(
                            (
                                {
                                    'features': tf.cast(xTest, dtype=tf.float32),
                                    'label': tf.cast([1], dtype=tf.int64)
                                }
                            )
                        )

                        dataset = dataset.take(1)
                        dataset = dataset.map(transform)
                        dataset = dataset.batch(1)

                        xTest, _ = next(iter(dataset))

                        score = model(xTest)
                        log.info(score)
                        # score = score.numpy()

                        # predict = classes[np.argmax(score)]
                        # score = np.max(score)

                        # predictResult = {
                        #     "label": predict,
                        #     "ACCURACY": float(score),
                        #     "MSG": None
                        # }
                    except ValueError as e:
                        log.error(traceback.format_exc())
                        predictResult = {
                            "label": None,
                            "ACCURACY": None,
                            "MSG": str(e)
                        }
                    except Exception as e:
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
