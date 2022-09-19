
import os
import sys
import traceback
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

# Model 폴더 찾기 위한 방법
pathA = os.path.join(os.path.dirname(__file__), os.path.pardir, "../../../")
pathB = os.path.join(os.path.join(pathA))
basePath = os.path.abspath(pathB)
# Model Path 등록

sys.path.append(basePath)

from Common.Logger.Logger import logger
from Output.GetGraph import graph
from Common.Utils.Utils import getEnData


log = logger("log")


# predict func
# input : model, yTrain, xTest, yTest, flag
# output : result
def runPredict(model, param=None, cuda=None, testDs=None, classes=None, colNames=None, flag=0, xTest=None):
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
                    test_loader = DataLoader(testDs, batch_size=len(testDs), shuffle=True)
                    with torch.no_grad():
                        xTest = None
                        yTest = None

                        for data, target in test_loader:
                            data, target = Variable(data), Variable(target)
                            if cuda:
                                data, target = data.cuda(), target.cuda()
                            target = target.type(torch.long)

                            xTest = data
                            yTest = target

                        yScore = model(xTest)

                        if cuda:
                            yScore = yScore.cpu().numpy()
                            yTest = yTest.cpu().numpy()
                        else:
                            yScore = yScore.numpy()
                            yTest = yTest.numpy()

                        newYScore = []
                        for tmp in yScore:
                            maxScore = np.max(tmp)
                            maxIdx = np.where(tmp == maxScore)
                            data = [0 for i in range(len(tmp))]
                            data[maxIdx[0][0]] = 1
                            data = np.array(data)

                            newYScore.append(data)
                        newYScore = np.array(newYScore)

                        newYPred = []
                        for tmp in newYScore:
                            maxIdx = np.argmax(tmp)
                            newYPred.append(maxIdx)

                        yTestBinary = label_binarize(yTest, classes=range(0, len(classes)))
                        yPredBinary = label_binarize(newYPred, classes=range(0, len(classes)))

                    # SCORE
                    score = metrics.accuracy_score(yTest, newYPred)

                    # ROC
                    rocOutput = g.roc(yTestBinary, yPredBinary)
                    output.append(rocOutput)

                    # Precision Recall Curve
                    preRecallOutput = g.precisionRecall(yTestBinary, yPredBinary)
                    output.append(preRecallOutput)

                    # Confusion Matrix
                    confMatOutput = g.confusionMatrix(yTest, newYPred, classes)
                    output.append(confMatOutput)

                return score, output

        # flag == 0 : predictMother에서 실행시키는 경우(predict)
        elif flag == 0:
            try:
                if xTest is None:
                    raise Exception("xTest is None!")
                else:
                    model.eval()
                    with torch.no_grad():
                        encodeData = getEnData(colNames, xTest, param)

                        xTest = np.array(encodeData)
                        xTest = xTest.reshape((1, -1))

                        xTest = torch.from_numpy(xTest).type(torch.FloatTensor)
                        xTest = Variable(xTest)
                        try:
                            score = model(xTest)
                            score = score.numpy()[0]
                            predict = classes[np.argmax(score)]
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