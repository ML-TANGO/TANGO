import os
import sys
import numpy as np
import pandas as pd
from scipy import interp
from sklearn import metrics
import math
from sklearn.inspection import permutation_importance

# Model 폴더 찾기 위한 방법
pathA = os.path.join(os.path.dirname(__file__), os.path.pardir, "../../../")
pathB = os.path.join(os.path.join(pathA))
basePath = os.path.abspath(pathB)
# Model Path 등록

sys.path.append(basePath)

from Common.Logger.Logger import logger

log = logger("log")


class graph:
    def __init__(self, param, classes=None):       
        self.classes = classes
        self.nClasses = 0
        self.param = param
        if classes is not None:
            self.nClasses = len(self.classes)

    # input : yTest, yPred
    # output : output
    # Check : yPred는 decision_function으로 계산해야 함. smpark
    def roc(self, yTest, yPred):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        # for i in range(self.nClasses):
        #     try:
        #         fpr[i], tpr[i], _ = metrics.roc_curve(yTest[:, i], yPred[:, i])
        #         roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        #     except:
        #         continue

        fpr["micro"], tpr["micro"], _ = metrics.roc_curve(yTest.ravel(), yPred.ravel())
        roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

        # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.nClasses)]))

        # Then interpolate all ROC curves at this points
        # mean_tpr = np.zeros_like(all_fpr)
        # for i in range(self.nClasses):
            # mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        # mean_tpr /= self.nClasses

        # fpr["macro"] = all_fpr
        # tpr["macro"] = mean_tpr
        # roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

        microROCPosition = []
        for i in range(len(fpr['micro'])):
            # if (not math.isnan(float(fpr['micro'][i]))) and (not math.isnan(float(tpr['micro'][i]))):
            microROCPosition.append(
                {
                    "X": '{:.4f}'.format(fpr['micro'][i]),
                    "Y": '{:.4f}'.format(tpr['micro'][i])
                }
            )

        # macroROCPosition = []
        # for i in range(len(fpr['macro'])):
        #     if (not math.isnan(float(fpr['macro'][i]))) and (not math.isnan(float(tpr['macro'][i]))):
        #         macroROCPosition.append(
        #             {
        #                 "X": '{:.4f}'.format(fpr['macro'][i]),
        #                 "Y": '{:.4f}'.format(tpr['macro'][i])
        #             }
        #         )

        output = {
            "AI_CD": self.param["SERVER_PARAM"]["AI_CD"] if "SERVER_PARAM" in self.param else None,
            "MDL_IDX": self.param["MDL_IDX"] if "MDL_IDX" in self.param else None,
            "MODEL_NAME": self.param["MODEL_NAME"] if "MODEL_NAME" in self.param else None,

            "GRAPH_TYPE": "ROC_Curve",
            "LEGEND_X": "False Positive Rate",
            "LEGEND_Y": "True Positive Rate",
            "GRAPH_DATA": [
                {
                    "GRAPH_NAME": "ROC Curve",
                    "AREA": '{:.4f}'.format(roc_auc["micro"]),
                    "GRAPH_POSITION": microROCPosition,
                }
                # {
                #     "GRAPH_NAME": "Macro Average ROC Curve",
                #     "AREA": '{:.4f}'.format(roc_auc["micro"]),
                #     "GRAPH_POSITION": macroROCPosition,
                # }
            ]
        }

        # for i in range(self.nClasses):
        #     ROCPositionClsses = []
        #     for j in range(len(fpr[i])):
        #         if (not math.isnan(float(fpr[i][j]))) and (not math.isnan(float(tpr[i][j]))):
        #             ROCPositionClsses.append(
        #                 {
        #                     "X": '{:.4f}'.format(fpr[i][j]),
        #                     "Y": '{:.4f}'.format(tpr[i][j])
        #                 }
        #             )

        #     output["GRAPH_DATA"].append(
        #         {
        #             "GRAPH_NAME": "ROC Curve of class {}".format(self.classes[i]),
        #             "AREA": '{:.4f}'.format(roc_auc[i]),
        #             "GRAPH_POSITION": ROCPositionClsses,
        #         }
        #     )

        return output

    # input : yTest, yPred
    # output : output
    # Check : yPred는 predict_proba로 계산해야 함. smpark
    def precisionRecall(self, yTest, yPred):
        precision = dict()
        recall = dict()
        average_precision = dict()
        graphData = []

        precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(yTest.ravel(), yPred.ravel())
        average_precision["micro"] = metrics.average_precision_score(yTest, yPred, average="micro")

        microPRPosition = []
        for i in range(len(precision['micro'])):
            # if (not math.isnan(float(precision['micro'][i]))) and (not math.isnan(float(recall['micro'][i]))):
            microPRPosition.append(
                {
                    "X": '{:.4f}'.format(recall['micro'][i]),
                    "Y": '{:.4f}'.format(precision['micro'][i])
                }
            )

        # for i in range(self.nClasses):
        #     preRecallPosition = []
        #     precision[i], recall[i], _ = metrics.precision_recall_curve(
        #         yTest[:, i],
        #         yPred[:, i]
        #     )

        #     for j in range(len(precision[i])):
        #         if (not math.isnan(float(precision[i][j]))) and (not(math.isnan(float(recall[i][j])))):
        #             preRecallPosition.append(
        #                 {
        #                     "X": '{:.4f}'.format(recall[i][j]),
        #                     "Y": '{:.4f}'.format(precision[i][j])
        #                 }
        #             )

        #     graphData.append({
        #         "GRAPH_NAME": "Precision Recall Curve of class {}".format(self.classes[i]),
        #         "GRAPH_POSITION": preRecallPosition,
        #     })

        # output = {
        #     "AI_CD": self.param["SERVER_PARAM"]["AI_CD"],
        #     "MDL_IDX": self.param["MDL_IDX"],
        #     "MODEL_NAME": self.param["MODEL_NAME"],
        #     "GRAPH_TYPE": "Precision_Recall_Curve",
        #     "LEGEND_X": "Recall",
        #     "LEGEND_Y": "Precision",
        #     "GRAPH_DATA": graphData
        # }

        output = {
            "AI_CD": self.param["SERVER_PARAM"]["AI_CD"] if "SERVER_PARAM" in self.param else None,
            "MDL_IDX": self.param["MDL_IDX"] if "MDL_IDX" in self.param else None,
            "MODEL_NAME": self.param["MODEL_NAME"] if "MODEL_NAME" in self.param else None,

            "GRAPH_TYPE": "Precision_Recall_Curve",
            "LEGEND_X": "recall",
            "LEGEND_Y": "precision",
            "GRAPH_DATA": [
                {
                    "GRAPH_NAME": "PR Curve",
                    "AREA": '{:.4f}'.format(average_precision["micro"]),
                    "GRAPH_POSITION": microPRPosition,
                }
            ]
        }

        return output

    # input : yTest, yPred
    # output : output
    # Check : yPred는 predict로 계산해야 함. smpark
    # Check2 : yTest, yPred 둘 다 binarize -> decode 필요함(1차원으로 만들어 줘야함)
    def confusionMatrix(self, yTest, yPred, classes):

        if type(yTest[0]) != str:
            labels = range(0, len(classes))
        else:
            labels = classes

        cm = metrics.confusion_matrix(yTest, yPred, normalize="all", labels=labels)
        cm = cm.tolist()
        newCm = []

        for data in cm:
            tmpCm = []
            for tmp in data:
                tmpCm.append(float('{:.4f}'.format(tmp)))
            newCm.append(tmpCm)

        output = {
            "AI_CD": self.param["SERVER_PARAM"]["AI_CD"] if "SERVER_PARAM" in self.param else None,
            "MDL_IDX": self.param["MDL_IDX"] if "MDL_IDX" in self.param else None,
            "MODEL_NAME": self.param["MODEL_NAME"] if "MODEL_NAME" in self.param else None,
            "GRAPH_TYPE": "Confusion_Matrix",
            "LEGEND_X": self.classes,
            "LEGEND_Y": self.classes,
            "GRAPH_DATA": [{
                "GRAPH_NAME": "Confusion Matrix",
                "GRAPH_POSITION": newCm,
            }]
        }
        return output

    # input : yTest, yPred
    # output : output
    # Check : yPred 는 Predict로 계산해야 함. smpark
    def regPlot(self, yTest, yPred):

        graphData = []

        m, b = np.polyfit(yTest, yPred, 1)

        pt = m * yTest + b

        xMin = min(yTest)
        xMax = max(yTest)

        yMin = min(pt)
        yMax = max(pt)

        for i in range(len(yTest)):
            if (not math.isnan(float(yTest[i]))) and (not math.isnan(float(yPred[i]))):
                graphData.append(
                    {
                        "X": '{:.4f}'.format(yTest[i]),
                        "Y": '{:.4f}'.format(yPred[i])
                    }
                )
        output = {
            "AI_CD": self.param["SERVER_PARAM"]["AI_CD"] if "SERVER_PARAM" in self.param else None,
            "MDL_IDX": self.param["MDL_IDX"] if "MDL_IDX" in self.param else None,
            "MODEL_NAME": self.param["MODEL_NAME"] if "MODEL_NAME" in self.param else None,
            "GRAPH_TYPE": "Reg_Plot",
            "LEGEND_X": "Y Label",
            "LEGEND_Y": "Y Pred",
            "BASE_GRAPH_DATA": [
                {
                    "X": '{:.4f}'.format(xMin),
                    "Y": '{:.4f}'.format(yMin)
                },
                {
                    "X": '{:.4f}'.format(xMax),
                    "Y": '{:.4f}'.format(yMax)
                }
            ],
            "GRAPH_DATA": [{
                "GRAPH_NAME": "Reg Plot",
                "GRAPH_POSITION": graphData,
            }]
        }
        return output

    def featureImportance(self, colNames, modelFi):
        
        fImp = list(modelFi)        
        fImpAll = 0
        featureImp = []

        for i in range(len(colNames)):
            fImpAll += fImp[i]

        for j in range(len(fImp)):
            fImp[j] = float(fImp[j]/fImpAll)

        dictFi = dict(zip(colNames, fImp))
        dictFi = sorted(dictFi.items(), reverse=True, key=lambda item: item[1])

        for k in range(len(dictFi)):
            featureImp.append({
                    "name": '{}'.format(dictFi[k][0]),
                    "value": '{:.4f}'.format(dictFi[k][1])
            })  

        output = {
            "AI_CD": self.param["SERVER_PARAM"]["AI_CD"] if "SERVER_PARAM" in self.param else None,
            "MDL_IDX": self.param["MDL_IDX"] if "MDL_IDX" in self.param else None,
            "MODEL_NAME": self.param["MODEL_NAME"] if "MODEL_NAME" in self.param else None,
            "GRAPH_TYPE": "Feature_Importance",
            "GRAPH_DATA": [{
                "GRAPH_NAME": "Feature Importance",
                "GRAPH_POSITION": featureImp,
            }]
        }
        return output

    def permutation_fi(self, model, xTest, yTest, colNames, n_repeats=None):

        pImp = permutation_importance(model, xTest, yTest, n_repeats=n_repeats, random_state=42)
        pImp_mean = pImp.importances_mean
        pImpAll = 0

        permutationImp = []
        for i in range(len(colNames)):
            if pImp_mean[i] < 0:
                pImp_mean[i] = 0.0
            pImpAll += pImp_mean[i]

        for j in range(len(pImp_mean)):
            pImp_mean[j] = float(pImp_mean[j]/pImpAll)

        dictPi = dict(zip(colNames, pImp_mean))
        dictPi = sorted(dictPi.items(), reverse=True, key=lambda item: item[1])

        for k in range(len(dictPi)):
            permutationImp.append({
                    "name": '{}'.format(dictPi[k][0]),
                    "value": '{:.4f}'.format(dictPi[k][1])
            })  

        output = {
            "AI_CD": self.param["SERVER_PARAM"]["AI_CD"] if "SERVER_PARAM" in self.param else None,
            "MDL_IDX": self.param["MDL_IDX"] if "MDL_IDX" in self.param else None,
            "MODEL_NAME": self.param["MODEL_NAME"] if "MODEL_NAME" in self.param else None,
            "GRAPH_TYPE": "Feature_Importance",
            "GRAPH_DATA": [{
                "GRAPH_NAME": "Feature Importance",
                "GRAPH_POSITION": permutationImp,
            }]
        }

        return output

    # input : yTest, yPred
    # output : output
    # Check : yPred 는 Predict로 계산해야 함. smpark
    # Check2 : yTest는 numpy array로 와야함.
    # 고려사항 : 샘플링 필요할 수 있음.
    def distributionPlot(self, yTest, yPred):
        yTestGraphData = []
        yPredGraphData = []
        graphData = []

        for i in range(len(yTest)):
            if not math.isnan(float(yTest[i])):
                yTestGraphData.append(
                    {
                        "X": '{:.4f}'.format(i),
                        "Y": '{:.4f}'.format(yTest[i]),
                    }
                )

        graphData.append(
            {
                "GRAPH_NAME": "Distribution Plot of Y Test Data",
                "GRAPH_POSITION": yTestGraphData,
            }
        )

        for i in range(len(yPred)):
            if not math.isnan(float(yPred[i])):
                yPredGraphData.append(
                    {
                        "X": '{:.4f}'.format(i),
                        "Y": '{:.4f}'.format(yPred[i]),
                    }
                )

        graphData.append(
            {
                "GRAPH_NAME": "Distribution Plot of Y Pred Data",
                "GRAPH_POSITION": yPredGraphData,
            }
        )

        output = {
            "AI_CD": self.param["SERVER_PARAM"]["AI_CD"] if "SERVER_PARAM" in self.param else None,
            "MDL_IDX": self.param["MDL_IDX"] if "MDL_IDX" in self.param else None,
            "MODEL_NAME": self.param["MODEL_NAME"] if "MODEL_NAME" in self.param else None,
            "GRAPH_TYPE": "Distribution_Plot",
            "LEGEND_X": "Amount of Test Dataset",
            "LEGEND_Y": "Y Test and Y Pred",
            "GRAPH_DATA": graphData
        }
        return output