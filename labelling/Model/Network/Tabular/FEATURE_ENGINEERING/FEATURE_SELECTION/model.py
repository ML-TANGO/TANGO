import os
import sys
import numpy as np

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

from collections import Counter

# Model 폴더 찾기 위한 방법
pathA = os.path.join(os.path.dirname(__file__), os.path.pardir, "../../../")
pathB = os.path.join(os.path.join(pathA))
basePath = os.path.abspath(pathB)
# Model Path 등록

sys.path.append(basePath)

from Common.Logger.Logger import logger

log = logger("log")

def getFeature(param, xTrain, yTrain, colNames, output):
    n_estimators = 10
    model_list = {
        "R": [
            RandomForestRegressor(n_estimators=n_estimators),
            ExtraTreesRegressor(n_estimators=n_estimators),
            XGBRegressor(n_estimators=n_estimators, use_label_encoder=False),
            LGBMRegressor(n_estimators=n_estimators),
            CatBoostRegressor(iterations=n_estimators, verbose=False, allow_writing_files=False)],
        "C": [
            RandomForestClassifier(n_estimators=n_estimators),
            ExtraTreesClassifier(n_estimators=n_estimators),
            XGBClassifier(n_estimators=n_estimators, use_label_encoder=False),
            LGBMClassifier(n_estimators=n_estimators),
            CatBoostClassifier(iterations=n_estimators, verbose=False, allow_writing_files=False)]
    }

    sfmFiDictList = list()
    sfmFiDictCount = list()
    features = np.array(colNames)

    if param["FE_TYPE"] is not None:
        for models in model_list[param["FE_TYPE"]]:
            model = models
            model.fit(xTrain, yTrain)

            fi = model.feature_importances_

            fiMax = 0
            newFi = list()
            if max(fi) > 1:
                for fiAll in fi:
                    fiMax += fiAll
                for fiChange in fi:
                    newFi.append(float(fiChange / fiMax))
                fiDict = dict(zip(features, newFi))
            else:
                fiDict = dict(zip(features, fi))

            # SelectFromModel 사용
            sfm = SelectFromModel(estimator=model, threshold='median').fit(xTrain, yTrain)

            sfm_fi = list()
            sfmFeatures = list(features[sfm.get_support()])

            for sfmFeature in sfmFeatures:
                sfm_fi.append(fiDict[sfmFeature])

            sfmFiDict = dict(zip(sfmFeatures, sfm_fi))
            sfmFiDictList.append(Counter(sfmFiDict))
            for i in range(len(sfmFeatures)):
                sfmFiDictCount.append(sfmFeatures[i])

        # Counter사용 Dict 연산하기
        addFi = Counter()
        for i in range(len(sfmFiDictList)):
            addFi += sfmFiDictList[i]

    dictCount = dict(Counter(sfmFiDictCount))
    dictValue = dict(addFi)

    # Counter사용 컬럼별 Dict 평균값 계산
    result = dict()
    for i in dictCount.keys():
        dictMean = dictValue[i] / dictCount[i]
        result[i] = dictMean

    result = sorted(result.items(), key=(lambda x: x[1]), reverse=True)

    # Threshold
    threshold = 0.05

    # Output
    columnInfo = list()
    for j in range(len(result)):
        if result[j][1] > threshold:
            columnInfo.append({
                "COLUMN_NAME": '{}'.format(result[j][0]),
                "VALUE": '{}'.format(result[j][1]),
                "RANK": '{}'.format(j + 1)
            })

    output = {
        "COLUMN_INFO": columnInfo,
        "SERVER_PARAM": param["SERVER_PARAM"],
        "MODEL_INFO": {
            "MODEL_NAME": "FEATURE_SELECTION",
            "MODEL_TYPE": param["MODEL_TYPE"],
            "DATA_TYPE": param["DATA_TYPE"],
            "THRESHOLD": threshold,
            "COMPONENT_COUNT": len(columnInfo)
        }
    }

    return output
