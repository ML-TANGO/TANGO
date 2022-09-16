# etc lib
import os
import sys
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Model 폴더 찾기 위한 방법
pathA = os.path.join(os.path.dirname(__file__), os.path.pardir, "../../../../")
pathB = os.path.join(os.path.join(pathA))
basePath = os.path.abspath(pathB)
# Model Path 등록

sys.path.append(basePath)

from Common.Logger.Logger import logger

log = logger("log")

def getFeature(param, xTrain, xTest, colNames, output):
    data = pd.concat([xTrain, xTest])
    data = data.dropna()
    x = data

    x = StandardScaler().fit_transform(x)
    nComponent = 0
    threshold = 0.95

    for i in range(1, len(colNames)):
        pca = PCA(n_components=i)
        pca.fit_transform(x)

        if pca.explained_variance_ratio_.sum() >= threshold:
            nComponent = i
            break

    output = {
        "COLUMN_INFO": [],
        "SERVER_PARAM": param["SERVER_PARAM"],
        "MODEL_INFO": {
            "MODEL_NAME": "PCA",
            "MODEL_TYPE": param["MODEL_TYPE"],
            "DATA_TYPE": param["DATA_TYPE"],
            "THRESHOLD": threshold,
            "COMPONENT_COUNT": nComponent
        }
    }

    return output
