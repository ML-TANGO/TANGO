import json
import os
import sys
import pandas as pd
import numpy as np

# Model 폴더 찾기 위한 방법
pathA = os.path.join(os.path.dirname(__file__), os.path.pardir, "../../../")
pathB = os.path.join(os.path.join(pathA))
basePath = os.path.abspath(pathB)

# Model Path 등록
sys.path.append(basePath)

from Common.Model.AutoModel.AutoML import AutoML

def createModel(param=None, iterations=None):
    model = AutoML(param=param, iterations=iterations)

    return model