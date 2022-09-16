# -*- coding: utf-8 -*-
import os
import sys
import pathlib
import json

# Model 폴더 찾기 위한 방법
pathA = os.path.join(os.path.dirname(__file__), os.path.pardir)
pathB = os.path.join(os.path.join(pathA))
basePath = os.path.abspath(pathB)
# Model Path 등록
sys.path.append(basePath)

# from Common.Logger.Logger import logger
# log = logger("log")


# get header
# input filePath, ext
# output header
def getHeader(filePath, ext):
    if ext.lower() == '.csv':
        import pandas as pd
        df = pd.read_csv(filePath, nrows=1)
        columns = df.columns.values.tolist()

    elif ext.lower() == '.json':
        import json
        with open(filePath, "r") as fp:
            for line in fp:
                try:
                    data = json.loads(line)
                    columns = list(data.keys())
                    break

                except:
                    continue

    return columns


if __name__ == "__main__":
    filePath = sys.argv[1]
    try:
        baseName = os.path.basename(filePath)
        ext = pathlib.Path(baseName).suffixes[0]

        columns = getHeader(filePath, ext)
        output = {"COLUMNS": columns}

        print(json.dumps(output))

    except Exception as e:
        print(json.dumps({"err": e}))

