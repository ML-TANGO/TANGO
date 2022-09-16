'''
EDA
'''

# pip install pandas-profiling
import os
import sys
import json
import pandas as pd
import numpy as np
import traceback
from pandas_profiling import ProfileReport, profile_report


# Model 폴더 찾기 위한 방법
pathA = os.path.join(os.path.dirname(__file__), os.path.pardir)
pathB = os.path.join(os.path.join(pathA))
basePath = os.path.abspath(pathB)

# Model Path 등록
sys.path.append(basePath)

from Common.Logger.Logger import logger
from DatasetLib import DatasetLib

log = logger("log")

'''
input : {"FILE_PATH": "/Users/upload/DataSets/RT210026/DTR_FINAL.csv"}
output : profile data
'''

if __name__ == "__main__":
    try:

        # # boston
        # data = '{"FILE_PATH":"/Users/upload/DataSets/RT210014/boston.csv"}'
        # iris
        # data = '{"FILE_PATH":"/Users/parksangmin/Downloads/iris.csv"}'
        # data = '{"FILE_PATH":"/Users/parksangmin/Downloads/test.csv"}'
        # # dtr
        # data = '{"FILE_PATH":"/Users/upload/DataSets/RT210026/DTR_FINAL.csv"}'
        # # kdd
        # data = '{"FILE_PATH":"/Users/parksangmin/Downloads/ptbdb.csv"}'

        # data = '{"DB_INFO":{"CLIENT":"oracle","ADDRESS":"www.wedalab.com","PORT":"9154","USER":"CTMSPLUS","PASSWORD":"HKTIRE_CTMS","DBNAME":"XE","QUERY":"SELECT * FROM CALYTS001"}}'

        # param = json.loads(data)

        param = json.loads(sys.argv[1])
        log.debug(param)

        data = pd.DataFrame()
        getDataOuptut = {}
        if "FILE_PATH" in param:
            try:
                for path in param["FILE_PATH"]:
                    tmpData = pd.read_csv(path, encoding='utf-8')
                    data = pd.concat([data, tmpData])

                getDataOuptut["STATUS"] = True
                getDataOuptut["MSG"] = None
            except Exception as e:
                getDataOuptut["STATUS"] = False
                getDataOuptut["MSG"] = str(e)

        elif "DB_INFO" in param:
            try:
                param = param["DB_INFO"]
                dataLib = DatasetLib.DatasetLib()
                data, getDataOuptut = dataLib.getDbData(param)
                getDataOuptut["STATUS"] = True
                getDataOuptut["MSG"] = None

            except Exception as e:
                getDataOuptut["STATUS"] = False
                getDataOuptut["MSG"] = str(e)

        if getDataOuptut["STATUS"] is False:
            print(json.dumps(getDataOuptut))
            sys.exit()

        configPath = os.path.join(basePath, "Analytics/config.yaml")

        rowCnt = len(data.index)
        colCnt = len(data.columns)
        totalCnt = rowCnt * colCnt
        isNullCnt = data.isnull().sum().sum()

        isNullRate = float(isNullCnt / totalCnt * 100)

        if isNullRate >= 50:
            profile = data.profile_report(
                config_file=configPath,
                correlations={"cramers": {"calculate": False}},
                interactions={"continuous": False}
            )
        else:
            profile = ProfileReport(data, config_file=configPath)

        profileData = json.loads(profile.to_json())

        # with open('/Users/parksangmin/Downloads/test.json', "r") as f:
        #     profileData = json.load(f)

        output = dict()

        # get overview
        overview = profileData["table"]

        # overview output
        output["OVER_VIEW"] = {
            "DATASET_STATISTICS": {
                "VARIABLES_COUNT": overview["n_var"],
                "COUNT": overview["n"],
                "MISSING_CELLS": overview["n_cells_missing"],
                "MISSING_CELLS(%)": overview["p_cells_missing"] * 100,
                "DUPLICATE_ROWS": overview["n_duplicates"],
                "DUPLICATE_ROWS(%)": overview["p_duplicates"] * 100,
            },
            "VARIABLE_TYPES": overview["types"]
        }

        # get variables
        variables = profileData["variables"]
        variable = []
        for key, data in variables.items():
            variableData = {
                "VARIABLE_NAME": key,
                "VARIABLE_TYPE": data["type"],
                "COUNT": data["count"],
                "DISTINCT": data["n_distinct"],
                "DISTINCT(%)": data["p_distinct"],
                "MISSING": data["n_missing"],
                "MISSING(%)": data["p_missing"],
            }
            if data["type"] == 'Numeric':
                variableData["MEAN"] = data["mean"]
                variableData["STD"] = data["std"]
                variableData["VARIANCE"] = data["variance"]
                variableData["MIN"] = data["min"]
                variableData["MAX"] = data["max"]
                variableData["KURTOSIS"] = data["kurtosis"]
                variableData["SKEWNESS"] = data["skewness"]
                variableData["SUM"] = data["sum"]
                variableData["MAD"] = data["mad"]
                variableData["5%"] = data["5%"]
                variableData["25%"] = data["25%"]
                variableData["50%"] = data["50%"]
                variableData["75%"] = data["75%"]
                variableData["95%"] = data["95%"]
                variableData["IQR"] = data["iqr"]
                variableData["CV"] = data["cv"]

                # histogram
                graphData = []
                for idx, value in enumerate(data["histogram"]["counts"]):
                    graphData.append(
                        {
                            "X": '{:.4f}'.format(data["histogram"]["bin_edges"][idx]),
                            "Y": '{:.4f}'.format(value),
                        }
                    )

            else:
                # histogram
                graphData = []
                for idx, value in data['value_counts_without_nan'].items():
                    graphData.append(
                        {
                            "X": idx,
                            "Y": '{:.4f}'.format(value),
                        }
                    )

            variableData["HISTOGRAM"] = {
                "GRAPH_TYPE": "HISTOGRAM",
                "LEGEND_X": "BINS" if data["type"] == 'Numeric' else "LABELS",
                "LEGEND_Y": "FREQUENCY" if data["type"] == 'Numeric' else "COUNT",
                "GRAPH_DATA": [
                    {
                        "GRAPH_NAME": "{} HISTOGRAM".format(key),
                        "GRAPH_POSITION": graphData
                    }
                ]
            }

            variable.append(variableData)

        output["VARIABLES"] = variable

        # get correlations
        correlations = profileData["correlations"]
        correlation = []
        for key, value in correlations.items():
            graphData = []
            for data in value:
                graphData.append(list(data.values()))

            graphData = np.array(graphData)
            # graphData = (graphData - graphData.min(axis=0)) / (graphData.max(axis=0) - graphData.min(axis=0))
            graphData = graphData.tolist()
            correlData = {
                "GRAPH_TYPE": "CORRELATION GRAPH",
                "LEGEND_X": list(data.keys()),
                "LEGEND_Y": list(data.keys()),
                "GRAPH_DATA": [
                    {
                        "GRAPH_NAME": key,
                        "GRAPH_POSITION": graphData
                    }
                ]
            }

            correlation.append(correlData)

        output["CORRELATIONS"] = correlation

        samples = []

        for sample in profileData["sample"]:
            try:
                sample["COLUMNS"] = list(sample["data"][0].keys())
                sample["ID"] = sample.pop("id")
                sample["DATA"] = sample.pop("data")
                sample["NAME"] = sample.pop("name")
                del(sample["caption"])

            except:
                continue

            samples.append(sample)

        output["SAMPLES"] = samples
        output["STATUS"] = 1
        output["MSG"] = None
        print(json.dumps(output, ensure_ascii=False))
        log.info("Analysis SUCCESS")

    except Exception as e:
        output = {
            "MSG": str(e),
            "STATUS": 0
        }
        log.error("Analysis Failed")
        log.error(traceback.format_exc())
        print(traceback.format_exc())
        print(json.dumps(output, ensure_ascii=False))
