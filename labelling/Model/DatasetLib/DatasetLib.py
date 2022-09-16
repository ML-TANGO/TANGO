'''
DataSetLib
'''
import os
import sys
import pandas as pd
import json
import traceback
import pathlib
import datetime
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from sklearn.model_selection import train_test_split

# Model 폴더 찾기 위한 방법
pathA = os.path.join(os.path.dirname(__file__), os.path.pardir)
pathB = os.path.join(os.path.join(pathA))
basePath = os.path.abspath(pathB)
# Model Path 등록
sys.path.append(basePath)

from Common.Logger.Logger import logger
from Common.Utils.Utils import getConfig
from Common.Process.Process import prcSendData, prcLogData

from random import randrange
# from Common.Process.Process import prcClose
# from Common.Process.Process import prcLogData

# srvIp, srvPort, logPath, tempPath, datasetPath, aiPath = getConfig()
# server = 'http://{}:{}/api/binary/trainBinLog'.format(srvIp, srvPort)
# headers = {"Content-Type": "application/json; charset=utf-8"}
log = logger("log")


def setClassName(param, labels, saveMdlPath):
    classInfo = []
    classNames = []
    if param["OBJECT_TYPE"] == 'C':
        for idx, className in enumerate(labels):
            classInfo.append(
                {
                    "CLASS_CD": idx,
                    "CLASS_NAME": str(className)
                }
            )
            if len(labels) - 1 != idx:
                classNames.append('{}\n'.format(str(className)))
            else:
                classNames.append(str(className))

    elif param["OBJECT_TYPE"] == 'R':
        classInfo.append(
            {
                "CLASS_CD": 0,
                "CLASS_NAME": param["LABEL_COLUMN_NAME"]
            }
        )
        classNames.append(param["LABEL_COLUMN_NAME"])

    classJson = {
        "CLASS_INFO": classInfo
    }
    with open(os.path.join(saveMdlPath, "classes.json"), "w") as f:
        json.dump(classJson, f)
        f.close()

    with open(os.path.join(saveMdlPath, "classes.names"), "w") as f:
        f.writelines(classNames)
        f.close()


class DatasetLib:
    def __init__(self):
        self.params = {}

    # set parameters server -> binary
    # input : jsonData(server -> binary)
    # output : params
    def setParams(self, jsonData):
        try:
            # for Dataset Loader
            inputData = jsonData["INPUT_DATA"]
            serverParam = jsonData["SERVER_PARAM"]
            modelInfo = jsonData["MODEL_INFO"]

            if "TRAIN_PATH" not in inputData or inputData["TRAIN_PATH"] is None:
                if "MODEL_INFO" not in jsonData or len(jsonData["MODEL_INFO"]) == 0 or jsonData["MODEL_INFO"] is None:
                    now = datetime.datetime.now()
                    nowDatetime = now.strftime('%Y-%m-%d_%H:%M:%S')

                    self.params = {
                        "FILE_PATH": inputData["FILE_PATH"] if "FILE_PATH" in inputData else None,
                        # 저장 파일명이 없으면 현재날짜_현재시간.csv로 저장(이 좋을지, FileNames에서 하나만 땡겨올지?)
                        "TARGET_NAME": inputData["TARGET_NAME"] if "TARGET_NAME" in inputData else nowDatetime + '.csv',
                        # Delimiter 입력 없으면 무조건 ','
                        "DELIMITER": inputData["DELIMITER"] if "DELIMITER" in inputData else ',',
                        "MAPING_INFO": inputData["MAPING_INFO"] if "MAPING_INFO" in inputData else None,
                        "SERVER_PARAM": serverParam,
                        "FE_TYPE": inputData["FE_TYPE"] if "FE_TYPE" in inputData else None,
                    }

                else:
                    self.params = {
                      "SERVER_PARAM": serverParam,
                      "TRAIN_PATH": inputData["TRAIN_PATH"] if "TRAIN_PATH" in inputData else None,
                      "DELIMITER": inputData["DELIMITER"] if "DELIMITER" in inputData else ',',
                      "LABEL_COLUMN_NAME": inputData["LABEL_COLUMN_NAME"] if "LABEL_COLUMN_NAME" in inputData else 'target',
                      "MAPING_INFO": inputData["MAPING_INFO"] if "MAPING_INFO" in inputData else None,
                      "SPLIT_RATIO": float(inputData["DATASET_SPLIT"] / 100) if inputData["SPLIT_YN"] == 'Y' else 0,
                      "TEST_PATH": inputData["TEST_PATH"] if inputData["SPLIT_YN"] == 'N' else None,
                      "FE_TYPE": inputData["FE_TYPE"] if "FE_TYPE" in inputData else None,
                    }

                    dbParam = inputData["DB_INFO"]
                    dbParamKeys = self.getKeys(dbParam)
                    for keys in dbParamKeys:
                        self.params[keys] = dbParam[keys]

                    modelParamKeys = self.getKeys(modelInfo)
                    for keys in modelParamKeys:
                        self.params[keys] = modelInfo[keys]

            else:
                if "MODEL_INFO" not in jsonData or len(jsonData["MODEL_INFO"]) == 0 or jsonData["MODEL_INFO"] is None:
                    inputData = jsonData["INPUT_DATA"]

                    self.params = {
                        "TRAIN_PATH": inputData["TRAIN_PATH"] if "TRAIN_PATH" in inputData else None,
                        "DELIMITER": inputData["DELIMITER"] if "DELIMITER" in inputData else ',',
                        "LABEL_COLUMN_NAME": inputData["LABEL_COLUMN_NAME"] if "LABEL_COLUMN_NAME" in inputData else 'target',
                        "MAPING_INFO": inputData["MAPING_INFO"] if "MAPING_INFO" in inputData else None,
                        "SPLIT_RATIO": float(inputData["DATASET_SPLIT"] / 100) if inputData["SPLIT_YN"] == 'Y' else 0,
                        "TEST_PATH": inputData["TEST_PATH"] if inputData["SPLIT_YN"] == 'N' else None,
                        "OBJECT_TYPE": inputData["OBJECT_TYPE"] if "OBJECT_TYPE" in inputData else "C",
                        "DATA_TYPE": inputData["DATA_TYPE"] if "DATA_TYPE" in inputData else "T",
                        "FE_TYPE": inputData["FE_TYPE"] if "FE_TYPE" in inputData else None,
                    }
                # for Train
                else:
                    self.params = {
                        "SERVER_PARAM": serverParam,
                        "DB_INFO": inputData["DB_INFO"] if "DB_INFO" in inputData else None,
                        "TRAIN_PATH": inputData["TRAIN_PATH"] if "TRAIN_PATH" in inputData else None,
                        "DELIMITER": inputData["DELIMITER"] if "DELIMITER" in inputData else ',',
                        "LABEL_COLUMN_NAME": inputData["LABEL_COLUMN_NAME"] if "LABEL_COLUMN_NAME" in inputData else 'target',
                        "MAPING_INFO": inputData["MAPING_INFO"] if "MAPING_INFO" in inputData else None,
                        "SPLIT_RATIO": float(inputData["DATASET_SPLIT"] / 100) if inputData["SPLIT_YN"] == 'Y' else 0,
                        "TEST_PATH": inputData["TEST_PATH"] if inputData["SPLIT_YN"] == 'N' else None,
                        "FE_TYPE": inputData["FE_TYPE"] if "FE_TYPE" in inputData else None,

                    }

                    modelParamKeys = self.getKeys(modelInfo)
                    for keys in modelParamKeys:
                        self.params[keys] = modelInfo[keys]

            return self.params

        except Exception as e:
            print(traceback.print_exc())
            return str(e)

    # get dict keys
    # input : jsonData
    # output : keys[]
    def getKeys(self, params):
        keys = list(params.keys()) 
        return keys

    # load CSV
    # input : filePath, delimiter
    # output : dataFrame
    def loadCSVFile(self, filePath, delimiter=','):
        if delimiter is None:
            delimiter = ','
        df = pd.read_csv(filePath, delimiter=delimiter, encoding="utf-8-sig")
        return df

    # save CSV File
    # input : dataFrame(dataFrame from loadCsvFile())
    # output : {"STATUS": True/False, "MSG": None/errMsg}
    def saveCSVFile(self, params):
        dataFrame = pd.DataFrame()
        try:
            delimiter = params["DELIMITER"]
            targetName = params["TARGET_NAME"]
            fileNames = params["FILE_NAMES"]
            targetPath = os.path.join(params["FILE_PATH"], targetName)
            for fileName in fileNames:
                filePath = os.path.join(self.params["FILE_PATH"], fileName)
                ext = pathlib.Path(fileName).suffixes
                ext = ext[len(ext) - 1].lower()

                if ext == '.csv':
                    df = self.loadCSVFile(filePath, delimiter)
                elif ext == '.json':
                    df = self.loadJSONFile(filePath)

                dataFrame = dataFrame.append(df)

            # mapping info로 column name mapping 코드 추가
            mappingInfo = params["MAPING_INFO"]
            targetColumn = dict()
            columnType = dict()
            saveDataFrame = pd.DataFrame()
            if mappingInfo is not None:
                for _targetColumn in mappingInfo["TARGET_COLUMN"]:
                    targetColumn.update({list(_targetColumn.keys())[0]: list(_targetColumn.values())[0]})
                    # columnKeys.append(list(targetColumn.keys())[0])
                    # columnValues.append(list(targetColumn.values())[0])

                for _columnType in mappingInfo["COLUMN_TYPE"]:
                    columnType.update({list(_columnType.keys())[0]: list(_columnType.values())[0].lower()})

                # target Column만 가져오기
                saveDataFrame = dataFrame.loc[:, list(targetColumn.keys())]
                # target Column의 type 변경
                saveDataFrame = saveDataFrame.astype(columnType)
                saveDataFrame = saveDataFrame.rename(columns=targetColumn)

            else:
                output = {"STATUS": False, "MSG": "MAPING_INFO is None"}
                saveDataFrame = dataFrame
                log.warning(output)

            saveDataFrame.to_csv(targetPath, sep=delimiter, na_rep='NaN', index=False)
            log.info("SAVE DONE")

            # output = {"STATUS": True, "tgtFilePath": self.params["tgtFilePath"], "tgtDelimiter": delimiter}
            dataFrame = pd.read_csv(targetPath, delimiter=delimiter, encoding="utf-8-sig")

            column = list(dataFrame)
            firstRow = dataFrame.iloc[0].values.tolist()
            randRow = dataFrame.iloc[randrange(len(dataFrame))].values.tolist()

            if len(column) == len(firstRow) and len(column) == len(randRow):
                output = {"STATUS": True, "MSG": None, "TARGET_NAME": targetName}
                log.info(output)
            else:
                output = {"STATUS": False, "MSG": "Delimiter Error"}
                log.error(output)

        except Exception as e:
            prcLogData(str(e))
            log.error(traceback.format_exc())

            output = {"STATUS": False, "MSG": str(e)}

        return output

    # load CSV
    # input : filePath
    # output : dataframe
    def loadJSONFile(self, filePath):
        with open(filePath, "r") as jsonFile:
            data = json.load(jsonFile)
        df = pd.DataFrame()
        for i in range(len(data)):
            tmp = pd.DataFrame(data=[list(data[i].values())], columns=list(data[i].keys()))
            df = df.append(tmp)
            if i % 1000 == 0:
                log.debug("{}/{} data append!".format(i, len(data)))
        return df

    # dataset split
    # input : trainFile(Not none), testFile(can none), splitRatio(can none)
    # output : trainDs, testDs
    def getSplitData(self, trainFile, testFile=None, splitRatio=None, delimiter=','):
        testFileCheck = False
        trainDf = pd.DataFrame()
        testDf = pd.DataFrame()
        # mapping info 반영
        # target label까지
        if splitRatio == 0:
            splitRatio = None

        columnNms = []

        le = LabelEncoder()

        for mapping in self.params["MAPING_INFO"]:
            if mapping["checked"] == 1:
                columnNms.append(mapping["COLUMN_NM"])

        if testFile is not None:
            testFileCheck = True

        if testFileCheck is True:
            for trainFPath in trainFile:
                tmp = pd.read_csv(trainFPath, delimiter=delimiter, encoding="utf-8-sig")
                trainDf = trainDf.append(tmp)

            for testFPath in testFile:
                tmp = pd.read_csv(testFPath, delimiter=delimiter, encoding="utf-8-sig")
                testDf = testDf.append(tmp)

            for column in columnNms:
                if trainDf[column].dtype == 'object':
                    trainDf[column] = le.fit_transform(trainDf[column])

                if testDf[column].dtype == 'object':
                    testDf[column] = le.fit_transform(testDf[column])

        else:
            # Upload data
            if len(self.params["TRAIN_PATH"]) != 0 or self.params["TRAIN_PATH"] is None:
                dataFrame = pd.DataFrame()
                for trainFPath in trainFile:
                    tmp = pd.read_csv(trainFPath, delimiter=delimiter, encoding="utf-8-sig")
                    dataFrame = dataFrame.append(tmp)

            # DB data
            else:
                dataFrame = trainFile

            encoderData = dict()
            for column in columnNms:
                if dataFrame[column].dtype == 'object':
                    dataFrame[column].fillna("None",inplace = True)
                    dataFrame[column] = le.fit_transform(dataFrame[column])
                    encoderData[column] = dict()

                    for i in range(len(list(le.classes_))):
                        encoderData[column][le.classes_[i]] = i
                        # encoderData[column][le.classes_[i]] = i

                    # dataFrame[column] = pd.Categorical(dataFrame[column])
                    # dataFrame[column] = dataFrame[column].cat.codes

            self.params["ENCODER_DATA"] = encoderData
            trainDf, testDf = train_test_split(dataFrame, test_size=splitRatio, random_state=66, shuffle=True)

        trainDf = trainDf[columnNms]
        testDf = testDf[columnNms]

        trainDf = trainDf.fillna(0)
        testDf = testDf.fillna(0)

        # # nan check
        # for data in trainDf:
        #     # for tmp in data:
        #     try:
        #         if data == 'nan':
        #             data = 0.0
        #     except:
        #         continue
        # for data in testDf:
        #     try:
        #         if data == 'nan':
        #             data = 0.0
        #     except:
        #         continue

        trainDf = pd.DataFrame(trainDf, columns=columnNms)
        testDf = pd.DataFrame(testDf, columns=columnNms)
        colNames = trainDf.columns.values.tolist()
        return trainDf, testDf, colNames

    # CSV -> ndArray
    # input : param
    # output : xTrain, yTrain, xTest, yTest, output({"SUCCESS": True/False, "MSG": None/ErrorMsg})
    def getNdArray(self, param):
        xTrain, yTrain, xTest, yTest = None, None, None, None
        output = {}

        try:
            # if param["TRAIN_PATH"] is None:
            #     df = self.getDbData(param)
            trainFile = []

            # Upload data
            if len(param["TRAIN_PATH"]) != 0 and param["TRAIN_PATH"] is not None:
                for tmp in param["TRAIN_PATH"]:
                    trainFile.append(tmp["FILE_PATH"])

            # DB data
            else:
                trainFile, output = self.getDbData(param["DB_INFO"])
                if output["STATUS"] is False:
                    return {"SUCCESS": False, "MSG": output["MSG"]}

            testFile = []
            if param["TEST_PATH"] is not None:
                for tmp in param["TEST_PATH"]:
                    testFile.append(tmp["FILE_PATH"])
            else:
                testFile = None
                splitRatio = param["SPLIT_RATIO"]

            trainDf, testDf, colNames = self.getSplitData(
                trainFile,
                testFile=testFile,
                splitRatio=splitRatio,
                delimiter=param["DELIMITER"]
            )
            
            if param["OBJECT_TYPE"] == 'C':
                if param["LABEL_COLUMN_NAME"] in param["ENCODER_DATA"]:
                    labels = list(param["ENCODER_DATA"][param["LABEL_COLUMN_NAME"]].keys())
                else:
                    tmp1 = trainDf[param["LABEL_COLUMN_NAME"]].astype(int).values.tolist()
                    tmp2 = testDf[param["LABEL_COLUMN_NAME"]].astype(int).values.tolist()
                    labels = tmp1 + tmp2
                    labels = list(set(labels))
            else:
                labels = None

            xTrain = trainDf.drop([param["LABEL_COLUMN_NAME"]], axis=1).values
            yTrain = trainDf[param["LABEL_COLUMN_NAME"]].values
          
            xTest = testDf.drop([param["LABEL_COLUMN_NAME"]], axis=1).values
            yTest = testDf[param["LABEL_COLUMN_NAME"]].values

            xTrain = np.nan_to_num(xTrain)
            yTrain = np.nan_to_num(yTrain)
            xTest = np.nan_to_num(xTest)
            yTest = np.nan_to_num(yTest)

            colNames.remove(param["LABEL_COLUMN_NAME"])

            # vision때 D, S 추가 필요 - smpark
            # classes.json 생성
            if "SERVER_PARAM" in param:
                saveMdlPath = os.path.join(param["SERVER_PARAM"]["AI_PATH"])
                setClassName(param, labels, saveMdlPath)
            output = {"SUCCESS": True, "MSG": None}

        except Exception as e:
            output = {"SUCCESS": False, "MSG": str(e)}
            log.error(traceback.format_exc())

        return xTrain, xTest, yTrain, yTest, colNames, param["LABEL_COLUMN_NAME"], labels, output

    # CSV -> dataFrame
    # input : param
    # output : xTrain, yTrain, xTest, yTest, output({"SUCCESS": True/False, "MSG": None/ErrorMsg})
    def getDataFrame(self, param):
        xTrain, yTrain, xTest, yTest = None, None, None, None
        output = {}
        try:
            trainFile = []

            # Upload data
            if len(param["TRAIN_PATH"]) != 0 and param["TRAIN_PATH"] is not None:
                for tmp in param["TRAIN_PATH"]:
                    trainFile.append(tmp["FILE_PATH"])

            # DB data
            else:
                trainFile, output = self.getDbData(param["DB_INFO"])
                if output["STATUS"] is False:
                    return {"SUCCESS": False, "MSG": output["MSG"]}

            testFile = []
            if param["TEST_PATH"] is not None:
                for tmp in param["TEST_PATH"]:
                    testFile.append(tmp["FILE_PATH"])
            else:
                testFile = None
                splitRatio = param["SPLIT_RATIO"]

            trainDf, testDf, colNames = self.getSplitData(
                trainFile,
                testFile=testFile,
                splitRatio=splitRatio,
                delimiter=param["DELIMITER"]
            )

            if param["OBJECT_TYPE"] == 'C':
                if param["LABEL_COLUMN_NAME"] in param["ENCODER_DATA"]:
                        labels = list(param["ENCODER_DATA"][param["LABEL_COLUMN_NAME"]].keys())
                else:
                    tmp1 = trainDf[param["LABEL_COLUMN_NAME"]].astype(int).values.tolist()
                    tmp2 = testDf[param["LABEL_COLUMN_NAME"]].astype(int).values.tolist()
                    labels = tmp1 + tmp2
                    labels = list(set(labels))
            else:
                labels = None

            xTrain = trainDf.drop([param["LABEL_COLUMN_NAME"]], axis=1)
            yTrain = trainDf[param["LABEL_COLUMN_NAME"]]

            xTest = testDf.drop([param["LABEL_COLUMN_NAME"]], axis=1)
            yTest = testDf[param["LABEL_COLUMN_NAME"]]

            colNames.remove(param["LABEL_COLUMN_NAME"])

            # vision때 D, S 추가 필요 - smpark
            # classes.json 생성
            if "SERVER_PARAM" in param:
                saveMdlPath = os.path.join(param["SERVER_PARAM"]["AI_PATH"])
                setClassName(param, labels, saveMdlPath)

            output = {"SUCCESS": True, "MSG": None}

        except Exception as e:
            output = {"SUCCESS": False, "MSG": str(e)}
            log.error(traceback.format_exc())

        return xTrain, yTrain, xTest, yTest, colNames, param["LABEL_COLUMN_NAME"], labels, output

    # CSV -> TFDS()
    # input : param
    # output : trainDf, testDf, column Names, label Name, output({"SUCCESS": True/False, "MSG": None/ErrorMsg})
    def getTFDS(self, param):
        trainDs, testDs, colNames = None, None, None
        output = {}
        try:
            trainFile = []

            # Upload data
            if len(param["TRAIN_PATH"]) != 0 and param["TRAIN_PATH"] is not None:
                for tmp in param["TRAIN_PATH"]:
                    trainFile.append(tmp["FILE_PATH"])

            # DB data
            else:
                trainFile, output = self.getDbData(param["DB_INFO"])
                if output["STATUS"] is False:
                    return {"SUCCESS": False, "MSG": output["MSG"]}

            testFile = []
            if param["TEST_PATH"] is not None:
                for tmp in param["TEST_PATH"]:
                    testFile.append(tmp["FILE_PATH"])
            else:
                testFile = None
                splitRatio = param["SPLIT_RATIO"]

            trainDf, testDf, colNames = self.getSplitData(
                trainFile,
                testFile=testFile,
                splitRatio=splitRatio,
                delimiter=param["DELIMITER"]
            )
            import re
            newColNames = []
            for colName in colNames:
                if re.search("^[A-Za-z0-9_.\\-/>]*$", colName):
                    pass
                else:
                    colName = colName.replace("(", "_")
                    colName = colName.replace(")", "_")
                    colName = colName.replace(" ", "_")

                newColNames.append(colName)

            colNames = []
            colNames = newColNames

            trainDf.columns = colNames
            testDf.columns = colNames

            if param["OBJECT_TYPE"] == 'C':
                if param["LABEL_COLUMN_NAME"] in param["ENCODER_DATA"]:
                        labels = list(param["ENCODER_DATA"][param["LABEL_COLUMN_NAME"]].keys())
                else:
                    tmp1 = trainDf[param["LABEL_COLUMN_NAME"]].astype(int).values.tolist()
                    tmp2 = testDf[param["LABEL_COLUMN_NAME"]].astype(int).values.tolist()
                    labels = tmp1 + tmp2
                    labels = list(set(labels))
            else:
                labels = None

            colNames.remove(param["LABEL_COLUMN_NAME"])
            target = trainDf.pop(param["LABEL_COLUMN_NAME"])

            trainDs = tf.data.Dataset.from_tensor_slices(
                (
                    {
                        'features': tf.cast(trainDf.values, dtype=tf.float32),
                        'label': tf.cast(target.values, dtype=tf.int64)
                    }
                )
            )

            target = testDf.pop(param["LABEL_COLUMN_NAME"])
            testDs = tf.data.Dataset.from_tensor_slices(
                (
                    {
                        'features': tf.cast(testDf.values, dtype=tf.float32),
                        'label': tf.cast(target.values, dtype=tf.int64)
                    }
                )
            )
            # vision때 D, S 추가 필요 - smpark
            # classes.json 생성
            if "SERVER_PARAM" in param:
                saveMdlPath = os.path.join(param["SERVER_PARAM"]["AI_PATH"])
                setClassName(param, labels, saveMdlPath)

            output = {"SUCCESS": True, "MSG": None}

        except Exception as e:
            output = {"SUCCESS": False, "MSG": str(e)}
            log.error(traceback.format_exc())

        return trainDs, testDs, colNames, param["LABEL_COLUMN_NAME"], labels, output

    # CSV -> torchDataset()
    # input : param
    # output : trainDs, testDs, column Names, label Name, output({"SUCCESS": True/False, "MSG": None/ErrorMsg})
    def getTorchDataset(self, param):
        trainDs, testDs, colNames = None, None, None
        output = {}
        try:
            trainFile = []

            # Upload data
            if len(param["TRAIN_PATH"]) != 0 and param["TRAIN_PATH"] is not None:
                for tmp in param["TRAIN_PATH"]:
                    trainFile.append(tmp["FILE_PATH"])
            # DB data
            else:
                trainFile, output = self.getDbData(param["DB_INFO"])
                if output["STATUS"] is False:
                    return {"SUCCESS": False, "MSG": output["MSG"]}

            testFile = []
            if param["TEST_PATH"] is not None:
                for tmp in param["TEST_PATH"]:
                    testFile.append(tmp["FILE_PATH"])
            else:
                testFile = None
                splitRatio = param["SPLIT_RATIO"]

            trainDf, testDf, colNames = self.getSplitData(
                trainFile,
                testFile=testFile,
                splitRatio=splitRatio,
                delimiter=param["DELIMITER"]
            )

            if param["OBJECT_TYPE"] == 'C':
                if param["LABEL_COLUMN_NAME"] in param["ENCODER_DATA"]:
                        labels = list(param["ENCODER_DATA"][param["LABEL_COLUMN_NAME"]].keys())
                else:
                    tmp1 = trainDf[param["LABEL_COLUMN_NAME"]].astype(int).values.tolist()
                    tmp2 = testDf[param["LABEL_COLUMN_NAME"]].astype(int).values.tolist()
                    labels = tmp1 + tmp2
                    labels = list(set(labels))
            else:
                labels = None

            colNames.remove(param["LABEL_COLUMN_NAME"])
            trainTarget = trainDf.pop(param["LABEL_COLUMN_NAME"])
            testTarget = testDf.pop(param["LABEL_COLUMN_NAME"])

            import torch

            class CustomDataset(torch.utils.data.Dataset):
                def __init__(self, df, target):
                    super(CustomDataset, self).__init__()
                    self.df = df
                    self.target = target
                    self.X, self.y = self.load_data(df, target)

                def __len__(self):
                    return len(self.X)

                def __getitem__(self, index):
                    return (self.X[index], self.y[index])

                def load_data(self, df, target):
                    X = df.values
                    y = target.values

                    X = torch.from_numpy(X).type(torch.FloatTensor)
                    y = torch.from_numpy(y).type(torch.IntTensor)
                    return X, y

            trainDs = CustomDataset(trainDf, trainTarget)
            testDs = CustomDataset(testDf, testTarget)

            # vision때 D, S 추가 필요 - smpark
            # classes.json 생성
            if "SERVER_PARAM" in param:
                saveMdlPath = os.path.join(param["SERVER_PARAM"]["AI_PATH"])
                setClassName(param, labels, saveMdlPath)
            output = {"SUCCESS": True, "MSG": None}

        except Exception as e:
            output = {"SUCCESS": False, "MSG": str(e)}
            log.error(traceback.format_exc())

        return trainDs, testDs, colNames, param["LABEL_COLUMN_NAME"], labels, output

    # train output setting
    # input : param, data
    # output : output
    def setTrainOutput(self, param, data):
        try:
            SRV_IP = param["SERVER_PARAM"]["SRV_IP"]
            SRV_PORT = param["SERVER_PARAM"]["SRV_PORT"]
            TRAINING_INFO_URL = param["SERVER_PARAM"]["TRAINING_INFO_URL"]

            server = "http://{}:{}/api{}".format(SRV_IP, SRV_PORT, TRAINING_INFO_URL)

            # ACCURACY, LOSS, VAL_ACCURACY, VAL_LOSS,
            # R2, VAL_R2, MSE, VAL_MSE, MAE, VAL_MAE, RMSE, VAL_RMSE
            # PRECISION, VAL_PRECISION, RECALL, VAL_RECALL, F1, VAL_F1, ACCURACY, VAL_ACCURACY

            output = {
                "STATUS": True,
                "MSG": None,
                "SRV_ADDR": server,
                "SEND_DATA": {
                    "AI_CD": param["SERVER_PARAM"]["AI_CD"],
                    "OBJECT_TYPE": param["OBJECT_TYPE"],
                    "DATA_TYPE": param["DATA_TYPE"],
                    "MDL_IDX": param["MDL_IDX"] if "MDL_IDX" in param else None,
                    "EPOCH": data["EPOCH"] if "EPOCH" in data else data["ESTIMATOR"],
                    "AI_ACC": data["ACCURACY"] if "ACCURACY" in data else None,
                    "AI_LOSS": data["LOSS"] if "LOSS" in data else None,
                    "AI_VAL_ACC": data["VAL_ACCURACY"] if "VAL_ACCURACY" in data else None,
                    "AI_VAL_LOSS": data["VAL_LOSS"] if "VAL_LOSS" in data else None,
                    "R2_SCORE": data["R2"] if "R2" in data else None,
                    "MSE": data["MSE"] if "MSE" in data else None,
                    "MAE": data["MAE"] if "MAE" in data else None,
                    "RMSE": data["RMSE"] if "RMSE" in data else None,
                    "AI_PRECISION": data["PRECISION"] if "PRECISION" in data else None,
                    "AI_RECALL": data["RECALL"] if "RECALL" in data else None,
                    "F1": data["F1"] if "F1" in data else None,
                    "REMANING_TIME": data["REMANING_TIME"]
                }
            }

        except Exception as e:
            output = {
                "STATUS": False,
                "MSG": str(e)
            }

        return output

    # train status ouput setting
    # input : param, data
    # output : trainDs, testDs, column Names, label Name, output({"SUCCESS": True/False, "MSG": None/ErrorMsg})
    def setStatusOutput(self, param, msg, pid, status):
        SRV_IP = param["SERVER_PARAM"]["SRV_IP"]
        SRV_PORT = param["SERVER_PARAM"]["SRV_PORT"]
        TRAIN_STATE_URL = param["SERVER_PARAM"]["TRAIN_STATE_URL"]

        server = "http://{}:{}/api{}".format(SRV_IP, SRV_PORT, TRAIN_STATE_URL)

        try:

            # ACCURACY, LOSS, VAL_ACCURACY, VAL_LOSS,
            # R2, VAL_R2, MSE, VAL_MSE, MAE, VAL_MAE, RMSE, VAL_RMSE
            # PRECISION, VAL_PRECISION, RECALL, VAL_RECALL, F1, VAL_F1, ACCURACY, VAL_ACCURACY

            output = {
                "SRV_ADDR": server,
                "SEND_DATA": {
                    "STATUS": status,
                    "MSG": msg,
                    "AI_CD": param["SERVER_PARAM"]["AI_CD"],
                    "PID": pid,
                    "MDL_IDX": param["MDL_IDX"],
                    "MODEL_NAME": param["MODEL_NAME"],
                }
            }

        except Exception as e:
            output = {
                "SRV_ADDR": server,
                "SEND_DATA": {
                    "STATUS": False,
                    "MSG": str(e),
                    "AI_CD": param["SERVER_PARAM"]["AI_CD"],
                    "PID": pid,
                    "MDL_IDX": param["MDL_IDX"],
                    "MODEL_NAME": param["MODEL_NAME"],
                }
            }
            log.error(traceback.format_exc())

        return output

    # set predict output
    # input : param, data
    # output : output
    def setPredictOutput(self, param, msg):
        SRV_IP = param["SERVER_PARAM"]["SRV_IP"]
        SRV_PORT = param["SERVER_PARAM"]["SRV_PORT"]
        TRAIN_RESULT_URL = param["SERVER_PARAM"]["TRAIN_RESULT_URL"]

        server = "http://{}:{}/api{}".format(SRV_IP, SRV_PORT, TRAIN_RESULT_URL)
        try:
            output = {
                "STATUS": True,
                "MSG": "Train Done",
                "SRV_ADDR": server,
                "SEND_DATA": msg
            }

        except Exception as e:
            output = {
                "STATUS": False,
                "MSG": str(e)
            }
            log.error(traceback.format_exc())

        return output

    # ORACLE DB - test OK
    def getOracleData(self, param):
        df = None
        try:
            import cx_Oracle as co

            oraclePath = os.environ["ORACLE_HOME"]
            co.init_oracle_client(lib_dir=os.path.join(oraclePath, "lib"))

            st = time.time()
            # db Connection
            dsnTns = co.makedsn(param["ADDRESS"], int(param["PORT"]), param["DBNAME"])
            conn = co.connect(user=param["USER"], password=param["PASSWORD"], dsn=dsnTns)
            df = pd.read_sql(param["QUERY"], conn)
            conn.close()
            et = time.time() - st
            output = {
                "STATUS": True,
                "EXE_TIME": et,
                "MSG": None

            }

        except Exception as e:
            output = {
                "STATUS": False,
                "EXE_TIME": 0,
                "MSG": str(e)
            }
            print(traceback.print_exc())
        return df, output

    # DB2 : test ok
    def getDB2Data(self, param):
        df = None
        try:
            import ibm_db_dbi
            # DB Connection
            st = time.time()
            connInfo = "DATABASE={};HOSTNAME={};PORT={};PROTOCOL=TCPIP;UID={};PWD={};".format(
                param["DBNAME"],
                param["ADDRESS"],
                int(param["PORT"]),
                param["USER"],
                param["PASSWORD"]
            )
            conn = ibm_db_dbi.connect(connInfo, "", "")
            df = pd.read_sql(param["QUERY"], conn)

            conn.close()
            et = time.time() - st

            output = {
                "STATUS": True,
                "EXE_TIME": et,
                "MSG": None

            }

        except Exception as e:
            output = {
                "STATUS": False,
                "MSG": str(e)
            }
        return df, output

    # postgre DB - test ok
    def getPostgreData(self, param):
        df = None
        try:
            import psycopg2 as pg

            # DB Connection
            conn = pg.connect(host=param["ADDRESS"],
                              port=int(param["PORT"]),
                              dbname=param["DBNAME"],
                              user=param["USER"],
                              password=param["PASSWORD"])

            st = time.time()
            df = pd.read_sql(param["QUERY"], conn)
            conn.close()
            et = time.time() - st
            output = {
                "STATUS": True,
                "EXE_TIME": et,
                "MSG": None

            }

        except Exception as e:
            output = {
                "STATUS": False,
                "MSG": str(e)
            }
        return df, output

    # mariaDB - test OK
    def getMariaData(self, param):
        df = None
        try:
            import pymysql
            # DB Connection
            conn = pymysql.connect(host=param["ADDRESS"],
                                   port=int(param["PORT"]),
                                   database=param["DBNAME"],
                                   user=param["USER"],
                                   password=param["PASSWORD"])

            st = time.time()
            df = pd.read_sql(param["QUERY"], conn)
            conn.close()
            et = time.time() - st
            output = {
                "STATUS": True,
                "EXE_TIME": et,
                "MSG": None
            }
        except Exception as e:
            output = {
                "STATUS": False,
                "MSG": str(e)
            }
        return df, output

    def getDbData(self, param):
        if "MYSQL" in param["CLIENT"].upper() or "MYSQL2" in param["CLIENT"].upper():
            return self.getMariaData(param)

        elif "PG" in param["CLIENT"].upper():
            return self.getPostgreData(param)

        elif "DB2" in param["CLIENT"].upper():
            return self.getDB2Data(param)

        elif "ORACLE" in param["CLIENT"].upper():
            return self.getOracleData(param)


# # test Code
# if __name__ == "__main__":
#     # train set param test
#     # data = '{"INPUT_DATA":{"FILE_PATH":"/Users/parksangmin/Downloads/","TRAIN_FILE_NAMES":["kddcup99_csv.csv","kddcup99_csv.csv"],"TARGET_FILE_PATH":"/Users/parksangmin/Downloads/aaa.csv","DELIMITER":",","SPLIT_RATIO":0.25,"LABEL_COLUMN_NAME":"label","MAPING_INFO":{"aaa":"aaa"}},"SERVER_PARAM":{"AI_CD":"eeeee"},"MODEL_INFO":{"MODEL_PATH":"Network/Tabular/TF/TabNetCLF","HYPER_PARAM":{"num_decision_steps":7,"relaxation_factor":1.5,"sparsity_coefficient":1e-05,"batch_momentum":0.98},"MODEL_PARAM":{"BATCH_SIZE":128,"EPOCH":1}}}'
#     # data = '{"INPUT_DATA":{"FILE_PATH":"/Users/parksangmin/Downloads/","FILE_NAMES":["kddcup99_csv.csv","kddcup99_csv.csv"],"TARGET_NAME":"aaa.csv","DELIMITER":",","MAPING_INFO":{"aaa":"aaa"}},"SERVER_PARAM":{"AI_CD":"eeeee"}}'

#     # data = '{"INPUT_DATA":{"DB_INFO":{"CLIENT":"db2","ADDRESS":"localhost","PORT":"50000","USER":"db2inst1","PASSWORD":"db2inst1","DBNAME":"test7","QUERY":"SELECT * FROM test7.stores"},"SPLIT_YN":"Y","DATASET_SPLIT":20,"TEST_DATASET_CD":null,"TEST_PATH":null,"LABEL_COLUMN_NAME":"label","MAPING_INFO":[{"DATASET_CD":"RT210111","COLUMN_NM":"crim","COLUMN_ALIAS":null,"DEFAULT_VALUE":null,"checked":1},{"DATASET_CD":"RT210111","COLUMN_NM":"zn","COLUMN_ALIAS":null,"DEFAULT_VALUE":null,"checked":1},{"DATASET_CD":"RT210111","COLUMN_NM":"indus","COLUMN_ALIAS":null,"DEFAULT_VALUE":null,"checked":1},{"DATASET_CD":"RT210111","COLUMN_NM":"chas","COLUMN_ALIAS":null,"DEFAULT_VALUE":null,"checked":1},{"DATASET_CD":"RT210111","COLUMN_NM":"nox","COLUMN_ALIAS":null,"DEFAULT_VALUE":null,"checked":1},{"DATASET_CD":"RT210111","COLUMN_NM":"rm","COLUMN_ALIAS":null,"DEFAULT_VALUE":null,"checked":1},{"DATASET_CD":"RT210111","COLUMN_NM":"age","COLUMN_ALIAS":null,"DEFAULT_VALUE":null,"checked":1},{"DATASET_CD":"RT210111","COLUMN_NM":"dis","COLUMN_ALIAS":null,"DEFAULT_VALUE":null,"checked":1},{"DATASET_CD":"RT210111","COLUMN_NM":"rad","COLUMN_ALIAS":null,"DEFAULT_VALUE":null,"checked":1},{"DATASET_CD":"RT210111","COLUMN_NM":"tax","COLUMN_ALIAS":null,"DEFAULT_VALUE":null,"checked":1},{"DATASET_CD":"RT210111","COLUMN_NM":"ptratio","COLUMN_ALIAS":null,"DEFAULT_VALUE":null,"checked":1},{"DATASET_CD":"RT210111","COLUMN_NM":"b","COLUMN_ALIAS":null,"DEFAULT_VALUE":null,"checked":1},{"DATASET_CD":"RT210111","COLUMN_NM":"lstat","COLUMN_ALIAS":null,"DEFAULT_VALUE":null,"checked":1},{"DATASET_CD":"RT210111","COLUMN_NM":"label","COLUMN_ALIAS":null,"DEFAULT_VALUE":null,"checked":1}]},"SERVER_PARAM":{"AI_CD":"RT210113","SRV_IP":"0.0.0.0","SRV_PORT":5000,"TRAIN_RESULT_URL":"/tab/binary/trainResultLog","TRAIN_STATE_URL":"/tab/binary/binaryStatusLog","AI_PATH":"/Users/gimminjong/Upload/RT210113/","TRAINING_INFO_URL":"/tab/binary/trainInfUrl"},"MODEL_INFO":{"DATA_TYPE":"T","OBJECT_TYPE":"R","MODEL_NAME":"MODEL_11","MODEL_TYPE":"DL","MDL_ALIAS":"MODEL_11","MDL_IDX":11,"epochs":"10","batch_size":"20","early_stopping":"TRUE","monitor":"r2","mode":"auto","MDL_PATH":"/Users/gimminjong/Documents/bluai_mlkit/Model/Network/Tabular/AUTOKERAS/AUTOKERAS_REG"}}'
#     data = '{"INPUT_DATA":{"TRAIN_PATH":[],"DELIMITER":",","SPLIT_YN":"Y","DATASET_SPLIT":10,"TEST_DATASET_CD":null,"TEST_PATH":null,"LABEL_COLUMN_NAME":"label","MAPING_INFO":[{"DATASET_CD":"CT210114","COLUMN_NM":"sepal_length","COLUMN_ALIAS":null,"DEFAULT_VALUE":null,"IS_CLASS":0,"COLUMN_IDX":0,"checked":1},{"DATASET_CD":"CT210114","COLUMN_NM":"sepal_width","COLUMN_ALIAS":null,"DEFAULT_VALUE":null,"IS_CLASS":0,"COLUMN_IDX":1,"checked":1},{"DATASET_CD":"CT210114","COLUMN_NM":"petal_length","COLUMN_ALIAS":null,"DEFAULT_VALUE":null,"IS_CLASS":0,"COLUMN_IDX":2,"checked":1},{"DATASET_CD":"CT210114","COLUMN_NM":"petal_width","COLUMN_ALIAS":null,"DEFAULT_VALUE":null,"IS_CLASS":0,"COLUMN_IDX":3,"checked":1},{"DATASET_CD":"CT210114","COLUMN_NM":"label","COLUMN_ALIAS":null,"DEFAULT_VALUE":null,"IS_CLASS":1,"COLUMN_IDX":4,"checked":1}],"DB_INFO":{"DATASET_CD":"CT210114","DB_SEQ":0,"CLIENT":"mysql","ADDRESS":"www.wedalab.com","PORT":9156,"DBNAME":"BLUAI_AI","USER":"bluai","PASSWORD":"WEDA_BLUAI_0717","QUERY":"select * from BLUAI_AI.TEST_IRIS","IS_TEST":0,"LIMIT":5}},"SERVER_PARAM":{"AI_CD":"CT20210198","AI_PATH":"/Users/upload/AiModel/CT20210198","SRV_IP":"127.0.0.1","SRV_PORT":10236,"TRAIN_RESULT_URL":"/tab/binary/trainResultLog","TRAIN_STATE_URL":"/tab/binary/binaryStatusLog","TRAINING_INFO_URL":"/tab/binary/trainInfoLog"},"MODEL_INFO":{"DATA_TYPE":"T","OBJECT_TYPE":"C","MODEL_NAME":"XGBClassifier","MODEL_TYPE":"ML","MDL_ALIAS":"XGBClassifier_0","MDL_IDX":0,"n_estimators":"100","max_depth":"6","min_child_weight":"1","gamma":"0","colsample_bytree":"1","colsample_bylevel":"1","colsample_bynode":"1","subsample":"1","learning_rate":"0.3","early_stopping":"TRUE","monitor":"accuracy","mode":"auto","MDL_PATH":"Network/Tabular/XGBOOST/XGB_CLF"}}'

#     param = json.loads(data)

#     datasetLib = DatasetLib()
#     param = datasetLib.setParams(param)

#     df, output = datasetLib.getDbData(param)

#     print(df)
#     print(output)

#     # datasetLib.saveCSVFile()



#     # trainDs, testDs, colNames, label, output = datasetLib.getTFDS(param)
#     # # output = datasetLib.dataLoader()
#     # print(colNames, label, output)

