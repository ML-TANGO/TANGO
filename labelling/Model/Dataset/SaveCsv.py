'''
uploaded data -> csv
input : param
output : {STATUS, MSG, TARGETR_NAME}
'''

import json
import os
import sys
import traceback

# Model 폴더 찾기 위한 방법
pathA = os.path.join(os.path.dirname(__file__), os.path.pardir)
pathB = os.path.join(os.path.join(pathA))
basePath = os.path.abspath(pathB)
# Model Path 등록
sys.path.append(basePath)

from DatasetLib.DatasetLib import DatasetLib

if __name__ == "__main__":
    try:
        data = json.loads(sys.argv[1])
        # data = '{"INPUT_DATA":{"FILE_PATH":"/Users/parksangmin/Downloads/","FILE_NAMES":["kddcup99_csv.csv"],"TARGET_NAME":"target.csv","DELIMITER":",","MAPPING_INFO":{"COLUMN_TYPE":[{"protocol_type":"str"},{"service":"str"},{"src_bytes":"int"},{"count":"float"}],"TARGET_COLUMN":[{"protocol_type":"PT"},{"service":"S"},{"src_bytes":"SB"},{"count":"CNT"}]}},"SERVER_PARAM":{"DATASET_CD":"TD000001"}}'
        # data = json.loads(data)

        datasetLib = DatasetLib()
        param = datasetLib.setParams(data)
        output = datasetLib.saveCSVFile(param)

        print(json.dumps(output))

    except Exception as e:
        output = {"STATUS": False, "MSG": str(e)}
        print(json.dumps(output))
