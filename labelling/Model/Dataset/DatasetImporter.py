
# pascal VOC
# MSCOCO

import os
import sys
import json
import simdjson
from glob import glob
import time

import random
import traceback

import pathlib


class datasetImporter():

    def __init__(self, param):
        self.param = param

        self.datasetCd = param["DATASET_CD"]
        self.purposeType = param["PURPOSE_TYPE"].upper()
        self.fileInfo = self.param["FILE_INFO"]

    # def voc2weda(self):
    #     import xmltodict

    def coco2weda(self):
        currentPath = os.path.dirname(self.fileInfo[0]["FILE_PATH"])

        jsonPath = glob(os.path.join(currentPath, "*.json"))
        errorFiles = []
        msg = ""

        try:
            for jsonFile in jsonPath:
                st = time.time()
                with open(jsonFile, 'rb') as f:
                    jsonData = simdjson.loads(f.read())

                categories = jsonData["categories"]
                annotations = jsonData["annotations"]
                images = jsonData["images"]

                colors = ["#%06x" % random.randint(0, 0xFFFFFF) for i in range(0, len(categories))]

                for idx, category in enumerate(categories):
                    category["TAG_CD"] = category["id"]
                    category["TAG_NAME"] = category["name"]
                    category["COLOR"] = colors[idx]

                    del(category["supercategory"])
                    del(category["name"])
                    del(category["id"])

                for fileIdx, fileInfo in enumerate(self.fileInfo):
                    try:
                        filePath = fileInfo["FILE_PATH"]
                        dataCd = fileInfo["DATA_CD"]

                        polygonData = []
                        baseName = os.path.basename(filePath)

                        tmpId = (item for item in images if item["file_name"] == baseName)
                        imageInfo = next(tmpId, False)

                        if imageInfo is not False:
                            imageId = imageInfo["id"]
                        else:
                            errorFiles.append(baseName)
                            msg = "File is not Found!"

                        # get annotaion Data for image_id
                        annoInfo = [item for item in annotations if item["image_id"] == imageId]

                        if self.purposeType == "D":
                            for annoData in annoInfo:
                                tmpId = (item for item in categories if item["TAG_CD"] == annoData["category_id"])
                                categoryInfo = next(tmpId, False)
                                polygonData.append({
                                    "DATASET_CD": self.datasetCd,
                                    "DATA_CD": dataCd,
                                    "TAG_CD": categoryInfo["TAG_CD"],
                                    "TAG_NAME": categoryInfo["TAG_NAME"],
                                    "COLOR": categoryInfo["COLOR"],
                                    "CLASS_CD": None,
                                    "CURSOR": "isRect",
                                    "NEEDCOUNT": 2,
                                    "POSITION": [
                                        {
                                            "X": float(round(annoData["bbox"][0], 2)),
                                            "Y": float(round(annoData["bbox"][1], 2))
                                        },
                                        {
                                            "X": float(round(annoData["bbox"][0] + annoData["bbox"][2], 2)),
                                            "Y": float(round(annoData["bbox"][1] + annoData["bbox"][3], 2))
                                        }
                                    ]
                                })
                            ext = pathlib.Path(baseName).suffix
                            baseName = baseName.split(ext)[0]
                            saveName = os.path.join(currentPath, baseName + ".dat")

                            with open(saveName, "w") as f2:
                                json.dump({"POLYGON_DATA": polygonData}, f2)

                        elif self.purposeType == "S":
                            for annoData in annoInfo:
                                position = []
                                try:
                                    segmentation = annoData["segmentation"][0]
                                except:
                                    continue

                                for i in range(0, len(segmentation), 2):
                                    position.append(
                                        {
                                            "X": float(round(segmentation[i], 2)),
                                            "Y": float(round(segmentation[i + 1], 2))
                                        }
                                    )

                                tmpId = (item for item in categories if item["TAG_CD"] == annoData["category_id"])
                                categoryInfo = next(tmpId, False)
                                polygonData.append({
                                    "DATASET_CD": self.datasetCd,
                                    "DATA_CD": dataCd,
                                    "TAG_CD": categoryInfo["TAG_CD"],
                                    "TAG_NAME": categoryInfo["TAG_NAME"],
                                    "COLOR": categoryInfo["COLOR"],
                                    "CLASS_CD": None,
                                    "CURSOR": "isPolygon",
                                    "NEEDCOUNT": -1,
                                    "POSITION": position
                                })

                            ext = pathlib.Path(baseName).suffix
                            baseName = baseName.split(ext)[0]
                            saveName = os.path.join(currentPath, baseName + ".dat")
                            # saveName = baseName + ".dat"

                            with open(saveName, "w") as f2:
                                json.dump({"POLYGON_DATA": polygonData}, f2)

                    except Exception as e2:
                        # print(e2)
                        errorFiles.append(baseName)
                        msg = str(traceback.format_exc())
                        continue

        except Exception as e:
            # print(e)
            print(traceback.format_exc())

        output = {
            "CLASS_INFO": categories,
            "ERROR_FILE": errorFiles,
            "MSG": msg
        }

        print(json.dumps(output))


if __name__ == "__main__":
    param = sys.argv[1]
    # param = '{"DATASET_CD":"D122131","PURPOSE_TYPE":"S","FILE_INFO":[{"FILE_PATH":"/Users/dmshin/onion/testdata/coco/COCO_val2014_000000025394.jpg","DATA_CD":"000001"}]}'
    param = json.loads(param)

    DI = datasetImporter(param)
    DI.coco2weda()