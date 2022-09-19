# -*- coding:utf-8 -*-
'''
Tracker Function
'''

import cv2
import os
import sys
import time
import traceback

# Model 폴더 찾기 위한 방법
basePath = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# Model Path 등록
sys.path.append(basePath)

from Common.Logger.Logger import logger
log = logger("log")


def tracker(videoPath, startFrame, endFrame, TrackerInfo):
    log.debug(TrackerInfo)
    try:
        vc = cv2.VideoCapture(videoPath)
        if startFrame is None:
            startFrame = 0

        # startFrame 으로 이동
        vc.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
        # startFrame 위치의 이미지 1장 가져오기 : tracker 대상 추출하기 위해
        ret, frame = vc.read()

        output = {}
        result = []
        datasetCd = None

        st = time.time()
        for trackData in TrackerInfo:
            datasetCd = trackData["DATASET_CD"]
            resultTmp = []
            cnt = startFrame
            x1 = int(trackData["POSITION"][0]["X"])
            y1 = int(trackData["POSITION"][0]["Y"])
            x2 = int(trackData["POSITION"][1]["X"]) - x1
            y2 = int(trackData["POSITION"][1]["Y"]) - y1

            bboxes = (x1, y1, x2, y2)

        # tracker = cv2.TrackerTLD_create()
        # tracker = cv2.Tracker_create()

        tracker = cv2.legacy.TrackerTLD_create()
        # bboxes = cv2.selectROI(frame, False)
        box = bboxes

        while True:
            bbox = []
            if cnt == endFrame:
                break

            ret, frame = vc.read()
            _ = tracker.init(frame, bboxes)

            (success, box) = tracker.update(frame)

            print(cnt, success)

            if success:
                (x1, y1, x2, y2) = [int(v) for v in box]
                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if x2 < 0:
                    x2 = 0
                if y2 < 0:
                    y2 = 0
                bbox.append([{"X": x1, "Y": y1}, {"X": x2 + x1, "Y": y2 + y1}])
            log.debug(len(bbox))
            if len(bbox) != 0:
                resultTmp.append({"FRAME_NUMBER": cnt, "CLASS_CD": trackData["CLASS_CD"],
                                  "COLOR": trackData["COLOR"], "CURSOR": trackData["CURSOR"],
                                  "DATASET_CD": trackData["DATASET_CD"], "DATA_CD": trackData["DATA_CD"],
                                  "POSITION": bbox[0], "TAG_CD": trackData["TAG_CD"],
                                  "TAG_NAME": trackData["TAG_NAME"], "NEEDCOUNT": trackData["NEEDCOUNT"]})
            cnt += 1

        result.append(resultTmp)

        output = {"DATASET_CD": datasetCd, "START_FRAME": startFrame,
                  "END_FRAME": endFrame, "FILE_PATH": videoPath, "TRACKER_INFO": result}

        log.info("TRAKER TIME is {}".format(time.time() - st))
        return output

    except Exception as e:
        log.error(traceback.format_exc())
        log.error(e)
