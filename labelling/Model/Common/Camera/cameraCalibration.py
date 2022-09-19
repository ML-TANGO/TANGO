"""
CameraCalibration.py
"""
# basler camera lib
try:
    from pypylon import pylon
except ImportError:
    pass

# etc lib
import cv2
import os
import numpy as np
import sys
import json
import requests
import traceback

# Model 폴더 찾기 위한 방법
basePath = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir, "../")))
# Model Path 등록
sys.path.append(basePath)

from Common.Utils.Utils import connectCamera
from Common.Logger.Logger import logger
log = logger("log")


# find contour
def findContours(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thre = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    cnts, hierarchy = cv2.findContours(thre, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    second = largest = -float('inf')
    realContour = None

    for c in cnts:
        area = cv2.contourArea(c)
        if area > largest:
            second = largest
            largest = area
            realContour = c

        elif second < area < largest:
            second = area
            realContour = c

    x, y, w, h = cv2.boundingRect(realContour)
    ppm = (w) / 50
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return img, ppm


def shift(lst, value):
    for i in range(1, len(lst)):
        lst[i - 1] = lst[i]
    lst[len(lst) - 1] = value
    return lst


if __name__ == '__main__':
    try:
        data = sys.argv[1]
        data = json.loads(data)

        isCd = data["IS_CD"]
        srvIp = data["SRV_IP"]
        srvPort = data["SRV_PORT"]
        serial = data["CAM_SERIAL"]
        sendServer = 'http://{}:{}/api/QI/hwsetting/camCali'.format(srvIp, srvPort)
        headers = {"Content-Type": "application/json; charset=utf-8"}
        camera, err = connectCamera(serial, width=800, height=800, exposureTime=1000)

        if err is None and camera is not None:
            converter = pylon.ImageFormatConverter()
            # converting to opencv bgr format
            converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            cnt = 0
            ppms = []

            while camera.IsGrabbing():
                grabResult = camera.RetrieveResult(20000, pylon.TimeoutHandling_ThrowException)

                if grabResult.GrabSucceeded():
                    image = converter.Convert(grabResult)
                    frame = image.GetArray()
                    frame, ppm = findContours(frame)
                    ppms.append(ppm)
                    cnt += 1
                    if cnt >= 10:
                        ppms = shift(ppms, ppm)
                        std = np.std(ppms)
                        if std <= 0.02:
                            break

            # send ppms[0]
            out = {"IS_CD": isCd, "PPM": ppms[0]}
            res = requests.post(sendServer, headers=headers, data=json.dumps(out))

        elif err is not None and camera is None:
            sendServer = 'http://{}:{}/api/QI/report/stateCheck'.format(srvIp, srvPort)
            headers = {"Content-Type": "application/json; charset=utf-8"}
            out = {"IS_CD": isCd, "CODE": "040", "MSG": str(err)}
            log.error(out)
            log.error(err)
            res = requests.post(sendServer, headers=headers, data=json.dumps(out))

    except Exception as e:
        sendServer = 'http://{}:{}/api/QI/report/stateCheck'.format(srvIp, srvPort)
        headers = {"Content-Type": "application/json; charset=utf-8"}
        out = {"IS_CD": isCd, "CODE": "040", "MSG": str(e)}
        log.error(traceback.format_exc())
        res = requests.post(sendServer, headers=headers, data=json.dumps(out))
