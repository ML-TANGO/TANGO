#!/usr/bin/env python
from importlib import import_module
import os, sys
from flask import Flask, render_template, Response, request
import time
import json

import traceback
import requests

'''
# import camera driver
#if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from camera import Camera
'''
basePath = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir)))

sys.path.append(basePath)

print(basePath)

from Common.Utils.Utils import makeDir
from Common.Camera.BaslerCamera import Camera
# Raspberry Pi camera module (requires picamera package)
# from camera_pi import Camera

app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen(camera):
    """Video streaming generator function."""
    while True:
        frame, rectData = camera.get_frame()
        # time.sleep(1)
        yield (
                b'--frame\r\n'
                b'Content-Type:image/jpeg\r\n'
                b'Content-Length: ' + f"{len(frame)}".encode() + b'\r\n'
                b'Data: ' + f"{rectData}".encode() + b'\r\n' 
                b'\r\n' + frame + b'\r\n'
        )


def getParams():
    data = json.loads(sys.argv[1])
    hwPort = int(json.loads(sys.argv[2]))
    # data = sys.argv[1]
    svrIp = data["SRV_IP"]
    svrPort = data["SRV_PORT"]
    mdlPath = data["MDL_PATH"]
    serialNo = data["SERIAL_NUMBER"]
    frameRatio = data["FRAME_RATIO"]
    target = data["TARGET"]
    hwType = data["MODE"]
    isCD = data["IS_CD"]
    isType = data["IS_TYPE"]
    return svrIp, svrPort, mdlPath, serialNo, frameRatio, target, hwType, isCD, isType, hwPort


svrIp, svrPort, mdlPath, serialNo, frameRatio, target, hwType, isCD, isType, hwPort = getParams()

print(svrIp, svrPort, mdlPath, serialNo, frameRatio, target, hwType, isCD, isType, hwPort)


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    imgData = Camera(svrIp, svrPort, mdlPath, serialNo, frameRatio, target, hwType, isCD, isType, hwPort)
    return Response(gen(imgData), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    out = {}
    sendServer = 'http://{}:{}/api/QI/report/stateCheck'.format(svrIp, svrPort)
    headers = {"Content-Type": "application/json; charset=utf-8"}

    try:
        out = {"IS_CD": isCD, "CODE": "000", "MSG": None, "HW_PID": os.getpid()}
        res = requests.post(sendServer, headers=headers, data=json.dumps(out))
        app.run(host='0.0.0.0', port=hwPort, threaded=True)

    except Exception as err:
        print("ROUGH DETECTOR ERROR : ", traceback.print_exc())
        out = {"IS_CD": isCD, "CODE": "030", "MSG": str(err)}
        res = requests.post(sendServer, headers=headers, data=json.dumps(out))

    finally:
        out = {"IS_CD": isCD, "CODE": "030", "MSG": "STOP Rough Detector Server"}
        res = requests.post(sendServer, headers=headers, data=json.dumps(out))
