import usb.core
import usb.util
import os

import time

from flask import Flask
from flask_restful import Api
from flask import request
from flask import jsonify
import flask

from threading import Thread

import json
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from Common.Logger import Logger

app = Flask(__name__)
api = Api(app)

isLight = False
lightCnt = 0
buffer = [0x0] * 8
endPoint = 0x01

with open("./lightConfig.json", "r") as f:
    jsonData = json.load(f)

SEC = int(jsonData["SEC"])
beep = jsonData["BEEP"]

log = Logger.logger("log")

dev = usb.core.find(idVendor=0x04d8, idProduct=0xe73c)

if dev.is_kernel_driver_active(0):
    reattch = True
    dev.detach_kernel_driver(0)

dev.set_configuration()

def setLight(error):
    global lightCnt
    global buffer
    global isLight
    global SEC
    global beep
    
    while True:
        lightCnt += 1
        if lightCnt >= SEC or error == 0:
            buffer[2] = 0 
            buffer[7] = 0 

            rtn = dev.write(endPoint, buffer)
            dev.reset()

            isLight = False
            lightCnt = 0 
            break
     
        else:
            # light on
            buffer[2] = error
            # sound on
            buffer[7] = error if beep else 0

            rtn = dev.write(endPoint, buffer)
            dev.reset()
            isLight = True

        time.sleep(1)

# run Light using user script
@app.route('/', methods=['POST'])
def runLight():
    global isLight
    global lightCnt
    global buffer
    global endPoint
    global loop

    global dev
    output = {"status": True}

    req = request.json
    req = json.loads(req)
    log.debug(req)

    error = 1 if req["error"] == True else 0

    buffer[0] = 0x57

    if not isLight:
        #loop.run_until_complete(setLight(error))
        thread = Thread(target=setLight, args=[error])
        thread.start()
 
    return jsonify(output)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3131)
