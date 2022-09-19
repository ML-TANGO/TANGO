#!/usr/bin/env python
# -*- coding:utf-8 -*-

# flask lib
from flask import Flask
from flask_restful import Resource, Api
from flask import request, jsonify, abort, make_response
from random import randint

# file object save lib
from werkzeug.utils import secure_filename

try:
    # basler camera lib
    from pypylon import pylon
except:
    # raise ImportError("pypylon is not installed")
    pass

# etc lib
import subprocess, psutil
import requests
import json, os, time, sys
import signal
import shutil
import traceback
# unzip lib
import zipfile as zf

app = Flask(__name__)
api = Api(app)

procList = []

# Model 폴더 찾기 위한 방법
basePath = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# Model Path 등록
sys.path.append(basePath)

from Common.Utils.Utils import makeDir


def sendMsg(server, msg):
    headers = {"Content-Type": "application/json; charset=utf-8"}
    res = requests.post(server, headers=headers, data=json.dumps(msg))
    return res


def removePrcList(isCD):
    global procList
    for v in procList:
        if v["IS_CD"] == isCD:
            procList.remove(v)


@app.route('/jetsonSetting', methods=['POST'])
def jetsonSetting():
    try:
        hwType = request.form["HW_TYPE"]
        camSerial = request.form["CAM_SERIAL"]
        svrIp = request.form["IP"]
        svrPort = request.form["PORT"]
        hwCD = request.form["HW_CD"]

        mdlFile = request.files["MDL_FILE"] if hwType != 'B' else None
        settingFilePath = os.path.join(basePath, "Jetson/settings/{}".format(hwCD))
        isMakeFile = makeDir(settingFilePath)
        modelSavePath = None

        if mdlFile is not None:
            mdlFile.save(os.path.join(settingFilePath, secure_filename(mdlFile.filename)))
            if isMakeFile:
                # unzip model file
                zipFilePath = os.path.join(settingFilePath, mdlFile.filename)
                zipfile = zf.ZipFile(zipFilePath)
                modelSavePath = os.path.join(settingFilePath, "model")
                zipfile.extractall(modelSavePath)
                zipfile.close()

                modelPathList = os.listdir(modelSavePath)
                for path in modelPathList:
                    if "__MACOSX" in path:
                        continue
                    if os.path.isdir(os.path.join(modelSavePath, path)):
                        modelSavePath = os.path.join(modelSavePath, path)

                modelPathList = os.listdir(modelSavePath)
                for path in modelPathList:
                    if "__MACOSX" in path:
                        continue
                    if os.path.isdir(os.path.join(modelSavePath, path)):
                        modelSavePath = os.path.join(modelSavePath, path)

                if os.path.isfile(zipFilePath):
                    os.remove(zipFilePath)

        saveJson = {
            "HW_TYPE": hwType,
            "CAM_SERIAL": camSerial,
            "SRV_IP": svrIp,
            "SRV_PORT": svrPort,
            "HW_CD": hwCD,
            "MDL_PATH": modelSavePath
        }

        with open(os.path.join(settingFilePath, "setting_{}.json".format(hwCD)), "w") as f:
            json.dump(saveJson, f)
            out = {"status": 1}

        return make_response(jsonify(out))

    except Exception as e:
        out = {"ERROR": e}
        return jsonify(out)


@app.route('/jetsonSettingUpdate', methods=['POST'])
def settingUpdate():
    try:
        out = {}
        isMakeFile = False
        resCode = None
        modelSavePath = None
        hwType = request.form["HW_TYPE"]
        camSerial = request.form["CAM_SERIAL"]
        svrIp = request.form["IP"]
        svrPort = request.form["PORT"]
        hwCD = request.form["HW_CD"]

        print(hwType, camSerial, svrIp, svrPort, hwCD)

        dirPath = os.path.join(basePath, "Jetson/settings/{}".format(hwCD))
        with open(os.path.join(dirPath, "setting_{}.json".format(hwCD)), "r") as f:
            tmp = json.load(f)

        if request.form["UPT_TYPE"].upper() == "MDL":
            preModelPath = os.path.join(dirPath, "model")
            shutil.rmtree(preModelPath, ignore_errors=True)

            mdlFile = request.files["MDL_FILE"] if hwType != "B" else None

        modelSavePath = tmp["MDL_PATH"] if request.form["UPT_TYPE"].upper() == "NONE" else None

        if mdlFile is not None:
            mdlFile.save(os.path.join(dirPath, secure_filename(mdlFile.filename)))
            if isMakeFile:
                # unzip model file
                zipFilePath = os.path.join(dirPath, mdlFile.filename)
                zipfile = zf.ZipFile(zipFilePath)
                modelSavePath = os.path.join(dirPath, "model")
                zipfile.extractall(modelSavePath)
                zipfile.close()

                modelPathList = os.listdir(modelSavePath)

                for path in modelPathList:
                    if "__MACOSX" in path:
                        continue
                    if os.path.isdir(os.path.join(modelSavePath, path)):
                        modelSavePath = os.path.join(modelSavePath, path)

                modelPathList = os.listdir(modelSavePath)
                for path in modelPathList:
                    if "__MACOSX" in path:
                        continue
                    if os.path.isdir(os.path.join(modelSavePath, path)):
                        modelSavePath = os.path.join(modelSavePath, path)

                if os.path.isfile(zipFilePath):
                    os.remove(zipFilePath)

        saveJson = {
            "HW_TYPE": hwType,
            "CAM_SERIAL": camSerial,
            "SRV_IP": svrIp,
            "SRV_PORT": svrPort,
            "HW_CD": hwCD,
            "MDL_PATH": modelSavePath
        }

        with open(os.path.join(dirPath, "setting_{}.json".format(hwCD)), "w") as f:
            json.dump(saveJson, f)
            out = {"status": 1}
            resCode = 200

        return jsonify(out), resCode

    except Exception as e:
        out = {"ERROR": e}
        return jsonify(out)


@app.route('/startChild', methods=['POST'])
def startChild():
    req = request.json
    
    out = {}
    isCD = None
    global procList
    isRun = False
    try:
        print("REQ :{} ".format(req))
        idx = 0
        childStateOut = []
        checkISCD = 0
        proc = None
        for i, data in enumerate(req):
            state = data["STATE"].upper()
            hwCD = str(data["HW_CD"])
            hwPort = data["HW_PORT"]
            isCD = data["IS_CD"]
            serverIP = data["SRV_IP"]
            serverPort = str(data["SRV_PORT"])
            isType = data["IS_TYPE"]

            for procData in procList:
                if procData["IS_CD"] == isCD:
                    out = {
                        "status": 1,
                        "IS_CD": procData["IS_CD"],
                        "PID": procData["PID"],
                        "PORT": procData["PORT"]}

                    isRun = True
                    break

            if not isRun:
                settingFilePath = os.path.join(basePath, "Jetson/settings/{}".format(hwCD))

                with open(os.path.join(settingFilePath, "setting_{}.json".format(hwCD)), "r") as f:
                    tmp = json.load(f)

                serialNumber = str(tmp["CAM_SERIAL"])
                mode = tmp["HW_TYPE"]
                postPID = tmp["PID"] if "PID" in tmp else None
                mdlPath = None
                if mode != "B":
                    mdlPath = tmp["MDL_PATH"] if "MDL_PATH" in tmp else None

                if state == "START":
                    target = data["TARGET_CLASS"]
                    frameRatio = str(data["FRAME_RATIO"])
                    target = json.dumps(target) if target is not None else ""
                    tmp["IS_CD"] = isCD
                    tmp["IS_TYPE"] = isType
                    tmp["HW_PORT"] = hwPort
                    tmp["HW_CD"] = hwCD
                    tmp["TARGET"] = target
                    tmp["FRAME_RATIO"] = frameRatio

                    inputData = {
                        "SRV_IP": serverIP,
                        "SRV_PORT": serverPort,
                        "HW_PORT": hwPort,
                        "MDL_PATH": mdlPath,
                        "SERIAL_NUMBER": serialNumber,
                        "FRAME_RATIO": frameRatio,
                        "TARGET": target,
                        "MODE": mode,
                        "IS_CD": isCD,
                        "IS_TYPE": isType
                    }
                    inputData = json.dumps(inputData)
                    if (postPID is not None) and (postPID is not -1):
                        if psutil.pid_exists(postPID):
                            os.kill(postPID, signal.SIGTERM)
                            os.wait()
                            tmp["PID"] = -1
                        else:
                            tmp["PID"] = -1

                    port = randint(20000, 20500)
                    for procData in procList:
                        if procData["PORT"] == port:
                            port = randint(20000, 20500)

                    processName = "./TxChild.py"
                    runArgs = [processName, str(inputData), str(port)]
                    proc = subprocess.Popen(runArgs)

                    if psutil.pid_exists(proc.pid):
                        tmp["PID"] = proc.pid
                        tmp["STATE"] = "START"
                        tmp["IS_SUCCESS"] = True
                    else:
                        try:
                            removePrcList(isCD)

                        except ValueError:
                            continue
                        tmp["PID"] = -1
                        tmp["STATE"] = "STOP"
                        tmp["IS_SUCCESS"] = False

                    procList.append({
                        "IS_CD": isCD,
                        "PORT": port,
                        "PID": proc.pid,
                        "STATE": tmp["STATE"],
                        "IS_SUCCESS": tmp["IS_SUCCESS"]
                    })
                    out = {"status": 1, "IS_CD": isCD, "PID": proc.pid, "PORT": port}

                elif state == "STOP":
                    isProc = False
                    for procData in procList:
                        if isCD == procData["IS_CD"]:
                            os.kill(procData["PID"], signal.SIGTERM)
                            os.wait()
                            procList.remove(procData)
                            out = {"status": 1, "IS_CD": isCD, "PID": None, "PORT": None}
                            isProc = True
                            break

                    if not isProc:
                        out = {"status": 1, "IS_CD": isCD, "PID": None, "PORT": None}

    except OSError as oe:
        out = {"status": 0, "IS_CD": isCD, "CODE": "032", "MSG": str(oe)}
    except ValueError as ve:
        out = {"status": 0, "IS_CD": isCD, "CODE": "032", "MSG": str(ve)}
    except Exception as e:
        out = {"status": 0, "IS_CD": isCD, "CODE": "020", "MSG": str(e)}

    finally:
        return jsonify(out)

    # sendServer = 'http://{}:{}/api/QI/report/stateCheck'.format(serverIP, serverPort)
    # _ = sendMsg(sendServer, out)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5637)