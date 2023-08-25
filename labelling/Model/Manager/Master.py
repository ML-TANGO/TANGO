# -*- coding:utf-8 -*-
'''
Master Flask
1. /makeThumbnail: Thumbnail 만드는 API
2. /getFps: 동영상에서 fps 가져오는 API

'''
# etc lib
from os import wait
from signal import SIGTERM
import subprocess
import json
import os
import sys
import time
import signal
from time import sleep
import psutil
import requests
from subprocess import Popen, PIPE
from queue import Queue
from threading import Thread
from random import randint
import pathlib

import cv2

# flask lib
from flask import Flask
from flask_restful import Api
from flask import request, make_response
from flask import jsonify

import traceback


# Model 폴더 찾기 위한 방법
basePath = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# Model Path 등록
sys.path.append(basePath)

from Common.Utils.Utils import getConfig, getFps, makeDir, getGpuState, getUsableGPU, getTotalUsableGPU, setConfig
from Output.Output import sendMsg
from Common.Logger.Logger import logger
from Dataset.ImageProcessing import getThumbnail
from Predict.Tracker import tracker

srvIp, srvPort, logPath, tempPath, datasetPath, aiPath = getConfig()

log = logger("log")

headers = {"Content-Type": "application/json; charset=utf-8"}
app = Flask(__name__)
api = Api(app)


runPredictProc = []
realTimeProc = []
childChecker = True
port = None
pid = 0

# aiCd in trainer: AI_CD
# aiCd in QI: IS_CD


def killProcess(pid):
    log.debug("Start Remove {}".format(pid))
    waitTime = 10
    isDead = False
    try:
        while waitTime > 0:
            if not isDead:
                os.kill(pid, signal.SIGTERM)
                time.sleep(1)
                if psutil.pid_exists(pid):
                    waitTime = waitTime - 1
                else:
                    isDead = True
            else:
                break

        if not isDead:
            waitTime = 10
            while waitTime > 0:
                if not isDead:
                    os.kill(pid, signal.SIGKILL)
                    time.sleep(1)
                    if psutil.pid_exists(pid):
                        waitTime = waitTime - 1
                    else:
                        isDead = True
                else:
                    break
    except Exception as e:
        log.error(traceback.format_exc())
        log.error("Fail Kill Process {}".format(pid))
        isDead = False

    return isDead


def appendPrcList(prcList, element):
    # 중복 검사하는 로직 추가했다
    for v in prcList:
        if v["MDL_PATH"] == element["MDL_PATH"]:
            return False
    return True


def removePrcList(tgt, key, value):
    for v in tgt:
        if v[key] == value:
            tgt.remove(v)


def checkPrcMsg(msg):
    return msg.find('#_%')


def strToBool(s):
    return s == "True"


def sendToServer(url, msg, code):
    sendServer = 'http://{}:{}/api/{}'.format(srvIp, srvPort, url)
    print(f"\n== [SendMessage] {sendServer}")
    headers = {"Content-Type": "application/json; charset=utf-8"}
    out = {"STATUS": code, "MESSAGE": msg}
    requests.post(sendServer, headers=headers, data=json.dumps(out))
    return 1


def sendThread(sq):
    isDone = True
    while isDone:
        # pid, msg = sq.get()
        runPid, prcName, isErr, msg, url, isSend = sq.get()
        if msg == 'END':
            print(f"== Done {runPid}")
            isDone = False
        elif isErr:
            msg = {"PID": runPid, "PRC_NAME": prcName, "MSG": msg}
            sendToServer(url, msg, int(isErr))
            isDone = False
        else:
            if isSend == "1":
                sendToServer(url, msg, int(isErr))
    return 0


def prcThread(q, sq, runArgs, args, url):
    global runPredictProc
    try:
        runPid = -1
        runPrcName = ''
        isErr = False
        msg = ''

        item = []
        item.append("python")
        for ele in runArgs:
            item.append(ele)
        # 파일명, 전송URL,
        proc = Popen(item, stdin=PIPE, stdout=PIPE)
        proc.stdin.write(bytes(args, 'utf-8'))
        proc.stdin.write(bytes(' ', 'utf-8'))
        proc.stdin.close()

        errMsg = ''
        isFirst = True

        while True:
            outs = proc.stdout.readline().decode('utf-8')
            if len(outs) != 0:
                # print(outs, end='')
                txtIdx = checkPrcMsg(outs)
                if txtIdx >= 0:
                    outs = outs[txtIdx + 3:].splitlines()[0]
                    try:
                        prcPid, prcName, isErr, msg, isSend = outs.split('&G&')

                    except Exception as e:
                        log.error(traceback.format_exc())
                        log.error(outs)

                    isErr = strToBool(isErr)
                    if isFirst:
                        runPid = int(prcPid)
                        runPrcName = prcName
                        q.put(runPid)
                        isFirst = False
                    if isErr:
                        errMsg = errMsg + msg + "\n"
                        print(f"\n== [ErrorMessage] {outs}")
                        break
                    else:
                        sq.put([runPid, runPrcName, isErr, msg, url, isSend])
                        # 로깅
                        # 전송 이걸 어디로 보낼지
                        # sendToServer(url, msg, 0)
                        print(f'\n== [ProcessMessage{isSend}] {prcName}:{prcPid}')
            # else:
            #     time.sleep(1)

            if proc.poll() is not None:
                removePrcList(runPredictProc, "PID", runPid)
                # errMsg = {"PID": runPid, "PRC_NAME": runPrcName, "MSG": "Process Killed"}
                # sq.put([runPid, runPrcName, True, errMsg, 'binary/binErrorHandle', "1"])
                break

        if isErr:
            try:
                # 로깅
                # 에러 전송
                print(f"[Error][ProcessKill] killing [{runPid}] ....")
                # errMsg = {"PID": runPid, "PRC_NAME": runPrcName, "MSG": errMsg}
                sq.put([runPid, runPrcName, True, errMsg, 'binary/binErrorHandle', "1"])

                # os.kill(runPid, signal.SIGTERM)
                killProcess(runPid)
                # os.wait()
                removePrcList(runPredictProc, "PID", runPid)
                print(f"[Error][ProcessKill] [{runPid}] dead")
            except ProcessLookupError:
                pass
            except Exception as e:
                print("==========Process still alive==============")
                print(e)
                log.error(traceback.format_exc())
        return runPid

    except Exception as e:
        print(e)
        log.error(traceback.format_exc())


def runProcess(runArgs, args, url):
    try:
        q = Queue()
        sq = Queue()
        proc = Thread(target=prcThread, args=(q, sq, runArgs, args, url))
        proc.start()
        procSend = Thread(target=sendThread, args=(sq,))
        procSend.start()

        return q.get()

    except Exception as e:
        print(e)
        log.error(traceback.format_exc())


# getThumbnail
@app.route('/makeThumbnail', methods=['POST'])
def makeThumbnail():
    log.info("START make Thumbnail")
    req = request.json
    req = req[0]

    errorFiles = []
    out = {"STATUS": 1}
    total_num = len(req)
    for current_num, data in enumerate(req, start=1):
        mdlType = data["MDL_TYPE"].lower()
        imgPath = data["PATH"]
        savePath = data["SAVE_PATH"]
        try:
            fps, err = getThumbnail(mdlType, imgPath, savePath)
            if err["MSG"] is not None:
                raise Exception
            out = {"STATUS": 1, "FPS": fps}
            log.info(f"[{current_num}/{total_num}] Thumbnail creation completed. Path={savePath}")

        except Exception as e:
            log.error(e)
            log.error(err)
            log.error(traceback.format_exc())
            errorFiles.append(imgPath)

    if len(errorFiles) > 0:
        return make_response(jsonify({"STATUS": 0, "MSG": "CREATE THUMBNAIL FAIL", "ERROR_FILE": errorFiles}))
    else:
        return make_response(jsonify(out))


@app.route('/check', methods=['POST'])
def healthChecker():
    req = request.json

    for i, data in enumerate(req):
        checkIsCd = data["IS_CD"]
        checkPid = int(data["PID"])
        srvIP = data["SRV_IP"]
        srvPort = data["SRV_PORT"]

        try:
            p = psutil.Process(checkPid).as_dict()
            procStatus = p["status"]

            if procStatus == 'zombie':
                out = {"IS_CD": checkIsCd, "CODE": "033", "MSG": "zombie"}
                os.kill(checkPid, signal.SIGTERM)
                os.wait()

            else:
                out = {"IS_CD": checkIsCd, "CODE": "000", "MSG": None, "HW_PID": checkPid}

        except Exception as e:
            out = {"IS_CD": checkIsCd, "CODE": "030", "MSG": str(e)}
            log.error(traceback.format_exc())

        _ = sendMsg('http://{}:{}/api/QI/report/stateCheck'.format(srvIP, srvPort), out)

    return make_response(jsonify({"STATE": 1}))


# getFPS
@app.route('/getFps', methods=['POST'])
def getFPS():
    try:
        log.info("START get FPS")
        req = request.json
        req = req[0]
        out = []

        for data in req:
            videoPath = data["PATH"]
            fileName = data["FILE_NAME"]
            fps = getFps(videoPath)
            out.append({"PATH": videoPath, "FILE_NAME": fileName, "FPS": fps})

        # print(json.dumps(out))

        log.info("Finish get FPS")
        return make_response(jsonify(out))

    except Exception as e:
        log.error(e)
        log.error(traceback.format_exc())
        return make_response(jsonify({"STATUS": 0, "MSG": str(e)}))


# getGPUStatus
@app.route('/getGpuStatus', methods=['POST'])
def gpuStat():
    output = getGpuState()
    return make_response(jsonify(output))


@app.route('/setMastConfig', methods=['POST'])
def setMastConfig():
    try:
        global srvIp, srvPort, logPath, tempPath, datasetPath, aiPath
        req = request.json[0]
        base = "../Server/src"
        configPath = req["CONFIG_PATH"]
        base = base + configPath
        setConfig(base)
        srvIp, srvPort, logPath, tempPath, datasetPath, aiPath = getConfig()
        log.info("Change Config File [{}]".format(configPath))
        return make_response(jsonify({"STATUS": 1}))
    except Exception as e:
        return make_response(jsonify({"STATUS": 0, "MSG": str(e)}))


@app.route('/setLogLevel', methods=['POST'])
def setLogLevel():
    try:
        req = request.json[0]
        global logLevel
        logLevel = req["LOG_LEVEL"]
        with open(os.path.join(basePath, "Common/Logger/log.conf"), "w") as f:
            f.write(logLevel)
        return make_response(jsonify({"STATUS": 1}))
    except Exception as e:
        return make_response(jsonify({"STATUS": 0, "MSG": str(e)}))

# ------------------------------------------ #
# -----------------AI_PROJECT--------------- #
# ------------------------------------------ #


# runTracker
@app.route('/tracker', methods=['POST'])
def runTracker():
    try:
        log.info("START Tracker")
        req = request.json

        req = req[0]
        videoPath = req["FILE_PATH"]
        startFrame = req["START_FRAME"]
        endFrame = req["END_FRAME"]
        TrackerInfo = req["TRACKER_INFO"]

        out = tracker(videoPath, startFrame, endFrame, TrackerInfo)
        log.info("FINISH Tracker")

        return make_response(jsonify(out))

    except Exception as e:
        log.error(e)
        log.error(traceback.format_exc())
        return make_response(jsonify({"STATUS": 0, "MSG": str(e)}))


@app.route('/getUsableGpu', methods=['POST'])
def getUsableGpu():
    usableGpu = getTotalUsableGPU()
    return jsonify(usableGpu)


# autoLabling
@app.route('/modelLoad', methods=['POST'])
def modelLoad():
    global runPredictProc
    port = 0
    pid = -1
    runMiniPredictorMode = "M"
    req = request.json[0]
    AI_CD = req["AI_CD"]
    MDL_PATH = req["MDL_PATH"]
    EPOCH = req["EPOCH"]

    # 현재 돌고 있는 모델인지 점검
    for procInfo in runPredictProc:
        if procInfo["MDL_PATH"] == MDL_PATH and procInfo["EPOCH"] == EPOCH:
            runMiniPredictorMode = "P"  # predict mode
            port = procInfo["PORT"]
            pid = procInfo["PID"]
            break

    print(runMiniPredictorMode)

    if runMiniPredictorMode == "M":
        # 모델 올리기
        port = randint(20000, 20050)

        for procInfo in runPredictProc:
            if procInfo["PORT"] == port:
                port = randint(20000, 20050)

        gpuState = getGpuState()
        processName = "Manager/Worker2.py"
        if len(gpuState) == 0:
            gpuIdx = -1
        else:
            gpuIdx = getUsableGPU(gpuState)

        runArgs = [os.path.join(basePath, processName), str(port), str(gpuIdx)]
        pid = runProcess(runArgs, json.dumps(req), "binary/predictBinLog")

        loadFlask = False
        reTryCount = 0
        while loadFlask is False:
            try:
                reTryCount = reTryCount + 1
                childServer = "http://0.0.0.0:{}/chekFlask".format(port)
                res = requests.post(childServer, headers=headers, data={})
                res = res.json()
                if res["STATUS"] == 1:
                    loadFlask = True
                    break
                if reTryCount > 60:
                    pid = 0
                    break
            except Exception:
                time.sleep(1)
                pass
        addPrc = ""
        if pid != 0:
            addPrc = {"AI_CD": AI_CD, "PID": pid, "PORT": port, "MDL_PATH": MDL_PATH, "EPOCH": EPOCH}
            if appendPrcList(runPredictProc, addPrc):
                runPredictProc.append(addPrc)

    return jsonify({"PID": pid})


# autoLabling
@app.route('/autoLabeling', methods=['POST'])
def runAutoLabeling():
    global runPredictProc
    port = -1
    pid = -1
    try:
        req = request.json[0]
        MDL_PATH = req["MDL_PATH"]
        AI_CD = req["AI_CD"]
        EPOCH = req["EPOCH"]
        URL = req["URL"]
        for procInfo in runPredictProc:
            if procInfo["MDL_PATH"] == MDL_PATH and procInfo["EPOCH"] == EPOCH:
                port = procInfo["PORT"]
                pid = procInfo["PID"]
                break
        log.info("Start Auto Label [{}][{}]".format(pid, port))
        res = ""
        # try:
        childServer = "http://0.0.0.0:{}/runAutoLabeling".format(port)
        res = requests.post(childServer, headers=headers, data=json.dumps(req))
        # except Exception as e:
        #     print("============================================")
        #     log.error(traceback.format_exc())
        #     msg = {"PID": pid, "PRC_NAME": "AutoLabeling", "MSG": "request Fail"}
        #     sendToServer("binary/binErrorHandle", msg, 1)

        dataType = req["DATA_TYPE"]
        resultInfo = []
        # print(res.json())
        for resData in res.json():
            tags = []
            filePath = resData["IMAGE_PATH"]
            extension = pathlib.Path(filePath).suffixes
            saveJsonPath = filePath.replace(extension[len(extension) - 1], ".dat")

            datasetCd = resData["DATASET_CD"]
            dataCd = resData["DATA_CD"]
            labels = resData["LABELS"]
            totalFrame = resData["TOTAL_FRAME"]

            for labelInfo in labels:
                if "label" in labelInfo:
                    tags.append({
                        "CLASS_DB_NM": labelInfo["label"],
                        "COLOR": labelInfo["COLOR"],
                        "ACCURACY": labelInfo["ACCURACY"]
                    })
            if dataType == "I":
                fps = 0
            if dataType == "V":
                fps = getFps(filePath)

            resultInfo.append({
                "FILE_PATH": saveJsonPath,
                "IMAGE_PATH": filePath,
                "DATASET_CD": datasetCd,
                "DATA_CD": dataCd,
                "TAGS": tags,
                "BASE_MDL": AI_CD,
                "TOTAL_FRAME": totalFrame,
                "FPS": fps
            })

        res = {"RESULT": resultInfo}
        sendServer = "http://{}:{}/api/binary/{}".format(srvIp, srvPort, URL)
        res = requests.post(sendServer, headers=headers, data=json.dumps(res))

        res = res.json()

        os.kill(pid, signal.SIGTERM)
        os.wait()
        removePrcList(runPredictProc, "PID", pid)
        log.info("[{}] AUTOLABELING DONE.".format(datasetCd))
        return jsonify(res)

    except ChildProcessError as e2:
        pass

    except Exception as e:
        msg = {"PID": pid, "PRC_NAME": "AutoLabeling", "MSG": "request Fail"}
        sendToServer("binary/binErrorHandle", msg, 1)
        log.error(traceback.format_exc())
        log.error("[{}] AUTOLABELING Fail.(error code : {}".format(datasetCd, e))
        log.info("error In Auto Label")
        log.error("error In Auto Label")


# miniPredictor
@app.route('/miniPredictor', methods=['POST'])
def runMiniPredictor():
    try:
        global runPredictProc
        port = -1
        req = request.json[0]

        print("MASTER REQ:", req)
        MDL_PATH = req["MDL_PATH"]
        EPOCH = req["EPOCH"]

        req = json.dumps(req)
        print("dump req : ", req)

        for procInfo in runPredictProc:
            if procInfo["MDL_PATH"] == MDL_PATH and procInfo["EPOCH"] == EPOCH:
                port = procInfo["PORT"]
                break
        print(port)

        childServer = "http://0.0.0.0:{}/runMiniPredictor".format(port)
        res = requests.post(childServer, headers=headers, data=req)
        res = res.json()

        return jsonify(res)

    except Exception as e:
        print(e)
        log.error(traceback.format_exc())


# procList
@app.route('/procList', methods=['POST'])
def procList():
    return jsonify(runPredictProc)


@app.route('/startTrain', methods=['POST'])
def runTrain():
    req = request.json[0]
    print(req)
    START_EPOCH = req["START_EPOCH"]
    GPU_IDX = json.dumps(req["GPU_IDX"])
    # MDL_PATH = req["MDL_PATH"]
    # 등록 로직 추가할까 고민중.
    # gpuState = getGpuState()
    processName = "Train/Trainer2.py"

    runArgs = [os.path.join(basePath, processName), str(GPU_IDX), str(START_EPOCH)]
    pid = runProcess(runArgs, json.dumps(req), "binary/trainBinLog")
    return str(pid)


@app.route('/killProc', methods=['POST'])
def stopTrain():
    req = request.json[0]
    PID = req["PID"]

    if PID != -1:
        result = killProcess(PID)
        log.debug("Stop Train Result {}".format(result))
    else:
        result = False
        log.debug("You Try Kill Process -1 ")
    return str(result)


@app.route('/runRealtimePredictor', methods=['POST'])
def runRealTimePredictor():
    global realTimeProc
    out = {}
    req = request.json
    req = req[0]
    try:
        pid = None
        # req = '[{"HW_INFO":{"IS_CD":54,"FEED_URL":"http://192.168.0.33:20052/video_feed",},"MDL_INFO":[{"OBJECT_TYPE":"D","MDL_PATH":"/Users/upload/InputSources/54/model/0168","NETWORK_NAME":"EFFICIENTDET","TARGET_CLASS":[{"OUT_CD":62,"IS_CD":54,"CLASS_CD":1312,"CLASS_NAME":"scratch","DP_LABEL":"SCR","COLOR":"#000fff","ACC_SCOPE":"gteq,0.01,none,,","LOCATION":"S","HW_CD":null}]}]}]'
        processName = "Predict/RealTimePredictor2.py"

        if req["STATE"] == "START":
            isStart = False
            for idx, realTimeProcData in enumerate(realTimeProc):
                if realTimeProcData["IS_CD"] == req["IS_CD"]:
                    pid = realTimeProcData["PID"]
                    isStart = True

            if not isStart:
                runArgs = [os.path.join(basePath, processName)]
                pid = runProcess(runArgs, json.dumps(req), "binary/predictBinLog")
                realTimeProc.append({"IS_CD": req["IS_CD"], "PID": pid, "STATUS": req["STATE"]})

            out = {"status": 1, "IS_CD": req["IS_CD"], "PID": pid, "PORT": None}

        elif req["STATE"] == "STOP":
            isStop = False
            for realTimeProcData in realTimeProc:
                if realTimeProcData["IS_CD"] == req["IS_CD"]:
                    os.kill(realTimeProcData["PID"], signal.SIGTERM)
                    os.wait()
                    removePrcList(realTimeProc, "IS_CD", req["IS_CD"])
                    out = {"status": 1, "IS_CD": req["IS_CD"], "PID": None, "PORT": None}
                    isStop = True
                    break

            if not isStop:
                out = {"status": 1, "IS_CD": req["IS_CD"], "PID": None, "PORT": None}

    except OSError as e2:
        removePrcList(realTimeProc, "IS_CD", req["IS_CD"])
        out = {"status": 1, "IS_CD": req["IS_CD"], "PID": None, "PORT": None}

    except Exception as e:
        out = {"status": 0, "IS_CD": req["IS_CD"], "PID": None, "PORT": None, "MSG": str(e)}

    finally:
        return jsonify(out)


# Train Tabular data
# input : parameters(server)
# output : pid
@app.route('/startTabTrain', methods=['POST'])
def runTabTrain():
    req = request.json
    print(req)
    processName = "Train/TrainManager.py"
    pid = None

    runArgs = [os.path.join(basePath, processName)]
    pid = runProcess(runArgs, json.dumps(req), "binary/trainBinLog")

    if pid is not None:
        output = {"STATUS": True, "MSG": None, "PID": pid}
    else:
        output = {"STATUS": False, "MSG": "TrainManager cannot started", "PID": pid}
    return jsonify(output)


if __name__ == "__main__":
    try:
        if len(sys.argv) >= 2:
            port = sys.argv[1]
        else:
            port = 5638

        app.run(host='0.0.0.0', port=port)

    except Exception as e:
        out = {"IS_CD": None, "CODE": "120", "MSG": str(e)}
        print(out)
        log.error(traceback.format_exc())
        # res = requests.post(sendServer2, headers=headers, data=json.dumps(out))
