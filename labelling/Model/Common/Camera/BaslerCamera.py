import os
import cv2
from datetime import datetime
import json
import sys
import traceback
# Model 폴더 찾기 위한 방법
basePath = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir, "../")))
# Model Path 등록
sys.path.append(basePath)

from Output.PredictOutput import operator
from Common.Camera.BaseCamera import BaseCamera
from Network.TFRT.YOLOv3.yolov3 import TrtYOLOv3
from Common.Logger.Logger import logger
from Common.Utils.Utils import connectCamera
from Output.Output import sendMsg

log = logger("log")

try:
    import pycuda.driver as cuda
    cuda.init()
except ImportError as e:
    log.error(e)
    pass

try:
    from pypylon import pylon
except ImportError as e:
    log.error(e)
    pass

pid = os.getpid()


class Camera(BaseCamera):
    video_source = 0
    mdlPath = None
    serialNo = None
    frameRatio = None
    target = None
    svrIp = None
    svrPort = None
    hwPort = None
    hwType = None
    isCD = None
    isType = None

    def __init__(self, svrIp, svrPort, mdlPath, serialNo, frameRatio, target, hwType, isCD, isType, hwPort):
        Camera.mdlPath = mdlPath
        Camera.serialNo = serialNo
        Camera.frameRatio = int(frameRatio)
        Camera.svrIp = svrIp
        Camera.svrPort = svrPort
        Camera.hwType = hwType
        Camera.isCD = isCD
        Camera.isType = isType
        Camera.hwPort = hwPort

        if target != "":
            Camera.target = target

        else:
            Camera.target = None

        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        try:
            # cuda_ctx = cuda.Device(0).make_context()
            camera = None
            try:
                # basler camera bind
                camera, err = connectCamera(Camera.serialNo, width=1920, height=1680, exposureTime=8000)
                if err is None:
                    pass
                else:
                    log.error(err)
                converter = pylon.ImageFormatConverter()

                # converting to opencv bgr format
                converter.OutputPixelFormat = pylon.PixelType_BGR8packed
                converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            except Exception as e:
                out = {"IS_CD": Camera.isCD, "CODE": "040", "MSG": str(e)}
                log.error(out)
                log.error(traceback.format_exc())
                try:
                    _ = sendMsg('http://{}:{}/api/QI/report/stateCheck'.format(Camera.svrIp, Camera.svrPort), out)
                    # # cuda_ctx.pop()
                    return
                except Exception as ee:
                    log.error(ee)
                    log.error(traceback.format_exc())

            cnt = 0
            targets = []

            target = json.loads(Camera.target)
            captureTime = None
            for tmp in target:
                classCD = tmp["CLASS_CD"]
                className = tmp["CLASS_NAME"]
                dpLabel = tmp["DP_LABEL"]
                location = tmp["LOCATION"]
                color = tmp["COLOR"]
                accScope = tmp["ACC_SCOPE"].split(',')

                con1 = accScope[0]
                accu1 = accScope[1]
                logical = accScope[2]
                con2 = accScope[3]
                accu2 = accScope[4]

                targets.append([classCD, className, con1, accu1, logical, con2, accu2, dpLabel, location, color])

            if Camera.hwType == 'B':
                try:
                    while camera.IsGrabbing():
                        # read current frame
                        try:
                            grabResult = camera.RetrieveResult(20000, pylon.TimeoutHandling_ThrowException)

                        except Exception as e:
                            out = {"IS_CD": Camera.isCD, "CODE": "042", "MSG": str(e)}
                            _ = sendMsg('http://{}:{}/api/QI/report/stateCheck'.format(Camera.svrIp, Camera.svrPort), out)
                            # cuda_ctx.pop()
                            log.error(out)
                            log.error(traceback.format_exc())
                            return

                        # grab success check
                        if grabResult.GrabSucceeded():
                            # gray -> color change
                            image = converter.Convert(grabResult)
                            img = image.GetArray()
                            today = datetime.today()
                            captureTime = "{}-{}-{} {}:{}:{}.{}".format(today.year, today.month, today.day, today.hour,
                                                                        today.minute, today.second, today.microsecond)

                            out = {"HW_TYPE": Camera.hwType, "BOXES": None, "ACC": None, "CLASS": None,
                                   "isRect": False, "IS_CD": Camera.isCD, "IS_TYPE": Camera.isType, "RAW_TIME": captureTime}

                            # res = requests.post(server, headers=headers, data=json.dumps(out))
                            yield cv2.imencode('.jpg', img)[1].tobytes(), json.dumps(out)

                except Exception as e:
                    out = {"IS_CD": Camera.isCD, "CODE": "041", "MSG": str(e)}
                    _ = sendMsg('http://{}:{}/api/QI/report/stateCheck'.format(Camera.svrIp, Camera.svrPort), out)
                    log.error(out)
                    log.error(traceback.format_exc())
                    # cuda_ctx.pop()
                    return

            # elif Camera.hwType == 'F' or Camera.hwType == 'R':
            #     mdlList = os.listdir(Camera.mdlPath)
            #     # get model file and class file
            #     for tmp in mdlList:
            #         if ".tf" in tmp:
            #             tmp = tmp.split(".tf")
            #             modelName = tmp[0] + '.tf'
            #             break

            #     clssDirList = os.listdir(os.path.join(Camera.mdlPath, os.path.pardir))
            #     for tmp in clssDirList:
            #         if "_." in tmp:
            #             continue
            #         elif ".txt" in tmp or ".csv" in tmp:
            #             classPath = os.path.join(os.path.join(Camera.mdlPath, os.path.pardir), tmp)

            #     # get classes and numClasses in class file
            #     with open(classPath, 'r') as f:
            #         numClasses = (len(f.readlines()))

            #     classesTmp = [c.strip() for c in open(classPath).readlines()]
            #     classes = []

            #     for tmp in classesTmp:
            #         tmp = tmp.split(",")
            #         classes.append(tmp[1])

            #     modelPath = (os.path.join(Camera.mdlPath, modelName))
            #     try:
            #         trtYolov3 = TrtYOLOv3(modelPath, (416, 416), numClasses)
            #         # trtYolov3 = predictLoadModel("D", (416, 416, 3), Camera.isCD, modelPath)

            #     except Exception as e:
            #         out = {"IS_CD": Camera.isCD, "CODE": "031", "MSG": str(e)}
            #         _ = sendMsg('http://{}:{}/api/QI/report/stateCheck'.format(Camera.svrIp, Camera.svrPort), out)
            #         # cuda_ctx.pop()
            #         log.error(out)
            #         log.error(traceback.format_exc())
            #         return

            #     try:
            #         out = {"IS_CD": Camera.isCD, "CODE": "000", "MSG": None, "HW_PID": os.getpid()}
            #         _ = sendMsg('http://{}:{}/api/QI/report/stateCheck'.format(Camera.svrIp, Camera.svrPort), out)
            #         log.debug(out)

            #         while camera.IsGrabbing():
            #             # read current frame
            #             grabResult = camera.RetrieveResult(20000, pylon.TimeoutHandling_ThrowException)
            #             # grab success check
            #             if grabResult.GrabSucceeded():
            #                 # gray -> color change
            #                 image = converter.Convert(grabResult)
            #                 img = image.GetArray()
            #                 today = datetime.today()
            #                 capturedTime = "{}-{}-{} {}:{}:{}.{}".format(today.year, today.month, today.day, today.hour,
            #                                                              today.minute, today.second, today.microsecond)
            #                 if cnt % Camera.frameRatio == 0:
            #                     boxes, acc, clss = trtYolov3.detect(img, 0.3)
            #                     boxes = boxes.tolist()
            #                     if len(boxes) > 0:
            #                         acc = acc.tolist()
            #                         clss = clss.tolist()
            #                         for idx in clss:
            #                             if Camera.target is not None:
            #                                 for j, targetData in enumerate(targets):
            #                                     (classCD,
            #                                         className,
            #                                         con1,
            #                                         accu1,
            #                                         logical,
            #                                         con2,
            #                                         accu2,
            #                                         tag,
            #                                         location,
            #                                         color) = targetData

            #                                     if classes[idx] == className:
            #                                         for accuracy in acc:
            #                                             # out = roughExpress(boxes, accuracy, idx, targetData, capturedTime,
            #                                                               #  Camera.hwType, classes, Camera.isCD, Camera.isType)
            #                                               out = operator()
            #                                     else:
            #                                         for accuracy in acc:
            #                                             # out = roughExpress(boxes, accuracy, idx, targetData, capturedTime,
            #                                                               #  Camera.hwType, classes, Camera.isCD, Camera.isType)

            #                     else:
            #                         out = {"HW_TYPE": Camera.hwType, "BOXES": None, "ACC": None, "CLASS": None, "isRect": False,
            #                                "IS_CD": Camera.isCD, "IS_TYPE": Camera.isType, "RAW_TIME": capturedTime}

            #             else:
            #                 out = {"HW_TYPE": Camera.hwType, "BOXES": None, "ACC": None, "CLASS": None, "isRect": False,
            #                        "IS_CD": Camera.isCD, "IS_TYPE": Camera.isType, "RAW_TIME": capturedTime}

            #             # encode as a jpeg image and return it
            #             yield cv2.imencode('.jpg', img)[1].tobytes(), json.dumps(out)
            #             cnt += 1
                    # cuda_ctx.pop()

                except Exception as e:
                    out = {"IS_CD": Camera.isCD, "CODE": "040", "MSG": str(e)}
                    _ = sendMsg('http://{}:{}/api/QI/report/stateCheck'.format(Camera.svrIp, Camera.svrPort), out)
                    log.error(out)
                    log.error(traceback.format_exc())
                    # cuda_ctx.pop()

        except Exception as e:
            out = {"IS_CD": Camera.isCD, "CODE": "030", "MSG": str(e)}
            _ = sendMsg('http://{}:{}/api/QI/report/stateCheck'.format(Camera.svrIp, Camera.svrPort), out)
            log.error(out)
            log.error(traceback.format_exc())
            # cuda_ctx.pop()
            return
