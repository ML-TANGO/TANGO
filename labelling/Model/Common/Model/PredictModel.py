# -*- coding:utf-8 -*-
'''
OBJECT_TYPE 별 모델 로드
'''
import os
import sys
import json
import traceback
import requests

import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras import layers

# Model 폴더 찾기 위한 방법
basePath = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir, "../")))
# Model Path 등록
sys.path.append(basePath)


from Common.Logger.Logger import logger
from Common.Utils.Utils import getConfig
from Common.Process.Process import prcSendData, prcErrorData, prcLogData

# classification AutoML
from Network.AUTOKERAS.AutoML import autoML

# detection Yolov3

from Network.TF.YOLOv3 import models as yv3
from Network.TF.YOLOv4 import models as yv4
from Network.TF.TinyYOLOv3 import models as tyv3
from Network.TF.YOLOv3.utils import freeze_all

# SEGMENTATION
from Network.KERAS.DeepLab import deeplab as dpl
from Network.KERAS.EfficientDet import efficientdet_keras
from Network.KERAS.EfficientDet import hparams_config
from Network.KERAS.Unet import Unet as unet


log = logger("log")
srvIp, srvPort, logPath, tempPath, datasetPath, aiPath = getConfig()
server = "http://{}:{}/api/binary/predictBinLog".format(srvIp, srvPort)
headers = {'Content-Type': 'application/json; charset=utf-8'}


def getJson(saveClassFile):

    with open(saveClassFile, "r") as f:
        jsonData = json.load(f)

    modelName = jsonData["NETWORK_NAME"]
    imgSize = jsonData["IMG_INFO"]["IMG_SIZE"]
    channel = jsonData["IMG_INFO"]["IMG_CHANNEL"]

    if modelName == "DEEP-LAB":
        if channel == 1:
            prcLogData("Cannot Use grayscale in segmentation!")
        inputShape = (imgSize, imgSize, 3)

    elif modelName == "EFFICIENTDET-SEG":
        if channel == 1:
            prcLogData("Cannot Use grayscale in segmentation!")
        inputShape = (512, 512, 3)

    elif modelName == "YOLOV3" or modelName == "EFFICIENTDET" or modelName == "YOLOV4":
        if channel == 1:
            prcLogData("Cannot Use grayscale in detection!")
        inputShape = (416, 416, 3)

    else:
        inputShape = (imgSize, imgSize, channel)

    return inputShape, modelName


def predictLoadModel(objectType, aiCd, mdlPath):
    try:
        model = None
        # classification
        if objectType == "C":
            log.debug(mdlPath)
            newMdlPath = os.path.abspath(os.path.join(mdlPath, '..'))
            classesPath = os.path.join(newMdlPath, 'classes.json')
            inputShape, modelName = getJson(classesPath)

            loadModel = tf.keras.models.load_model(mdlPath)
            log.info("model load success")

            model = loadModel

        # detection
        elif objectType == "D":
            if aiCd == "YOLOV3":
                lenClasses = 80
                modelName = 'YOLOV3'
                inputShape = (416, 416, 3)

            elif aiCd == "YOLOV4":
                lenClasses = 80
                modelName = 'YOLOV4'
                inputShape = (416, 416, 3)

            elif aiCd == "EFFICIENTDET":
                lenClasses = 80
                modelName = 'EFFICIENTDET'
                inputShape = (512, 512, 3)

            else:
                newMdlPath = os.path.abspath(os.path.join(mdlPath, '..'))

                classesPath = '{}/classes.names'.format(newMdlPath)
                with open(classesPath, 'r') as f:
                    lenClasses = len(f.readlines())

                classesPath = os.path.join(newMdlPath, 'classes.json')
                inputShape, modelName = getJson(classesPath)

            log.debug("Predict Model LenClasses = {}".format(lenClasses))
            if modelName == 'YOLOV3':
                if lenClasses == 1:
                    lenClasses += 1
                model = yv3.YoloV3(classes=lenClasses)
                mdlPath = os.path.join(mdlPath, "{}.tf".format(modelName))
                model.load_weights(mdlPath)

            elif modelName == 'YOLOV4':
                if lenClasses == 1:
                    lenClasses += 1
                model = yv4.YoloV4(classes=lenClasses)
                mdlPath = os.path.join(mdlPath, "{}.tf".format(modelName))
                model.load_weights(mdlPath)

            else:
                mdlPath = os.path.join(mdlPath, "{}.tf".format(modelName))
                log.debug(mdlPath)

                config = hparams_config.get_detection_config("efficientdet-d0")
                config.nms_configs.score_thresh = 0.4
                config.nms_configs.max_output_size = 100
                config.image_size = "512x512"
                config.heads = ['object_detection']
                model = efficientdet_keras.EfficientDetModel(config=config)
                model.build((None, 512, 512, 3))
                model.load_weights(mdlPath)

                class ExportModel(tf.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model

                    @tf.function
                    def f(self, imgs):
                        return self.model(imgs, training=False, post_mode='global')

                model = ExportModel(model)

        # segmentation
        elif objectType == "S":
            if aiCd == "DEEP-LAB":
                lenClasses = 21
                modelName = "DEEP-LAB"
                inputShape = (512, 512, 3)
                mdlPath = os.path.join(
                    mdlPath, '{}.h5'.format(modelName)
                )

                model = dpl.Deeplabv3(
                    classes=lenClasses,
                    input_shape=inputShape,
                    backbone="xception"
                )

            else:
                newMdlPath = os.path.abspath(os.path.join(mdlPath, '..'))

                classesPath = '{}/classes.names'.format(newMdlPath)
                classes = [c.strip() for c in open(classesPath).readlines()]
                lenClasses = len(classes)

                classesPath = '{}/classes.json'.format(newMdlPath)
                inputShape, modelName = getJson(classesPath)

                if modelName == "DEEP-LAB":
                    inputShape = (512, 512, 3)
                    mdlPath = os.path.join(mdlPath, '{}.h5'.format(modelName))
                    model = dpl.Deeplabv3(
                        classes=lenClasses,
                        input_shape=inputShape,
                        backbone="xception"
                    )

                elif modelName == "U-NET":
                    mdlPath = os.path.join(mdlPath, '{}.tf'.format(modelName))
                    model = unet.unet(
                        input_shape=(inputShape[0], inputShape[1]),
                    )

                else:
                    mdlPath = os.path.join(mdlPath, '{}.tf'.format(modelName))
                    config = hparams_config.get_efficientdet_config('efficientdet-d0')
                    config.heads = ['segmentation']
                    config.seg_num_classes = lenClasses
                    model = efficientdet_keras.EfficientDetNet(config=config)
                    model.build((1, 512, 512, 3))

            model.load_weights(mdlPath)

        else:
            log.error("CHECK OBJECT_TYPE!")
            output = {"STATUS": 0, "code": str("CHECK OBJECT_TYPE!")}
            # _ = requests.post(server, headers=headers, data=json.dumps(output))

        return model, modelName, inputShape

    except Exception as e:
        log.error(e)
        log.error(traceback.format_exc())
        output = {"STATUS": 0, "code": str(e)}
        # _ = requests.post(server, headers=headers, data=json.dumps(output))
