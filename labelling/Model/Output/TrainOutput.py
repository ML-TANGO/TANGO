# -*- coding:utf-8 -*-
'''
epoch print, server 전송 스크립트

1. classificationState:classification 출력
2. yoloState:Yolo 출력
3. retinaState:Retinanet 출력
'''
import time
import os
import sys
import tensorflow as tf
import json
import math

# Model 폴더 찾기 위한 방법
pathA = os.path.join(os.path.dirname(__file__), os.path.pardir, "../")
pathB = os.path.join(os.path.join(pathA))
basePath = os.path.abspath(pathB)
# Model Path 등록
sys.path.append(basePath)

from Common.Logger.Logger import logger
from Common.Utils.Utils import getConfig
from Common.Process.Process import prcSendData
from DatasetLib import DatasetLib
from Output.Output import sendMsg

# from Common.Process.Process import prcClose
# from Common.Process.Process import prcLogData


srvIp, srvPort, logPath, tempPath, datasetPath, aiPath = getConfig()
server = 'http://{}:{}/api/binary/trainBinLog'.format(srvIp, srvPort)
headers = {"Content-Type": "application/json; charset=utf-8"}
log = logger("log")


def getTime(t):
    t = time.gmtime(t)
    return "{}-{}-{} {}:{}:{}".format(
        t.tm_year,
        t.tm_mon,
        t.tm_mday,
        t.tm_hour,
        t.tm_min,
        t.tm_sec)


def getOutput(o, epoch, loss, acc, vloss, vacc, path, onTrainEnd, last, remTime):
    return {
        "AI_CD": o.AI_CD,
        "EPOCH": epoch + 1 if not onTrainEnd else epoch,
        "UPT_DTM": o.et,
        "OBJECT_TYPE": o.OBJECT_TYPE,
        "DATA_TYPE": o.DATA_TYPE,
        "GPU_RATE": o.GPU_RATE,
        "CRN_USR": o.CRN_USR,
        "AI_LOSS": loss if not math.isnan(loss) else -1,
        "AI_ACC": acc if not math.isnan(acc) else -1,
        "AI_VAL_LOSS": vloss if not math.isnan(vloss) else -1,
        "AI_VAL_ACC": vacc if not math.isnan(vacc) else -1,
        "MDL_PATH": path,
        "NETWORK_NAME": o.NETWORK_NAME,
        "IS_LAST": last,
        "REMANING_TIME": remTime
    }


class classificationState(tf.keras.callbacks.Callback):
    def __init__(
        self,
        AI_CD,
        OBJECT_TYPE,
        DATA_TYPE,
        GPU_RATE,
        CRN_USR,
        IS_AUTO,
        NETWORK_NAME,
        EPOCH,
        savePath
    ):
        self.batch_losses = []
        self.batch_acc = []
        self.st = 0
        self.epoch = EPOCH
        self.cur_epoch = 0
        self.AI_CD = AI_CD
        self.OBJECT_TYPE = OBJECT_TYPE
        self.DATA_TYPE = DATA_TYPE
        self.GPU_RATE = GPU_RATE * 100
        self.CRN_USR = CRN_USR
        self.isAuto = IS_AUTO
        self.et = 0
        self.output = None
        self.NETWORK_NAME = NETWORK_NAME
        self.savePath = savePath
        self.preOutput = None
        self.fstEpochStartTime = 0
        self.fstEpochEndTime = 0
        self.remTime = 0


    def on_epoch_begin(self, epoch, logs=None):
        self.st = time.time()
        self.cur_epoch = epoch

        if epoch == 0:
            self.fstEpochStartTime = time.time()

        if self.preOutput is not None:
            prcSendData(__file__, json.dumps(self.preOutput))
            log.info("[{}] Result epoch={}/{}, acc={}, loss={}, val_acc={}, val_loss={}".format(
                self.AI_CD,
                epoch,
                self.epoch,
                self.preOutput["AI_ACC"],
                self.preOutput["AI_LOSS"],
                self.preOutput["AI_VAL_ACC"],
                self.preOutput["AI_VAL_LOSS"]))

        log.debug("[{}] Training epoch={}/{}".format(
            self.AI_CD,
            epoch + 1,
            self.epoch))

    def on_batch_end(self, batch, logs=None):
        log.debug("[{}] Training batch={}/{}:{}, acc={}, loss={}".format(
            self.AI_CD,
            self.cur_epoch,
            self.epoch,
            batch + 1,
            float("{0:.4f}".format(logs["accuracy"])),
            float("{0:.4f}".format(logs["loss"]))))

    def on_epoch_end(self, epoch, logs=None):
        self.et = time.time() - self.st

        if epoch == 0:
            self.fstEpochEndTime = time.time() - self.fstEpochStartTime

        self.remTime = self.fstEpochEndTime * (self.epoch - epoch + 1)

        tmp = '{0:04d}'.format(epoch + 1)
        MDL_PATH = os.path.join(self.AI_CD, tmp)

        loss = float("{0:.4f}".format(logs["loss"])) if "loss" in logs else 0
        acc = float("{0:.4f}".format(logs["accuracy"])) if "accuracy" in logs else 0
        valAcc = float("{0:.4f}".format(logs["val_accuracy"])) if "val_accuracy" in logs else 0
        valLoss = float("{0:.4f}".format(logs["val_loss"])) if "val_loss" in logs else 0

        self.output = getOutput(
            self,
            epoch,
            loss,
            acc,
            valLoss,
            valAcc,
            MDL_PATH,
            False,
            False,
            self.remTime
        )

        self.preOutput = self.output

    def on_train_end(self, epoch, logs=None):
        tmp = '{0:04d}'.format(int(self.epoch))
        MDL_PATH = os.path.join(aiPath, self.AI_CD, tmp)

        loss = float("{0:.4f}".format(epoch["loss"])) if "loss" in epoch else 0
        acc = float("{0:.4f}".format(epoch["accuracy"])) if "accuracy" in epoch else 0
        valAcc = float("{0:.4f}".format(epoch["val_accuracy"])) if "val_accuracy" in epoch else 0
        valLoss = float("{0:.4f}".format(epoch["val_loss"])) if "val_loss" in epoch else 0

        self.output = getOutput(
            self,
            self.epoch,
            loss,
            acc,
            valLoss,
            valAcc,
            MDL_PATH,
            True,
            True,
            self.remTime
        )

        if self.output is not None:
            prcSendData(__file__, json.dumps(self.output))
            log.info("[{}] Result epoch={}/{}, acc={}, loss={}, val_acc={}, val_loss={}".format(
                self.AI_CD,
                self.epoch,
                self.epoch,
                self.output["AI_ACC"],
                self.output["AI_LOSS"],
                self.output["AI_VAL_ACC"],
                self.output["AI_VAL_LOSS"]))

        log.debug("[{}] Training epoch={}/{}".format(
            self.AI_CD,
            self.epoch,
            self.epoch))

        # self.output = getOutput(
        #     self,
        #     self.epoch,
        #     loss,
        #     acc,
        #     valLoss,
        #     valAcc,
        #     MDL_PATH,
        #     True,
        #     True
        # )


class yoloState(tf.keras.callbacks.Callback):
    def __init__(
        self,
        AI_CD,
        OBJECT_TYPE,
        DATA_TYPE,
        GPU_RATE,
        CRN_USR,
        NETWORK_NAME,
        EPOCH,
        savePath
    ):
        self.batch_losses = []
        self.batch_acc = []
        self.st = 0
        self.epoch = EPOCH
        self.cur_epoch = 0
        self.AI_CD = AI_CD
        self.OBJECT_TYPE = OBJECT_TYPE
        self.DATA_TYPE = DATA_TYPE
        self.GPU_RATE = GPU_RATE * 100
        self.CRN_USR = CRN_USR
        self.et = 0
        self.output = None
        self.NETWORK_NAME = NETWORK_NAME
        self.savePath = savePath
        self.preOutput = None
        self.lossMaxThreshold = 1000000
        self.fstEpochStartTime = 0
        self.fstEpochEndTime = 0
        self.remTime = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.st = time.time()
        self.cur_epoch = epoch

        if epoch == 0:
            self.fstEpochStartTime = time.time()

        if self.preOutput is not None:
            prcSendData(__file__, json.dumps(self.preOutput))
            log.info("[{}] Result epoch={}/{}, acc={}, loss={}, val_acc={}, val_loss={}".format(
                self.AI_CD,
                epoch,
                self.epoch,
                self.preOutput["AI_ACC"],
                self.preOutput["AI_LOSS"],
                self.preOutput["AI_VAL_ACC"],
                self.preOutput["AI_VAL_LOSS"]))

        log.debug("[{}] Training epoch={}/{}".format(
            self.AI_CD,
            epoch + 1,
            self.epoch))

    def on_batch_end(self, batch, logs=None):
        if "yolo_output_0_accuracy" in logs:
            outAcc1 = float(logs["yolo_output_0_accuracy"])
        if "yolo_output_1_accuracy" in logs:
            outAcc2 = float(logs["yolo_output_1_accuracy"])
        if "yolo_output_2_accuracy" in logs:
            outAcc3 = float(logs["yolo_output_2_accuracy"])
        # acc = max(outAcc1, outAcc2, outAcc3)
        acc = outAcc1

        log.debug("[{}] Training batch={}/{}:{}, acc={}, loss={}".format(
            self.AI_CD,
            self.cur_epoch,
            self.epoch,
            batch + 1,
            float("{0:.4f}".format(acc)),
            float("{0:.4f}".format(logs["loss"]))))

    def on_epoch_end(self, epoch, logs=None):
        self.et = time.time() - self.st

        if epoch == 0:
            self.fstEpochEndTime = time.time() - self.fstEpochStartTime

        self.remTime = self.fstEpochEndTime * (self.epoch - epoch + 1)

        tmp = '{0:04d}'.format(epoch + 1)
        MDL_PATH = os.path.join(self.AI_CD, tmp)

        outAcc1 = float(logs["yolo_output_0_accuracy"]) if "yolo_output_0_accuracy" in logs else 0
        outAcc2 = float(logs["yolo_output_1_accuracy"]) if "yolo_output_1_accuracy" in logs else 0
        outAcc3 = float(logs["yolo_output_2_accuracy"]) if "yolo_output_2_accuracy" in logs else 0
        outValAcc1 = float(logs["val_yolo_output_0_accuracy"]) if "val_yolo_output_0_accuracy" in logs else 0
        outValAcc2 = float(logs["val_yolo_output_1_accuracy"]) if "val_yolo_output_1_accuracy" in logs else 0
        outValAcc3 = float(logs["val_yolo_output_2_accuracy"]) if "val_yolo_output_2_accuracy" in logs else 0

        # acc = max(outAcc1, outAcc2, outAcc3)
        # val_acc = max(outValAcc1, outValAcc2, outValAcc3)

        acc = max(outAcc1, outAcc2, outAcc3)
        val_acc = max(outValAcc1, outValAcc2, outValAcc3)
        loss = 0
        valLoss = 0

        if "loss" in logs:
            loss = float("{0:.4f}".format(logs["loss"]))
        if "val_loss" in logs:
            valLoss = float("{0:.4f}".format(logs["val_loss"]))

        valAcc = float("{0:.4f}".format(val_acc))
        acc = float("{0:.4f}".format(acc))

        self.output = getOutput(
            self,
            epoch,
            loss if loss < self.lossMaxThreshold else self.lossMaxThreshold,
            acc,
            valLoss if valLoss < self.lossMaxThreshold else self.lossMaxThreshold,
            valAcc,
            MDL_PATH,
            False,
            False,
            self.remTime
        )
        self.preOutput = self.output

    def on_train_end(self, epoch, logs=None):
        tmp = '{0:04d}'.format(int(self.epoch))
        MDL_PATH = os.path.join(aiPath, self.AI_CD, tmp)
        outAcc1 = float(epoch["yolo_output_0_accuracy"]) if "yolo_output_0_accuracy" in epoch else 0
        outAcc2 = float(epoch["yolo_output_1_accuracy"]) if "yolo_output_1_accuracy" in epoch else 0
        outAcc3 = float(epoch["yolo_output_2_accuracy"]) if "yolo_output_2_accuracy" in epoch else 0
        outValAcc1 = float(epoch["val_yolo_output_0_accuracy"]) if "val_yolo_output_0_accuracy" in epoch else 0
        outValAcc2 = float(epoch["val_yolo_output_1_accuracy"]) if "val_yolo_output_1_accuracy" in epoch else 0
        outValAcc3 = float(epoch["val_yolo_output_2_accuracy"]) if "val_yolo_output_2_accuracy" in epoch else 0

        # acc = max(outAcc1, outAcc2, outAcc3)
        # val_acc = max(outValAcc1, outValAcc2, outValAcc3)

        acc = max(outAcc1, outAcc2, outAcc3)
        val_acc = max(outValAcc1, outValAcc2, outValAcc3)
        loss = 0
        valLoss = 0

        if "loss" in epoch:
            loss = float("{0:.4f}".format(epoch["loss"]))
        if "val_loss" in epoch:
            valLoss = float("{0:.4f}".format(epoch["val_loss"]))

        valAcc = float("{0:.4f}".format(val_acc))
        acc = float("{0:.4f}".format(acc))

        self.output = getOutput(
            self,
            self.epoch,
            loss if loss < self.lossMaxThreshold else self.lossMaxThreshold,
            acc,
            valLoss if valLoss < self.lossMaxThreshold else self.lossMaxThreshold,
            valAcc,
            MDL_PATH,
            True,
            True,
            0
        )

        if self.output is not None:
            prcSendData(__file__, json.dumps(self.output))
            log.info("[{}] Result epoch={}/{}, acc={}, loss={}, val_acc={}, val_loss={}".format(
                self.AI_CD,
                self.epoch,
                self.epoch,
                self.output["AI_ACC"],
                self.output["AI_LOSS"],
                self.output["AI_VAL_ACC"],
                self.output["AI_VAL_LOSS"]))

        log.debug("[{}] Training epoch={}/{}".format(
            self.AI_CD,
            self.epoch,
            self.epoch))

        # self.output = getOutput(
        #     self,
        #     self.epoch,
        #     loss,
        #     acc,
        #     valLoss,
        #     valAcc,
        #     MDL_PATH,
        #     True,
        #     True
        # )


class efficientDetState(tf.keras.callbacks.Callback):
    def __init__(
        self,
        AI_CD,
        OBJECT_TYPE,
        DATA_TYPE,
        GPU_RATE,
        CRN_USR,
        NETWORK_NAME,
        EPOCH,
        savePath
    ):
        self.batch_losses = []
        self.batch_acc = []
        self.st = 0
        self.epoch = EPOCH
        self.cur_epoch = 0
        self.AI_CD = AI_CD
        self.OBJECT_TYPE = OBJECT_TYPE
        self.DATA_TYPE = DATA_TYPE
        self.GPU_RATE = GPU_RATE * 100
        self.CRN_USR = CRN_USR
        self.et = 0
        self.output = None
        self.NETWORK_NAME = NETWORK_NAME
        self.savePath = savePath
        self.preOutput = None
        self.fstEpochStartTime = 0
        self.fstEpochEndTime = 0
        self.remTime = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.st = time.time()
        self.cur_epoch = epoch
        if epoch == 0:
            self.fstEpochStartTime = time.time()

        if self.preOutput is not None:
            prcSendData(__file__, json.dumps(self.preOutput))

            log.info("[{}] Result epoch={}/{}, acc={}, loss={}, val_acc={}, val_loss={}".format(
                self.AI_CD,
                epoch,
                self.epoch,
                self.preOutput["AI_ACC"],
                self.preOutput["AI_LOSS"],
                self.preOutput["AI_VAL_ACC"],
                self.preOutput["AI_VAL_LOSS"]))

        log.debug("[{}] Training epoch={}/{}".format(
            self.AI_CD,
            epoch + 1,
            self.epoch))

    def on_batch_end(self, batch, logs=None):
        acc = 0.0
        log.debug("[{}] Training batch={}/{}:{}, acc={}, loss={}".format(
            self.AI_CD,
            self.cur_epoch,
            self.epoch,
            batch + 1,
            float("{0:.4f}".format(acc)),
            float("{0:.4f}".format(logs["loss"]))))

    def on_epoch_end(self, epoch, logs=None):
        self.et = time.time() - self.st
        tmp = '{0:04d}'.format(epoch + 1)
        MDL_PATH = os.path.join(self.AI_CD, tmp)

        # acc = max(outAcc1, outAcc2, outAcc3)
        # val_acc = max(outValAcc1, outValAcc2, outValAcc3)

        acc = 0
        val_acc = 0
        loss = 0
        valLoss = 0

        if "loss" in logs:
            loss = float("{0:.4f}".format(logs["loss"]))
        if "val_loss" in logs:
            valLoss = float("{0:.4f}".format(logs["val_loss"]))

        valAcc = float("{0:.4f}".format(val_acc))
        acc = float("{0:.4f}".format(acc))

        self.output = getOutput(
            self,
            epoch,
            loss,
            acc,
            valLoss,
            valAcc,
            MDL_PATH,
            False,
            False,
            self.remTime
        )

        self.preOutput = self.output

    def on_train_end(self, epoch, logs=None):
        self.et = time.time() - self.st
        tmp = '{0:04d}'.format(int(self.epoch))
        MDL_PATH = os.path.join(self.AI_CD, tmp)

        acc = 0
        val_acc = 0
        loss = 0
        valLoss = 0

        if "loss" in epoch:
            loss = float("{0:.4f}".format(epoch["loss"]))
        if "val_loss" in epoch:
            valLoss = float("{0:.4f}".format(epoch["val_loss"]))

        valAcc = float("{0:.4f}".format(val_acc))
        acc = float("{0:.4f}".format(acc))

        self.output = getOutput(
            self,
            self.epoch,
            loss,
            acc,
            valLoss,
            valAcc,
            MDL_PATH,
            True,
            True,
            0
        )

        if self.output is not None:
            prcSendData(__file__, json.dumps(self.output))
            log.info("[{}] Result epoch={}/{}, acc={}, loss={}, val_acc={}, val_loss={}".format(
                self.AI_CD,
                self.epoch,
                self.epoch,
                self.output["AI_ACC"],
                self.output["AI_LOSS"],
                self.output["AI_VAL_ACC"],
                self.output["AI_VAL_LOSS"]))

        log.debug("[{}] Training epoch={}/{}".format(
            self.AI_CD,
            self.epoch,
            self.epoch))

        # self.output = getOutput(
        #     self,
        #     self.epoch,
        #     loss,
        #     acc,
        #     valLoss,
        #     valAcc,
        #     MDL_PATH,
        #     True,
        #     True
        # )


class deepLabState(tf.keras.callbacks.Callback):
    def __init__(
        self,
        AI_CD,
        OBJECT_TYPE,
        DATA_TYPE,
        GPU_RATE,
        CRN_USR,
        NETWORK_NAME,
        EPOCH,
        savePath
    ):
        self.batch_losses = []
        self.batch_acc = []
        self.st = 0
        self.epoch = EPOCH
        self.cur_epoch = 0
        self.AI_CD = AI_CD
        self.OBJECT_TYPE = OBJECT_TYPE
        self.DATA_TYPE = DATA_TYPE
        self.GPU_RATE = GPU_RATE * 100
        self.CRN_USR = CRN_USR
        self.et = 0
        self.output = None
        self.NETWORK_NAME = NETWORK_NAME
        self.savePath = savePath
        self.preOutput = None
        self.fstEpochStartTime = 0
        self.fstEpochEndTime = 0
        self.remTime = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.st = time.time()
        self.cur_epoch = epoch

        if epoch == 0:
            self.fstEpochStartTime = time.time()

        tmp = '{0:04d}'.format(epoch + 1)
        MDL_PATH = os.path.join(aiPath, self.AI_CD, tmp)
        if not os.path.isdir(MDL_PATH):
            os.makedirs(MDL_PATH, exist_ok=True)

        if self.preOutput is not None:
            prcSendData(__file__, json.dumps(self.preOutput))
            log.info("[{}] Result epoch={}/{}, acc={}, loss={}, val_acc={}, val_loss={}".format(
                self.AI_CD,
                epoch,
                self.epoch,
                self.preOutput["AI_ACC"],
                self.preOutput["AI_LOSS"],
                self.preOutput["AI_VAL_ACC"],
                self.preOutput["AI_VAL_LOSS"]))

        log.debug("[{}] Training epoch={}/{}".format(
            self.AI_CD,
            epoch + 1,
            self.epoch))

    def on_batch_end(self, batch, logs=None):
        log.debug("[{}] Training batch={}/{}:{}, acc={}, loss={}".format(
            self.AI_CD,
            self.cur_epoch,
            self.epoch,
            batch + 1,
            float("{0:.4f}".format(logs["accuracy"])),
            float("{0:.4f}".format(logs["loss"]))))

    def on_epoch_end(self, epoch, logs=None):
        self.et = time.time() - self.st

        if epoch == 0:
            self.fstEpochEndTime = time.time() - self.fstEpochStartTime

        self.remTime = self.fstEpochEndTime * (self.epoch - epoch + 1)

        tmp = '{0:04d}'.format(epoch + 1)
        MDL_PATH = os.path.join(aiPath, self.AI_CD, tmp)
        loss = 0
        acc = 0
        valLoss = 0
        valAcc = 0
        log.info(logs)

        if "loss" in logs:
            loss = float("{0:.4f}".format(logs["loss"])) if "loss" in logs else 0
        if "accuracy" in logs:
            acc = float("{0:.4f}".format(logs["accuracy"])) if "accuracy" in logs else 0
        if "val_loss" in logs:
            valLoss = float("{0:.4f}".format(logs["val_loss"])) if "val_loss" in logs else 0
        if "accuracy" in logs:
            valAcc = float("{0:.4f}".format(logs["val_accuracy"])) if "val_accuracy" in logs else 0

        self.output = getOutput(
            self,
            epoch,
            loss,
            acc,
            valLoss,
            valAcc,
            MDL_PATH,
            False,
            False,
            self.remTime
        )
        self.preOutput = self.output

    def on_train_end(self, epoch, logs=None):
        tmp = '{0:04d}'.format(int(self.epoch))
        MDL_PATH = os.path.join(aiPath, self.AI_CD, tmp)
        loss = 0
        acc = 0
        valLoss = 0
        valAcc = 0
        if "loss" in epoch:
            loss = float("{0:.4f}".format(epoch["loss"]))
        if "accuracy" in epoch:
            acc = float("{0:.4f}".format(epoch["accuracy"]))
        if "val_loss" in epoch:
            valLoss = float("{0:.4f}".format(epoch["val_loss"]))
        if "accuracy" in epoch:
            valAcc = float("{0:.4f}".format(epoch["val_accuracy"]))

        self.output = getOutput(
            self,
            self.epoch,
            loss,
            acc,
            valLoss,
            valAcc,
            MDL_PATH,
            True,
            True,
            0
        )

        if self.output is not None:
            prcSendData(__file__, json.dumps(self.output))
            log.info("[{}] Result epoch={}/{}, acc={}, loss={}, val_acc={}, val_loss={}".format(
                self.AI_CD,
                self.epoch,
                self.epoch,
                self.output["AI_ACC"],
                self.output["AI_LOSS"],
                self.output["AI_VAL_ACC"],
                self.output["AI_VAL_LOSS"]))

        log.debug("[{}] Training epoch={}/{}".format(
            self.AI_CD,
            self.epoch,
            self.epoch))

        # self.output = getOutput(
        #     self,
        #     self.epoch,
        #     loss,
        #     acc,
        #     valLoss,
        #     valAcc,
        #     MDL_PATH,
        #     True,
        #     True
        # )


# tabnet Output
class tabnetState(tf.keras.callbacks.Callback):
    def __init__(self, param, saveMdlPath):
        self.dataLib = DatasetLib.DatasetLib()
        self.AI_CD = param["SERVER_PARAM"]["AI_CD"]
        self.MDL_IDX = param["MDL_IDX"]
        self.MODEL_NAME = param["MODEL_NAME"]
        self.saveMdlPath = saveMdlPath

        self.epoch = param["epochs"]
        self.cur_epoch = 0

        self.param = param
        self.output = None
        self.preOutput = None

        self.fstEpochStartTime = 0
        self.fstEpochEndTime = 0
        self.remTime = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.st = time.time()
        self.cur_epoch = epoch

        if epoch == 0:
            self.fstEpochStartTime = time.time()

        if self.preOutput is not None and "SRV_ADDR" in self.preOutput:
            _ = sendMsg(self.preOutput["SRV_ADDR"], self.preOutput["SEND_DATA"])
            if self.param["OBJECT_TYPE"] == "C":
                log.debug("[{}] ({}: {}) Result epoch={}/{}, acc={}, loss={}, val_acc={}, val_loss={}".format(
                    self.AI_CD,
                    self.MDL_IDX,
                    self.MODEL_NAME,
                    epoch,
                    self.epoch,
                    self.preOutput["SEND_DATA"]["AI_ACC"],
                    self.preOutput["SEND_DATA"]["AI_LOSS"],
                    self.preOutput["SEND_DATA"]["AI_VAL_ACC"],
                    self.preOutput["SEND_DATA"]["AI_VAL_LOSS"]))

            elif self.param["OBJECT_TYPE"] == "R":
                log.debug("[{}] ({}: {}) Result epoch={}/{}, loss={}, mse={}".format(
                    self.AI_CD,
                    self.MDL_IDX,
                    self.MODEL_NAME,
                    epoch,
                    self.epoch,
                    self.preOutput["SEND_DATA"]["AI_LOSS"],
                    self.preOutput["SEND_DATA"]["MSE"]))

        log.debug("[{}] Training epoch={}/{}".format(
            self.AI_CD,
            epoch + 1,
            self.epoch))

    def on_batch_end(self, batch, logs=None):
        if self.param["OBJECT_TYPE"] == "C":
            log.debug("[{}] ({}: {}) Training batch={}/{}:{}, acc={}, loss={}".format(
                self.AI_CD,
                self.MDL_IDX,
                self.MODEL_NAME,
                self.cur_epoch,
                self.epoch,
                batch + 1,
                float("{0:.4f}".format(logs["accuracy"] if "accuracy" in logs else 0)),
                float("{0:.4f}".format(logs["loss"]))))

        elif self.param["OBJECT_TYPE"] == "R":
            log.debug("[{}] ({}: {}) Training batch={}/{}:{}, loss={}, mse={}".format(
                self.AI_CD,
                self.MDL_IDX,
                self.MODEL_NAME,
                self.cur_epoch,
                self.epoch,
                batch + 1,
                float("{0:.4f}".format(logs["loss"])) if "loss" in logs else 0.0,
                float("{0:.4f}".format(logs["mean_squared_error"])) if "mean_squared_error" in logs else 0.0))

    def on_epoch_end(self, epoch, logs=None):
        self.et = time.time() - self.st
        # tmp = '{0:04d}'.format(epoch + 1)
        # MDL_PATH = os.path.join(self.AI_CD, tmp)
        if epoch == 0:
            self.fstEpochEndTime = time.time() - self.fstEpochStartTime

        self.remTime = self.fstEpochEndTime * (int(self.epoch) - epoch + 1)

        if self.param["OBJECT_TYPE"] == "C":
            loss = float("{0:.4f}".format(logs["loss"])) if "loss" in logs else 0.0
            acc = float("{0:.4f}".format(logs["accuracy"])) if "accuracy" in logs else 0.0
            valAcc = float("{0:.4f}".format(logs["val_accuracy"])) if "val_accuracy" in logs else 0.0
            valLoss = float("{0:.4f}".format(logs["val_loss"])) if "val_loss" in logs else 0.0

            self.output = {
                "EPOCH": epoch + 1,
                "LOSS": loss if not math.isnan(loss) else 0.0,
                "ACCURACY": acc if not math.isnan(acc) else 0.0,
                "VAL_ACCURACY": valAcc if not math.isnan(valAcc) else 0.0,
                "VAL_LOSS": valLoss if not math.isnan(valLoss) else 0.0,
                "MDL_IDX": self.param["MDL_IDX"],
                "MODEL_NAME": self.param["MODEL_NAME"],
                "REMANING_TIME": self.remTime
            }
        elif self.param["OBJECT_TYPE"] == "R":
            loss = float("{0:.4f}".format(logs["loss"])) if "loss" in logs else 0.0
            mse = float("{0:.4f}".format(logs["mean_squared_error"])) if "mean_squared_error" in logs else 0.0

            self.output = {
                "EPOCH": epoch + 1,
                "LOSS": loss if not math.isnan(loss) else 0.0,
                "MSE": mse if not math.isnan(mse) else 0.0,
                "MDL_IDX": self.param["MDL_IDX"],
                "MODEL_NAME": self.param["MODEL_NAME"],
                "REMANING_TIME": self.remTime
            }
        self.output = self.dataLib.setTrainOutput(self.param, self.output)
        self.preOutput = self.output

    def on_train_end(self, epoch, logs=None):
        if self.param["OBJECT_TYPE"] == "C":
            loss = float("{0:.4f}".format(epoch["loss"])) if "loss" in epoch else 0.0
            acc = float("{0:.4f}".format(epoch["accuracy"])) if "accuracy" in epoch else 0.0
            valAcc = float("{0:.4f}".format(epoch["val_accuracy"])) if "val_accuracy" in epoch else 0.0
            valLoss = float("{0:.4f}".format(epoch["val_loss"])) if "val_loss" in epoch else 0.0

            self.output = {
                "EPOCH": self.cur_epoch - 1,
                "LOSS": loss,
                "ACCURACY": acc,
                "VAL_ACCURACY": valAcc,
                "VAL_LOSS": valLoss,
                "MDL_IDX": self.param["MDL_IDX"],
                "MODEL_NAME": self.param["MODEL_NAME"],
                "SAVE_MDL_PATH": self.saveMdlPath,
                "REMANING_TIME": 0

            }
        elif self.param["OBJECT_TYPE"] == "R":
            loss = float("{0:.4f}".format(epoch["loss"])) if "loss" in epoch else 0.0
            mse = float("{0:.4f}".format(epoch["mean_squared_error"])) if "mean_squared_error" in epoch else 0.0

            self.output = {
                "EPOCH": self.cur_epoch - 1,
                "LOSS": loss if not math.isnan(loss) else 0.0,
                "MSE": mse if not math.isnan(mse) else 0.0,
                "MDL_IDX": self.param["MDL_IDX"],
                "MODEL_NAME": self.param["MODEL_NAME"],
                "REMANING_TIME": 0
            }
        self.output = self.dataLib.setTrainOutput(self.param, self.output)

        # if self.output is not None:
        #     # _ = sendMsg(self.output["SRV_ADDR"], self.output["SEND_DATA"])
        #     log.debug("[{}] ({}: {}) Result epoch={}/{}, acc={}, loss={}, val_acc={}, val_loss={}".format(
        #         self.AI_CD,
        #         self.MDL_IDX,
        #         self.MODEL_NAME,
        #         self.cur_epoch - 1,
        #         self.epoch,
        #         self.preOutput["SEND_DATA"]["AI_ACC"],
        #         self.preOutput["SEND_DATA"]["AI_LOSS"],
        #         self.preOutput["SEND_DATA"]["AI_VAL_ACC"],
        #         self.preOutput["SEND_DATA"]["AI_VAL_LOSS"]))

        # log.debug("[{}] Training epoch={}/{}".format(
        #     self.AI_CD,
        #     self.cur_epoch - 1,
        #     self.epoch))
