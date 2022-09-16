# -*- coding:utf-8 -*-

# trainer
import json
import os
import sys
import signal

# tf.compat.v1.disable_eager_execution()
# print("==============================")
# print(tf.executing_eagerly())
# print("==============================")

# from tensorflow.config.experimental import set_virtual_device_configuration
# from tensorflow.config.experimental import VirtualDeviceConfiguration

import traceback
# import time

# Model 폴더 찾기 위한 방법
basePath = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# Model Path 등록
sys.path.append(basePath)

from Common.Logger.Logger import logger
from Common.Utils.Utils import getConfig, setParameter, makeDir
from Output.TrainOutput import classificationState, yoloState, deepLabState, efficientDetState
from Common.Process.Process import prcErrorData, prcSendData, prcGetArgs, prcLogData

# dataset load
from Dataset.ImageDataSet import clsDatasets, detectDataset, segDataset
from Dataset.ImageDataSet import videoDetectDataset, videoSegDataset

from Common.Model.TrainModel import classification, classificationAuto
from Common.Model.TrainModel import detection, detectionAuto
from Common.Model.TrainModel import segmentation, segmentationAuto


log = logger("log")
srvIp, srvPort, logPath, tempPath, datasetPath, aiPath = getConfig()
try:
    import tensorflow as tf

except Exception as err:
    log.error(traceback.format_exc())
    prcErrorData(__file__, str(err))


class Trainer:
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        try:
            # prcSendData(__file__, json.dumps({"TYPE": "STATE_DOWN", "PID": os.getpid(), "AI_STS": "DONE"}))
            log.info("[{}] Get signal {}".format(self.AI_CD, signum))
            sys.exit(0)
        except Exception as e:
            log.error(traceback.format_exc())
            prcLogData(str(e))

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def printAttr(self):
        log.info((self))

    def makeClassesFile(self):
        makeDir(os.path.join(aiPath, self.AI_CD))
        saveClassFile = os.path.join(aiPath, self.AI_CD, "classes.json")
        # if self.IS_AUTO:
        #     self.NETWORK_NAME = "AUTO_CLASSIFICATION" if self.OBJECT_TYPE == 'C' else self.NETWORK_NAME
        #     self.NETWORK_NAME = "AUTO_DETECTION" if self.OBJECT_TYPE == 'D' else self.NETWORK_NAME
        #     self.NETWORK_NAME = "AUTO_SEGMENTATION" if self.OBJECT_TYPE == 'S' else self.NETWORK_NAME

        saveJsonData = {
            "IMG_INFO": {
                "IMG_SIZE": self.IMG_SIZE,
                "IMG_CHANNEL": self.IMG_CHANNEL,
            },
            "NETWORK_NAME": self.NETWORK_NAME,
            "CLASS_INFO": {},
            "OBJECT_TYPE": self.OBJECT_TYPE
        }

        with open(saveClassFile, "w") as f:
            json.dump(saveJsonData, f)

    def gpuChecker(self):
        gpuStatus = self.GPU_STATUS
        gpuStatusStr = ""
        for tmp in gpuStatus:
            gpuStatusStr += "{}," .format(tmp)

        gpuStatusStr = gpuStatusStr[:-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuStatusStr
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

            except RuntimeError as e:
                log.error(traceback.format_exc())
                prcErrorData(__file__, str(e))

        self.strategy = tf.distribute.MirroredStrategy()

    def getCallbacks(self):
        # patience는 best모델 기준으로 시작되므로
        # epoch가 큰 경우 충분히 큰 값을 넣어줘야 함
        self.PATIENCE = self.EPOCH
        if self.EPOCH > 100:
            self.PATIENCE = int(self.EPOCH / 10)
        # 일반적으로 쓰는 모니터는 val_acc, val_loss라고함
        try:
            saveModelPath = os.path.join(aiPath, self.AI_CD)
            if self.OBJECT_TYPE == "C":
                savePath = '{}/{{epoch:04d}}'.format(saveModelPath)
                batchState = classificationState(
                    self.AI_CD,
                    self.OBJECT_TYPE,
                    self.DATA_TYPE,
                    self.GPU_RATE,
                    self.CRN_USR,
                    self.IS_AUTO,
                    self.NETWORK_NAME,
                    self.EPOCH,
                    savePath
                )
                modelCheckpoint = tf.keras.callbacks.ModelCheckpoint(
                    savePath,
                    verbose=0,
                    mode='auto',
                    save_weights_only=False
                )

            elif self.OBJECT_TYPE == "D":
                savePath = "{}/{{epoch:04d}}/{}.tf".format(saveModelPath, self.NETWORK_NAME)
                if self.NETWORK_NAME == "YOLOV3":
                    batchState = yoloState(
                        self.AI_CD,
                        self.OBJECT_TYPE,
                        self.DATA_TYPE,
                        self.GPU_RATE,
                        self.CRN_USR,
                        self.NETWORK_NAME,
                        self.EPOCH,
                        savePath
                    )
                elif self.NETWORK_NAME == "YOLOV4":
                    batchState = yoloState(
                        self.AI_CD,
                        self.OBJECT_TYPE,
                        self.DATA_TYPE,
                        self.GPU_RATE,
                        self.CRN_USR,
                        self.NETWORK_NAME,
                        self.EPOCH,
                        savePath
                    )
                else:
                    batchState = efficientDetState(
                        self.AI_CD,
                        self.OBJECT_TYPE,
                        self.DATA_TYPE,
                        self.GPU_RATE,
                        self.CRN_USR,
                        self.NETWORK_NAME,
                        self.EPOCH,
                        savePath
                    )

                modelCheckpoint = tf.keras.callbacks.ModelCheckpoint(
                    savePath,
                    verbose=0,
                    mode='auto',
                    save_weights_only=True
                )

            elif self.OBJECT_TYPE == "S":
                if self.NETWORK_NAME == "DEEP-LAB":
                    savePath = '{}/{{epoch:04d}}/{}.h5'.format(saveModelPath, self.NETWORK_NAME)
                else:
                    savePath = '{}/{{epoch:04d}}/{}.tf'.format(saveModelPath, self.NETWORK_NAME)
                batchState = deepLabState(
                    self.AI_CD,
                    self.OBJECT_TYPE,
                    self.DATA_TYPE,
                    self.GPU_RATE,
                    self.CRN_USR,
                    self.NETWORK_NAME,
                    self.EPOCH,
                    savePath
                )
                log.info(self.NETWORK_NAME)

                modelCheckpoint = tf.keras.callbacks.ModelCheckpoint(
                    savePath,
                    verbose=0,
                    mode='auto',
                    save_weights_only=True
                )

            if self.IS_EARLYSTOP:
                earlyStopping = tf.keras.callbacks.EarlyStopping(
                    monitor=self.EARLY_MONITOR.lower(),
                    mode=self.EARLY_MODE.lower(),
                    patience=self.PATIENCE,
                    verbose=0,
                    restore_best_weights=True)

                callbacks = [batchState, modelCheckpoint, earlyStopping]
            else:
                callbacks = [batchState, modelCheckpoint]

            return callbacks
        except RuntimeError as e:
            log.error(traceback.format_exc())
            prcErrorData(__file__, str(e))

    def loadData(self):
        self.makeClassesFile()
        try:
            if self.DATA_TYPE == 'I':
                if self.OBJECT_TYPE == "C":
                    self.DATA_PATH = os.path.join(aiPath, self.AI_CD, "{}_TEMP".format(self.AI_CD))
                    self.VAL_SPLIT = 0.25
                    # self.BATCH_SIZE = 8
                    # self.EPOCH = 100
                    train_ds, val_ds, lenClasses = clsDatasets(
                        self.DATA_PATH,
                        self.BATCH_SIZE,
                        (self.IMG_SIZE, self.IMG_SIZE, self.IMG_CHANNEL),
                        self.VAL_SPLIT,
                        self.AI_CD,
                        self.CLASS_MAP,
                        self.IMAGE_LIST
                    )
                    return train_ds, val_ds, lenClasses

                elif self.OBJECT_TYPE == "D":
                    self.IMG_SIZE = 416
                    self.VAL_SPLIT = 0.25
                    # self.BATCH_SIZE = 8
                    # self.EPOCH = 100
                    train_ds, val_ds, lenClasses = detectDataset(
                        self.BATCH_SIZE,
                        self.IMG_SIZE,
                        self.IMG_SIZE,
                        self.VAL_SPLIT,
                        self.AI_CD,
                        self.CLASS_MAP,
                        self.IMAGE_LIST
                    )
                    if lenClasses == 1:
                        lenClasses += 1
                    return train_ds, val_ds, lenClasses

                elif self.OBJECT_TYPE == "S":
                    self.VAL_SPLIT = 0.25
                    # self.BATCH_SIZE = 8
                    # self.EPOCH = 100
                    # log.info(self.IMAGE_LIST)
                    train_ds, val_ds, lenClasses = segDataset(
                        self.BATCH_SIZE,
                        self.IMG_SIZE,
                        self.IMG_SIZE,
                        self.VAL_SPLIT,
                        self.AI_CD,
                        self.CLASS_MAP,
                        self.IMAGE_LIST
                    )
                    return train_ds, val_ds, lenClasses

            elif self.DATA_TYPE == 'V':
                if self.OBJECT_TYPE == "C":
                    self.DATA_PATH = os.path.join(aiPath, self.AI_CD, "{}_TEMP".format(self.AI_CD))
                    self.VAL_SPLIT = 0.25
                    # self.BATCH_SIZE = 8
                    # self.EPOCH = 100
                    train_ds, val_ds, lenClasses = clsDatasets(
                        self.DATA_PATH,
                        self.BATCH_SIZE,
                        (self.IMG_SIZE, self.IMG_SIZE, self.IMG_CHANNEL),
                        self.VAL_SPLIT,
                        self.AI_CD,
                        self.CLASS_MAP,
                        self.IMAGE_LIST
                    )
                    return train_ds, val_ds, lenClasses

                elif self.OBJECT_TYPE == "D":
                    self.IMG_SIZE = 416
                    self.VAL_SPLIT = 0.25
                    # self.BATCH_SIZE = 8
                    # self.EPOCH = 100
                    train_ds, val_ds, lenClasses = videoDetectDataset(
                        self.BATCH_SIZE,
                        self.IMG_SIZE,
                        self.IMG_SIZE,
                        self.VAL_SPLIT,
                        self.AI_CD,
                        self.CLASS_MAP,
                        self.IMAGE_LIST
                    )
                    if lenClasses == 1:
                        lenClasses += 1
                    return train_ds, val_ds, lenClasses

                elif self.OBJECT_TYPE == "S":
                    self.VAL_SPLIT = 0.25
                    # self.BATCH_SIZE = 8
                    # self.EPOCH = 100
                    # log.info(self.IMAGE_LIST)
                    train_ds, val_ds, lenClasses = videoSegDataset(
                        self.BATCH_SIZE,
                        self.IMG_SIZE,
                        self.IMG_SIZE,
                        self.VAL_SPLIT,
                        self.AI_CD,
                        self.CLASS_MAP,
                        self.IMAGE_LIST
                    )
                    return train_ds, val_ds, lenClasses

            else:
                raise ValueError
        except RuntimeError as e:
            log.error(traceback.format_exc())
            prcErrorData(__file__, str(e))

    def Classification(self, train_ds, val_ds, lenClasses):
        try:
            if self.IS_AUTO is True:
                log.info("[{}] Define Auto Model. Network={}, Image size={}".format(
                    self.AI_CD,
                    self.NETWORK_NAME,
                    self.IMG_SIZE)
                )
                if os.path.isdir(os.path.join(aiPath, self.AI_CD, "image_classifier")):
                    import shutil
                    shutil.rmtree(os.path.join(aiPath, self.AI_CD, "image_classifier"))

                autoModel = classificationAuto(
                    os.path.join(aiPath, self.AI_CD),
                    self.strategy,
                    self.MAX_TRIAL
                )
                earlyStopping = tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    mode='auto',
                    patience=10,
                    verbose=0
                )

                autoModel.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=10,
                    batch_size=self.BATCH_SIZE * self.strategy.num_replicas_in_sync,
                    callbacks=[earlyStopping],
                    verbose=0,
                    initial_epoch=self.START_EPOCH
                )

                model = autoModel.export_model()

                log.info("[{}] Run model. epoch={}, batch={}".format(
                    self.AI_CD,
                    self.EPOCH,
                    self.BATCH_SIZE))

                model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=self.EPOCH,
                    batch_size=self.BATCH_SIZE * self.strategy.num_replicas_in_sync,
                    callbacks=self.getCallbacks(),
                    verbose=0,
                    initial_epoch=self.START_EPOCH
                )
            else:
                networkPath = None
                if self.IS_TRANSFER:
                    networkPath = self.NETWORK_PATH

                # set model
                log.info("[{}] Define Model. {},{},{},{},{}".format(
                    self.AI_CD,
                    self.IS_TRANSFER,
                    self.ACTIVE_FUNC,
                    self.OPTIMIZER,
                    self.LOSS_FUNC,
                    networkPath)
                )
                with self.strategy.scope():
                    model = classification(
                        self.IS_TRANSFER,
                        self.NETWORK_NAME,
                        self.ACTIVE_FUNC,
                        self.OPTIMIZER,
                        self.LOSS_FUNC,
                        (self.IMG_SIZE, self.IMG_SIZE, self.IMG_CHANNEL),
                        lenClasses,
                        networkPath,
                        self.AI_CD
                    )

                log.info("[{}] Run model. epoch={}, batch={}".format(
                    self.AI_CD,
                    self.EPOCH,
                    self.BATCH_SIZE))

                model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=self.EPOCH,
                    batch_size=self.BATCH_SIZE * self.strategy.num_replicas_in_sync,
                    callbacks=self.getCallbacks(),
                    verbose=0,
                    initial_epoch=self.START_EPOCH
                )
        except RuntimeError as e:
            log.error(traceback.format_exc())
            prcErrorData(__file__, str(e))
        except Exception as e:
            log.error(traceback.format_exc())
            prcErrorData(__file__, str(e))

    def Detection(self, train_ds, val_ds, lenClasses):
        try:
            if self.IS_AUTO is True:
                log.info("[{}] Define Auto Model. Network={}, Image size={}".format(
                    self.AI_CD,
                    self.NETWORK_NAME,
                    self.IMG_SIZE)
                )
                with self.strategy.scope():
                    model = detectionAuto(self.IMG_SIZE, lenClasses, self.BATCH_SIZE, self.EPOCH)

                log.info("[{}] Run model. epoch={}, batch={}".format(
                    self.AI_CD,
                    self.EPOCH,
                    self.BATCH_SIZE))

                model.fit(
                    train_ds,
                    epochs=self.EPOCH,
                    batch_size=self.BATCH_SIZE * self.strategy.num_replicas_in_sync,
                    callbacks=self.getCallbacks(),
                    validation_data=val_ds,
                    verbose=0,
                    initial_epoch=self.START_EPOCH,
                    # steps_per_epoch=len(self.IMAGE_LIST[0]["files"]) // self.BATCH_SIZE,
                )

            else:
                networkPath = None
                if self.IS_TRANSFER:
                    networkPath = self.NETWORK_PATH

                # set model
                log.info("[{}] Define Model. {},{},{},{},{}".format(
                    self.AI_CD,
                    self.IS_TRANSFER,
                    self.ACTIVE_FUNC,
                    self.OPTIMIZER,
                    self.LOSS_FUNC,
                    networkPath)
                )
                with self.strategy.scope():
                    model = detection(
                        self.IS_TRANSFER,
                        self.NETWORK_NAME,
                        self.IMG_SIZE,
                        self.OPTIMIZER,
                        lenClasses,
                        self.BATCH_SIZE,
                        networkPath
                    )
                log.info("[{}] Run model. epoch={}, batch={}".format(
                    self.AI_CD,
                    self.EPOCH,
                    self.BATCH_SIZE))

                model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=self.EPOCH,
                    batch_size=self.BATCH_SIZE * self.strategy.num_replicas_in_sync,
                    callbacks=self.getCallbacks(),
                    verbose=0,
                    initial_epoch=self.START_EPOCH,
                    # steps_per_epoch=len(self.IMAGE_LIST[0]["files"]) // self.BATCH_SIZE,
                )
        except RuntimeError as e:
            log.error(traceback.format_exc())
            prcErrorData(__file__, str(e))
        except Exception as e:
            log.error(traceback.format_exc())
            prcErrorData(__file__, str(e))

    def Segmentation(self, train_ds, val_ds, lenClasses):
        try:
            backbone = "xception"
            if self.IS_AUTO is True:
                log.info("[{}] Define Auto Model. Network={}, Image size={}".format(
                    self.AI_CD,
                    self.NETWORK_NAME,
                    self.IMG_SIZE)
                )
                with self.strategy.scope():
                    model = segmentationAuto(self.IMG_SIZE, lenClasses, backbone)

                log.info("[{}] Run model. epoch={}, batch={}".format(
                    self.AI_CD,
                    self.EPOCH,
                    self.BATCH_SIZE))

                model.fit(
                    train_ds,
                    epochs=self.EPOCH,
                    batch_size=self.BATCH_SIZE * self.strategy.num_replicas_in_sync,
                    callbacks=self.getCallbacks(),
                    validation_data=val_ds,
                    verbose=0,
                    initial_epoch=self.START_EPOCH,
                    steps_per_epoch=len(self.IMAGE_LIST[0]["files"]) // self.BATCH_SIZE
                )

            else:
                networkPath = None
                if self.IS_TRANSFER:
                    networkPath = self.NETWORK_PATH
                # set model
                log.info("[{}] Define Model. {},{},{},{},{}".format(
                    self.AI_CD,
                    self.IS_TRANSFER,
                    self.ACTIVE_FUNC,
                    self.OPTIMIZER,
                    self.LOSS_FUNC,
                    networkPath)
                )
                with self.strategy.scope():
                    model = segmentation(
                        self.IS_TRANSFER,
                        self.IMG_SIZE,
                        self.OPTIMIZER,
                        self.LOSS_FUNC,
                        self.ACTIVE_FUNC,
                        self.NETWORK_NAME,
                        lenClasses,
                        backbone,
                        networkPath
                    )
                log.info("[{}] Run model. epoch={}, batch={}".format(
                    self.AI_CD,
                    self.EPOCH,
                    self.BATCH_SIZE))

                model.fit(
                    train_ds,
                    epochs=self.EPOCH,
                    batch_size=self.BATCH_SIZE * self.strategy.num_replicas_in_sync,
                    callbacks=self.getCallbacks(),
                    validation_data=val_ds,
                    verbose=0,
                    initial_epoch=self.START_EPOCH,
                    steps_per_epoch=len(self.IMAGE_LIST[0]["files"]) // self.BATCH_SIZE
                )

        except RuntimeError as e:
            log.error(traceback.format_exc())
            prcErrorData(__file__, str(e))

        except Exception as e:
            log.error(traceback.format_exc())
            prcErrorData(__file__, str(e))


if __name__ == "__main__":
    TRAINER = Trainer()
    modelData = json.loads(prcGetArgs(0))

    # with open("./attr.txt", "w") as f:
    #     json.dump(modelData, f)

    # prcSendData(__file__, json.dumps({"TYPE": "LOG", "DATA": "Trainer Run"}))
    try:
        # define trainer with data from parent pipe
        # set parameter
        TRAINER = setParameter(TRAINER, modelData)
        TRAINER.gpuChecker()
        prcSendData(__file__, json.dumps({"TYPE": "LOG", "DATA": "Trainer Run", "AI_CD": TRAINER.AI_CD}))
        # load data
        train_ds, val_ds, lenClasses = TRAINER.loadData()
        # classification
        if TRAINER.OBJECT_TYPE == "C":
            TRAINER.Classification(train_ds, val_ds, lenClasses)

        elif TRAINER.OBJECT_TYPE == "D":
            TRAINER.Detection(train_ds, val_ds, lenClasses)

        elif TRAINER.OBJECT_TYPE == "S":
            TRAINER.Segmentation(train_ds, val_ds, lenClasses)
        else:
            raise ValueError

    except tf.errors.ResourceExhaustedError as err:
        log.error(traceback.format_exc())
        if TRAINER.IS_AUTO:
            out = modelData
            out["ERR_TYPE"] = "OOM"
            out = json.dumps(out)
            log.error(out)
            prcErrorData(__file__, out)
        else:
            prcErrorData(__file__, str(err))

    except tf.errors.OpError as err:
        log.error(traceback.format_exc())
        if TRAINER.IS_AUTO:
            out = modelData
            out["ERR_TYPE"] = "OOM"
            out = json.dumps(out)
            log.error(out)
            prcErrorData(__file__, out)
        else:
            prcErrorData(__file__, str(err))

    except tf.errors.UnknownError as err:
        log.error(traceback.format_exc())
        if TRAINER.IS_AUTO:
            out = modelData
            out["ERR_TYPE"] = "OOM"
            out = json.dumps(out)
            log.error(out)
            prcErrorData(__file__, out)
        else:
            prcErrorData(__file__, str(err))

    except tf.python.framework.errors_impl.ResourceExhaustedError as err2:
        log.error(traceback.format_exc())
        if TRAINER.IS_AUTO:
            out = modelData
            out["ERR_TYPE"] = "OOM"
            out = json.dumps(out)
            log.error(out)
            prcErrorData(__file__, out)
        else:
            prcErrorData(__file__, str(err2))

    except MemoryError as err4:
        log.error(traceback.format_exc())
        if TRAINER.IS_AUTO:
            out = modelData
            out["ERR_TYPE"] = "OOM"
            out = json.dumps(out)
            log.error(out)
            prcErrorData(__file__, out)
        else:
            prcErrorData(__file__, str(err4))

    except Exception as err3:
        log.error(traceback.format_exc())
        prcErrorData(__file__, str(err3))

    finally:
        output = {"AI_CD": TRAINER.AI_CD, "IS_LAST": True}
        prcSendData(__file__, json.dumps(output))