# -*- coding:utf-8 -*-

# trainer
import os
import sys
import json
import tensorflow as tf
import traceback

# Model 폴더 찾기 위한 방법
basePath = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir, "../")))
# Model Path 등록
sys.path.append(basePath)

from Common.Logger.Logger import logger
from Common.Utils.Utils import getConfig
from Output.Output import sendMsg
import Common.Model.Model as ML
from Common.Output.TrainOutput import classificationState, yoloState, deepLabState
from Dataset.DataSet import classificationDataset, yoloDataset, deeplabDataset

from Network.TF.YOLOv3 import models as yv3

from Network.TF.YOLOv3 import dataset as yv3_dataset

# from Network.Detection.Yolov3.TF import dataset


log = logger("log")
srvIp, srvPort, logPath, tempPath, datasetPath, aiPath = getConfig()

server = "http://{}:{}/api/binary/trainBinLog".format(srvIp, srvPort)


if __name__ == "__main__":
    try:
        gpuIdx = sys.argv[1]
        # gpuIdx = -1
        # print(data, gpuIdx)
        string = ''
        while True:
            pipeData = sys.stdin.read(1024)
            if pipeData != '':
                string = string + pipeData
            else:
                break
        # print(string)
        data = json.loads(string)

        # base info
        objectType = data["OBJECT_TYPE"]
        imageList = data["IMAGE_LIST"]
        aiCd = data["AI_CD"]
        dataType = data["DATA_TYPE"]
        isAuto = data["IS_AUTO"]
        crnUsr = data["CRN_USR"]
        gpuRate = float(data["GPU_RATE"] / 100)

        print(os.getpid())
        if gpuIdx != "-1":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuIdx)
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                # 텐서플로가 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
                try:
                    tf.config.experimental.set_virtual_device_configuration(gpus[int(gpuIdx)],
                                                                            [tf.config.experimental
                                                                            .VirtualDeviceConfiguration
                                                                            (memory_limit=24220 * gpuRate)])
                except RuntimeError as e:
                    output = {"STATUS": 0, "code": str(e)}
                    _ = sendMsg("http://{}:{}/api/binary/trainBinLog".format(srvIp, srvPort), output)

        if objectType == "C":
            # get Classification Train Datas
            trainX, trainY, classes, lenClasses = classificationDataset(imageList, aiCd)

            if isAuto is True:
                epoch = 100
                batchSize = 1
                optimizer = 'adam'
                activeFunc = 'relu'
                lossFunc = 'binary_crossentropy'
                baseAiCd = None
                networkName = None
                isTransfer = None
                model = ML.trainLoadModel(objectType, isAuto, isTransfer, lenClasses, (512, 512, 3), aiCd,
                                          networkName=networkName, networkPath=baseAiCd)

            elif isAuto is False:
                # network info
                isTransfer = data["IS_TRANSFER"]
                if isTransfer is True:
                    # base aicd
                    baseAiCd = data["NETWORK_PATH"]
                    networkName = data["NETWORK_NAME"]
                else:
                    baseAiCd = None
                    networkName = None

            # hyper parameters
            epoch = data["EPOCH"]
            batchSize = data["BATCH_SIZE"]
            optimizer = data["OPTIMIZER"]
            activeFunc = data["ACTIVE_FUNC"]
            lossFunc = data["LOSS_FUNC"]

            model = ML.trainLoadModel(objectType, isAuto, isTransfer, lenClasses, (512, 512, 3), aiCd,
                                      networkName=networkName, networkPath=baseAiCd)
            model.compile(optimizer=optimizer, loss=lossFunc, metrics=['accuracy'])

            # print for each epoch callback
            batchState = classificationState(aiCd, objectType, dataType, gpuRate, crnUsr, isAuto, isTransfer)

            # model checkpoint save callback
            saveModelPath = os.path.join(aiPath, aiCd)
            modelCheckpoint = tf.keras.callbacks.ModelCheckpoint('{}/{{epoch:04d}}'.format(saveModelPath), verbose=1,
                                                                 mode='auto', save_weights_only=False)

            # early stopping callback
            earlyStopping = tf.keras.callbacks.EarlyStopping(patience=10, monitor="accuracy")
            callbacks = [batchState, modelCheckpoint, earlyStopping]

            history = model.fit(trainX, trainY, epochs=epoch, callbacks=callbacks, verbose=1)

        elif objectType == "D":
            # load dataset
            trainDataPath, valDataPath, classes, lenClasses = yoloDataset(imageList, dataType, aiCd)
            train_dataset = yv3_dataset.load_tfrecord_dataset(trainDataPath, classes, 416)
            val_dataset = yv3_dataset.load_tfrecord_dataset(valDataPath, classes, 416)

            if isAuto is True:
                epoch = 100
                batchSize = 1
                optimizer = tf.keras.optimizers.Adam(lr=1e-5, clipnorm=0.001)
                baseAiCd = None
                networkName = None
                isTransfer = None

            if isAuto is False:
                # network info
                isTransfer = data["IS_TRANSFER"]
                if isTransfer is True:
                    # base aicd
                    baseAiCd = data["NETWORK_PATH"]
                    networkName = data["NETWORK_NAME"]
                else:
                    baseAiCd = None
                    networkName = None

                # hyper parameters
                epoch = data["EPOCH"]
                batchSize = data["BATCH_SIZE"]
                optimizer = data["OPTIMIZER"].lower()
                activeFunc = data["ACTIVE_FUNC"].lower()
                lossFunc = data["LOSS_FUNC"].lower()

                if optimizer == 'adam':
                    optimizer = tf.keras.optimizers.Adam(lr=1e-5, clipnorm=0.001)
                elif optimizer == 'sgd':
                    optimizer = tf.keras.optimizers.SGD()
                elif optimizer == 'rmsprop':
                    optimizer = tf.keras.optimizers.RMSprop()
                elif optimizer == 'adagrad':
                    optimizer = tf.keras.optimizers.Adagrad()
                elif optimizer == 'adadelta':
                    optimizer = tf.keras.optimizers.Adadelta()

            # load Model
            model, anchors, anchorMasks = ML.trainLoadModel(objectType, isAuto, isTransfer, lenClasses, (416, 416, 3), aiCd,
                                                            networkName=networkName, networkPath=baseAiCd)

            # yolo loss Function
            lossFunc = [yv3.YoloLoss(anchors[mask], classes=lenClasses) for mask in anchorMasks]

            # print out for Train
            batch_stats_callback = yoloState(aiCd, objectType, dataType, gpuRate * 100, crnUsr, isTransfer)

            saveModelPath = os.path.join(aiPath, aiCd)
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('{}/{{epoch:04d}}/yolov3.tf'.
                                                                           format(saveModelPath, aiCd), verbose=1, mode='auto',
                                                                           save_weights_only=True, monitor='val_acc')
            earlyStopping = tf.keras.callbacks.EarlyStopping(patience=10, monitor="accuracy")

            callbacks = [batch_stats_callback, model_checkpoint_callback,
                         tf.keras.callbacks.ReduceLROnPlateau(verbose=1), earlyStopping]

            model.compile(optimizer=optimizer, loss=lossFunc, run_eagerly='eager_fit', metrics=['accuracy'])
            history = model.fit(train_dataset, epochs=epoch, callbacks=callbacks, validation_data=val_dataset, verbose=0)

        elif objectType == "S":
            classesData = data["CLASSES"]
            if isAuto is True:
                epoch = 100
                batchSize = 1
                optimizer = tf.keras.optimizers.Adam(lr=1e-5, clipnorm=0.001)
                baseAiCd = None
                networkName = None
                isTransfer = None
                lossFunc = 'categorical_crossentropy'

            if isAuto is False:
                # network info
                isTransfer = data["IS_TRANSFER"]
                if isTransfer is True:
                    # base aicd
                    baseAiCd = data["NETWORK_PATH"]
                    networkName = data["NETWORK_NAME"]
                else:
                    baseAiCd = None
                    networkName = None

                # hyper parameters
                epoch = data["EPOCH"]
                batchSize = data["BATCH_SIZE"]
                optimizer = data["OPTIMIZER"].lower()
                activeFunc = data["ACTIVE_FUNC"].lower()
                lossFunc = data["LOSS_FUNC"].lower()

                if optimizer == 'adam':
                    optimizer = tf.keras.optimizers.Adam(lr=1e-5, clipnorm=0.001)
                elif optimizer == 'sgd':
                    optimizer = tf.keras.optimizers.SGD()
                elif optimizer == 'rmsprop':
                    optimizer = tf.keras.optimizers.RMSprop()
                elif optimizer == 'adagrad':
                    optimizer = tf.keras.optimizers.Adagrad()
                elif optimizer == 'adadelta':
                    optimizer = tf.keras.optimizers.Adadelta()

            trainGen, ValiGen, classes, lenClasses, fileCnt = deeplabDataset(imageList, classesData, dataType, aiCd, batchSize)

            model = ML.trainLoadModel(objectType, isAuto, isTransfer, lenClasses, (416, 416, 3), aiCd,
                                      networkName=networkName, networkPath=baseAiCd)

            model.compile(optimizer=optimizer, loss=lossFunc, metrics=['accuracy'])

            saveModelPath = os.path.join(aiPath, aiCd)
            batch_stats_callback = deepLabState(aiCd, objectType, dataType, gpuRate * 100, crnUsr)
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint("{}/{{epoch:04d}}/{}.h5".
                                                                           format(saveModelPath, networkName), verbose=0,
                                                                           save_weights_only=True)
            callbacks = [batch_stats_callback, model_checkpoint_callback, tf.keras.callbacks.ReduceLROnPlateau(verbose=1)]

            history = model.fit_generator(trainGen, steps_per_epoch=fileCnt // batchSize, validation_data=ValiGen,
                                          validation_steps=batchSize, epochs=epoch, callbacks=callbacks, verbose=0)

    except tf.errors.ResourceExhaustedError as err:
        log.error(err)
        output = {"STATUS": 0, "code": str(err)}
        log.error(traceback.format_exc())
        _ = sendMsg("http://{}:{}/api/binary/trainBinLog".format(srvIp, srvPort), output)

    except Exception as e:
        log.error(e)
        log.error(traceback.format_exc())
        output = {"STATUS": 0, "code": str(e)}
        _ = sendMsg("http://{}:{}/api/binary/trainBinLog".format(srvIp, srvPort), output)
