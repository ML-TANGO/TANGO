import tensorflow as tf
import os
import sys
import traceback
# Model 폴더 찾기 위한 방법
basePath = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# Model Path 등록
sys.path.append(basePath)

# classification
import autokeras as ak
from Network.KERAS.EifficientNet import Efficientnet as efficientNet

# yolo Model
from Network.TF.YOLOv4 import models as yoloV4

from Network.TF.YOLOv3 import models as yoloV3
from Network.TF.TinyYOLOv3 import models as TinyYoloV3

from Network.TF.YOLOv3.utils import freeze_all
from Network.TF.YOLOv4.utils import freeze_all as v4_freeze

# deeplab model
from Network.KERAS.DeepLab.deeplab import Deeplabv3

# Unet Model
from Network.KERAS.Unet.Unet import unet

# efficientdet
from Network.KERAS.EfficientDet import hparams_config
from Network.KERAS.EfficientDet import efficientdet_keras
from Network.KERAS.EfficientDet.keras import train_lib, util_keras

from tensorflow.keras.layers import Input

from Common.Logger.Logger import logger
from Common.Utils.Utils import getConfig
from Common.Process.Process import prcLogData

log = logger("log")
srvIp, srvPort, logPath, tempPath, datasetPath, aiPath = getConfig()


# isTransfer 추가 필요 - smpark
def getOptimizer(optimizer):
    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr=1e-4)
    elif optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(lr=1e-4)
    elif optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(lr=1e-4)
    elif optimizer == 'adagrad':
        optimizer = tf.keras.optimizers.Adagrad(lr=1e-4)
    elif optimizer == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(lr=1e-4)
    elif optimizer == 'adamax':
        optimizer = tf.keras.optimizers.Adamax(lr=1e-4)
    elif optimizer == 'nadam':
        optimizer = tf.keras.optimizers.Nadam(lr=1e-4)
    elif optimizer == 'ftrl':
        optimizer = tf.keras.optimizers.Ftrl(lr=1e-4)
    return optimizer


def classification(isTransfer, networkName, activeFunc, optimizer, lossFunc, input_size, lenClasses, networkPath=None, aiCd=None):
    try:
        if networkName == 'AUTOKERAS':
            model = classificationAuto(os.path.join(aiPath, aiCd), None, 1)

        else:
            optimizer = getOptimizer(optimizer)
            if isTransfer:
                model = tf.keras.models.load_model(networkPath)
            else:
                if networkName == "efficientnet":
                    # efficientnet
                    model = efficientNet.EfficientNetB3(include_top=True,
                                                        weights=None,
                                                        input_shape=input_size,
                                                        pooling='max',
                                                        classes=lenClasses,
                                                        activation=activeFunc)
                elif networkName == 'vgg16':
                    from Network.KERAS.VGG import vgg
                    model = vgg.vgg16(input_tensor=input_size, activeFunc=activeFunc, lenClasses=lenClasses)

                elif networkName == 'vgg19':
                    from Network.KERAS.VGG import vgg
                    model = vgg.vgg19(input_tensor=input_size, activeFunc=activeFunc, lenClasses=lenClasses)

                elif networkName == 'resnet101':
                    from Network.KERAS.ResNet import resnet
                    model = resnet.resnet101(input_tensor=input_size, activeFunc=activeFunc, lenClasses=lenClasses)

                elif networkName == 'resnet101v2':
                    from Network.KERAS.ResNet import resnet
                    model = resnet.resnet101v2(input_tensor=input_size, activeFunc=activeFunc, lenClasses=lenClasses)

                elif networkName == 'resnet152':
                    from Network.KERAS.ResNet import resnet
                    model = resnet.resnet152(input_tensor=input_size, activeFunc=activeFunc, lenClasses=lenClasses)

                elif networkName == 'resnet152v2':
                    from Network.KERAS.ResNet import resnet
                    model = resnet.resnet152v2(input_tensor=input_size, activeFunc=activeFunc, lenClasses=lenClasses)

                elif networkName == 'resnet50':
                    from Network.KERAS.ResNet import resnet
                    model = resnet.resnet50(input_tensor=input_size, activeFunc=activeFunc, lenClasses=lenClasses)

                elif networkName == 'resnet50v2':
                    from Network.KERAS.ResNet import resnet
                    model = resnet.resnet50v2(input_tensor=input_size, activeFunc=activeFunc, lenClasses=lenClasses)

                elif networkName == 'inceptionv3':
                    from Network.KERAS.Inception import inception
                    model = inception.inceptionv3(input_tensor=input_size, activeFunc=activeFunc, lenClasses=lenClasses)

                elif networkName == 'inceptionresnetv2':
                    from Network.KERAS.Inception import inception
                    model = inception.inceptionResnetv2(input_tensor=input_size, activeFunc=activeFunc, lenClasses=lenClasses)

                elif networkName == 'mobilenet':
                    from Network.KERAS.MobileNet import mobilenet
                    model = mobilenet.inceptionResnetv2(input_tensor=input_size, activeFunc=activeFunc, lenClasses=lenClasses)

                elif networkName == 'mobilenetv2':
                    from Network.KERAS.MobileNet import mobilenet
                    model = mobilenet.inceptionResnetv2(input_tensor=input_size, activeFunc=activeFunc, lenClasses=lenClasses)

            model.compile(
                optimizer=optimizer,
                loss=lossFunc,
                metrics=['accuracy']
            )

        return model

    except Exception as e:
        log.error(traceback.format_exc())
        prcLogData(str(e))


def classificationAuto(modelPath, strategy, MAX_TRIAL):
    try:
        model = ak.ImageClassifier(
            # inputs=[ak.ImageInput()],
            # outputs=[ak.ClassificationHead()],
            overwrite=True,
            max_trials=MAX_TRIAL,
            directory=modelPath,
            distribution_strategy=strategy
        )
        return model

    except Exception as e:
        log.error(traceback.format_exc())
        prcLogData(str(e))


def detectionAuto(IMAGE_SIZE, lenClasses, batch_size, epoch):
    try:
        modelName = "efficientdet-d0"
        config = hparams_config.get_detection_config(modelName)
        config.num_epochs = 10
        config.image_size = "512x512"
        config.heads = ['object_detection']
        params = dict(
            config.as_dict(),
            model_name=modelName,
            optimizer='adam',
            iterations_per_loop=100,
            batch_size=batch_size,
            steps_per_epoch=epoch // batch_size,
            num_shards=1,
            mode='train'
        )
        model = train_lib.EfficientDetNetTrain(params['model_name'], config)
        model.compile(
            optimizer=train_lib.get_optimizer(params),
            metrics=['accuracy'],
            loss={
                'box_loss':
                    train_lib.BoxLoss(
                        params['delta'], reduction=tf.keras.losses.Reduction.NONE),
                'box_iou_loss':
                    train_lib.BoxIouLoss(
                        params['iou_loss_type'],
                        params['min_level'],
                        params['max_level'],
                        params['num_scales'],
                        params['aspect_ratios'],
                        params['anchor_scale'],
                        params['image_size'],
                        reduction=tf.keras.losses.Reduction.NONE),
                'class_loss':
                    train_lib.FocalLoss(
                        params['alpha'],
                        params['gamma'],
                        label_smoothing=params['label_smoothing'],
                        reduction=tf.keras.losses.Reduction.NONE),
                'seg_loss':
                    tf.keras.losses.SparseCategoricalCrossentropy(
                        from_logits=True,
                        reduction=tf.keras.losses.Reduction.NONE)
            }
        )

        return model
    except Exception as e:
        log.error(traceback.format_exc())
        prcLogData(str(e))


def detection(isTransfer, networkName, IMAGE_SIZE, optimizer, lenClasses, batch_size, networkPath=None):
    try:
        optimizer = getOptimizer(optimizer.lower())
        # yolo, retinanet, .... 선택 가능하도록
        if networkName == 'YOLOV3':
            # make yolo
            if isTransfer:
                log.info(isTransfer, networkName, IMAGE_SIZE, optimizer, lenClasses, batch_size, networkPath)
                classPath = networkPath.split(aiPath)[1]
                classPath = classPath.split('/')[1]
                classPath = os.path.join(aiPath, classPath, 'classes.names')
                with open(classPath, 'r') as f:
                    lenClasses = len(f.readlines())

                model = yoloV3.YoloV3(
                    IMAGE_SIZE,
                    training=True,
                    classes=lenClasses
                )
                networkPath = os.path.join(networkPath, "YOLOV3.tf")
                model.load_weights(networkPath)
                for i in model.layers:
                    if not i.name.startswith('yolo_output'):
                        i.set_weights(model.get_layer(
                            i.name).get_weights())
                        freeze_all(i)
            else:
                model = yoloV3.YoloV3(IMAGE_SIZE, training=True, classes=lenClasses)
                # model_pretrained = yoloV3.YoloV3(
                #     IMAGE_SIZE,
                #     training=True,
                #     classes=lenClasses
                # )
                # log.debug(lenClasses)
                # preTrainedYolo = os.path.abspath(os.path.join(aiPath, "../models/detection/yolo/yolov3_checkpoints/YOLOV3.tf"))
                # log.debug("===============================")
                # log.debug(preTrainedYolo)
                # model_pretrained.load_weights(preTrainedYolo)
                # model.get_layer('yolo_darknet').set_weights(
                #     model_pretrained.get_layer('yolo_darknet').get_weights())
                # freeze_all(model.get_layer('yolo_darknet'))

            anchors = yoloV3.yolo_anchors
            anchorMasks = yoloV3.yolo_anchor_masks
            lossFunc = [yoloV3.YoloLoss(anchors[mask], classes=lenClasses) for mask in anchorMasks]
            model.compile(
                optimizer=optimizer,
                loss=lossFunc,
                run_eagerly=False,
                metrics=['accuracy']
            )

        elif networkName == "TINY_YOLOV3":
            # make tiny yolo
            if isTransfer:
                classPath = networkPath.split(aiPath)[1]
                classPath = classPath.split('/')[1]
                classPath = os.path.join(aiPath, classPath, 'classes.names')
                with open(classPath, 'r') as f:
                    lenClasses = len(f.readlines())

                model = yoloV3.YoloV3(
                    IMAGE_SIZE,
                    training=True,
                    classes=lenClasses
                )
                networkPath = os.path.join(networkPath, "YOLOV3.tf")
                model.load_weights(networkPath)
                for i in model.layers:
                    if not i.name.startswith('yolo_output'):
                        i.set_weights(model.get_layer(
                            i.name).get_weights())
                        freeze_all(i)
            else:
                model = TinyYoloV3.YoloV3Tiny(
                    IMAGE_SIZE,
                    training=True,
                    classes=lenClasses
                )

            anchors = TinyYoloV3.yolo_tiny_anchors
            anchorMasks = TinyYoloV3.yolo_tiny_anchor_masks
            lossFunc = [TinyYoloV3.YoloLoss(anchors[mask], classes=lenClasses) for mask in anchorMasks]
            model.compile(
                optimizer=optimizer,
                loss=lossFunc,
                run_eagerly='eager_fit',
                metrics=['accuracy']
            )

        elif networkName == 'YOLOV4':
            if isTransfer:
                log.info(isTransfer, networkName, IMAGE_SIZE, optimizer, lenClasses, batch_size, networkPath)
                classPath = networkPath.split(aiPath)[1]
                classPath = classPath.split('/')[1]
                classPath = os.path.join(aiPath, classPath, 'classes.names')
                with open(classPath, 'r') as f:
                    lenClasses = len(f.readlines())

                model = yoloV4.YoloV4(
                    IMAGE_SIZE,
                    training=True,
                    classes=lenClasses
                )
                networkPath = os.path.join(networkPath, "YOLOV4.tf")
                model.load_weights(networkPath)
                for i in model.layers:
                    if not i.name.startswith('yolo_output'):
                        i.set_weights(model.get_layer(
                            i.name).get_weights())
                        v4_freeze(i)
            else:
                model = yoloV4.YoloV4(IMAGE_SIZE, training=True, classes=lenClasses)

            anchors = yoloV4.yolo_anchors
            anchorMasks = yoloV4.yolo_anchor_masks
            lossFunc = [yoloV4.YoloLoss(anchors[mask], classes=lenClasses) for mask in anchorMasks]

            model.compile(
                optimizer=optimizer,
                loss=lossFunc,
                run_eagerly=False,
                metrics=['accuracy']
            )

        elif networkName == "EFFICIENTDET":
            modelName = "efficientdet-d0"
            config = hparams_config.get_detection_config(modelName)
            config.num_epochs = 10
            config.image_size = "512x512"
            config.heads = ['object_detection']

            params = dict(
                config.as_dict(),
                model_name=modelName,
                optimizer='sgd',
                iterations_per_loop=100,
                batch_size=batch_size,
                steps_per_epoch=100 // batch_size,
                num_shards=1,
                mode='train'
            )
            model = train_lib.EfficientDetNetTrain(params['model_name'], config)
            if isTransfer:
                classPath = networkPath.split(aiPath)[1]
                classPath = classPath.split('/')[1]
                classPath = os.path.join(aiPath, classPath, 'classes.names')
                with open(classPath, 'r') as f:
                    lenClasses = len(f.readlines())

                networkPath = os.path.join(networkPath, "EFFICIENTDET.tf")
                model.build((None, 512, 512, 3))
                model.load_weights(networkPath)

            model.compile(
                optimizer=train_lib.get_optimizer(params),
                metrics=['accuracy'],
                loss={
                    'box_loss':
                        train_lib.BoxLoss(
                            params['delta'], reduction=tf.keras.losses.Reduction.NONE),
                    'box_iou_loss':
                        train_lib.BoxIouLoss(
                            params['iou_loss_type'],
                            params['min_level'],
                            params['max_level'],
                            params['num_scales'],
                            params['aspect_ratios'],
                            params['anchor_scale'],
                            params['image_size'],
                            reduction=tf.keras.losses.Reduction.NONE),
                    'class_loss':
                        train_lib.FocalLoss(
                            params['alpha'],
                            params['gamma'],
                            label_smoothing=params['label_smoothing'],
                            reduction=tf.keras.losses.Reduction.NONE),
                    'seg_loss':
                        tf.keras.losses.SparseCategoricalCrossentropy(
                            from_logits=True,
                            reduction=tf.keras.losses.Reduction.NONE)
                }
            )
        return model
    # elif mode == "RETINANET":
    #     # make retinanet
    # # ....
    except Exception as e:
        log.error(traceback.format_exc())
        prcLogData(str(e))


def segmentationAuto(IMAGE_SIZE, lenClasses, backbone):
    try:
        config = hparams_config.get_efficientdet_config('efficientdet-d0')
        config.heads = ['segmentation']
        config.seg_num_classes = lenClasses
        model = efficientdet_keras.EfficientDetNet(config=config)
        IMAGE_SIZE = 512
        model.build((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=1e-8),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
        return model

    except RuntimeError as e:
        log.error(traceback.format_exc())
        prcLogData(str(e))


def segmentation(isTransfer, IMAGE_SIZE, optimizer, lossFunc, activatation, networkName, lenClasses, backbone, networkPath):
    try:
        if networkName == "EFFICIENTDET-SEG":
            model = segmentationAuto(IMAGE_SIZE, lenClasses, backbone)
            return model
        else:
            optimizer = getOptimizer(optimizer.lower())
            lossFunc = lossFunc.lower()
            if isTransfer:
                classPath = networkPath.split(aiPath)[1]
                classPath = classPath.split('/')[1]
                classPath = os.path.join(aiPath, classPath, 'classes.names')
                with open(classPath, 'r') as f:
                    lenClasses = len(f.readlines())

                model = Deeplabv3(
                    classes=lenClasses,
                    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                    activation=activatation,
                    backbone=backbone,
                    weights=None
                )
                networkPath = os.path.join(networkPath, "{}.h5".format(networkName))
                model = model.load_weights(os.path.join(networkPath))

            elif networkName == 'DEEP-LAB':
                model = Deeplabv3(
                    classes=lenClasses,
                    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                    activation=activatation,
                    backbone=backbone,
                    weights=None
                )

            elif networkName == 'U-NET':
                model = unet(input_shape=(IMAGE_SIZE, IMAGE_SIZE))

            model.compile(
                optimizer=optimizer,
                loss=lossFunc,
                metrics=['accuracy']
            )
            return model
    except Exception as e:
        log.error(traceback.format_exc())
        prcLogData(str(e))
