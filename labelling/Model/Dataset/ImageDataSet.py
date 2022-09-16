# -*- coding:utf-8 -*-
'''
OBJECT_TYPE 별 데이터셋 만드는 함수: data argumentation 추가 해야함.
1. classificationDataset(): classification dataset
2. yoloDataset(): yolo dataset
3. retinanetDataset(): yolo dataset
4. segmentationDataset(): segmentation dataset
5. makeClassesFile(): class.txt 파일 만드는 함수
'''
import cv2
import os
import errno
import sys
import json
import traceback
import tensorflow as tf
import numpy as np
from functools import partial
import shutil
import pathlib
import hashlib
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
from PIL import ImageOps
import yaml
import glob

from multiprocessing import Process
import time


from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.preprocessing import LabelEncoder
_labelEncoder = LabelEncoder()

# Model 폴더 찾기 위한 방법
pathA = os.path.join(os.path.dirname(__file__), os.path.pardir, "../")
pathB = os.path.join(os.path.join(pathA))
basePath = os.path.abspath(pathB)
# Model Path 등록
sys.path.append(basePath)

from Common.Logger.Logger import logger
from Common.Utils.Utils import getConfig, makeDir, hex2rgb

from Network.TF.YOLOv4.dataset import transform_images as v4_transform_images
from Network.TF.YOLOv4.dataset import load_tfrecord_dataset as v4_load_tfrecord_dataset
from Network.TF.YOLOv4.dataset import transform_targets as v4_transform_targets
from Network.TF.YOLOv4.models import yolo_anchors as v4_yolo_anchors
from Network.TF.YOLOv4.models import yolo_anchor_masks as v4_yolo_anchor_masks

from Network.TF.YOLOv3.dataset import transform_images, load_tfrecord_dataset, transform_targets
from Network.TF.YOLOv3.models import yolo_anchors, yolo_anchor_masks
from Common.Process.Process import prcErrorData, prcLogData, prcSendData

from Network.KERAS.EfficientDet import dataloader, hparams_config


# logger 등록
log = logger("log")
# configuration load
srvIp, srvPort, logPath, tempPath, datasetPath, aiPath = getConfig()
# server ip
server = "http://{}:{}/api/binary/trainBinLog".format(srvIp, srvPort)
# req header setting
headers = {'Content-Type': 'application/json; charset=utf-8'}


def uniqueList(inputList):
    input_dic = {}
    r_list = []
    for i, v in enumerate(inputList):
        get_value = input_dic.get(v, None)
        if get_value is None:
            input_dic[v] = i
            r_list.append(v)
    return r_list


def getClasses(CLASS_MAP):
    CLASS_NAME = []
    classData = []
    for item in CLASS_MAP:
        CLASS_NAME.append(CLASS_MAP[item]["CLASS_NAME"])
        classData.append([CLASS_MAP[item]["CLASS_NO"], CLASS_MAP[item]["CLASS_NAME"]])

    return CLASS_NAME, classData


def makeDataset(DATA_PATH, CLASS_MAP, AI_CD):
    try:
        dirFlag = makeDir(DATA_PATH)
        if dirFlag is not True:
            print("cannot make directory")

        CLASS_NAME, classData = getClasses(CLASS_MAP)
        for tmp in CLASS_NAME:
            tmpClsPath = os.path.join(DATA_PATH, tmp)
            # clearing directory
            log.debug("[{}] Clearing link directory. path={}".format(AI_CD, tmpClsPath))
            shutil.rmtree(tmpClsPath, ignore_errors=True)
            # make link directory after clearing
            dirFlag = makeDir(tmpClsPath)

        return True, CLASS_NAME, classData

    except Exception as e:
        log.error(e)
        prcErrorData(__file__, str(e))
        return False, CLASS_NAME


# symbolic link
def symLink(target, linkName):
    errorCnt = 0
    try:
        os.symlink(target, linkName)

    except OSError as e:
        print(target, linkName)
        print(e)
        if e.errno == errno.EEXIST:
            os.remove(linkName)
            os.symlink(target, linkName)
            errorCnt += 1
        else:
            raise e
    return errorCnt


jsonData = {}


def makeClassesFile(classes, aiCd, OBJECT_TYPE):
    saveClassFile = os.path.join(aiPath, aiCd, "classes.json")
    saveClassFile2 = os.path.join(aiPath, aiCd, "classes.names")

    global jsonData
    with open(saveClassFile, "r") as f:
        jsonData = json.load(f)

    classInfo = []
    with open(saveClassFile, "w") as f:
        try:
            if OBJECT_TYPE == "S":
                classInfo.append(
                    {
                        "CLASS_CD": -1,
                        "CLASS_NAME": "BG"
                    }
                )
            for i, (classNo, label) in enumerate(classes):
                classInfo.append(
                    {
                        "CLASS_CD": classNo,
                        "CLASS_NAME": label
                    }
                )
            jsonData["CLASS_INFO"] = classInfo
            json.dump(jsonData, f)

        except Exception as e:
            log.error(traceback.format_exc())
            prcLogData(str(e))

    with open(saveClassFile2, "w") as f:
        try:
            if OBJECT_TYPE == "S":
                tmp = "{}\n".format('BG')
                f.write(tmp)

            for i, label in enumerate(classes):
                if i < len(classes) - 1:
                    tmp = "{}\n".format(label[1])
                else:
                    tmp = "{}".format(label[1])
                f.write(tmp)

        except Exception as e:
            log.error(traceback.format_exc())
            prcLogData(str(e))

    return len(classes)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask


def loadSegTrain(example_proto):
    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'segmentation_mask': tf.io.FixedLenFeature([], tf.string),

    }
    example = tf.io.parse_single_example(example_proto, image_feature_description)
    input_image = tf.image.decode_jpeg(example['image'])

    input_mask = tf.image.decode_jpeg(example['segmentation_mask'])
    if jsonData["NETWORK_NAME"] == "EFFICIENTDET-SEG":
        mask_size = 128
        img_size = 512
    else:
        img_size = jsonData["IMG_INFO"]["IMG_SIZE"]
        mask_size = img_size

    input_image = tf.image.resize(input_image, (img_size, img_size))
    input_mask = tf.image.resize(input_mask, (mask_size, mask_size))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def loadSegVal(example_proto):
    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'segmentation_mask': tf.io.FixedLenFeature([], tf.string),

    }
    example = tf.io.parse_single_example(example_proto, image_feature_description)
    input_image = tf.image.decode_jpeg(example['image'])

    input_mask = tf.image.decode_jpeg(example['segmentation_mask'])
    if jsonData["NETWORK_NAME"] == "EFFICIENTDET-SEG":
        mask_size = 128
        img_size = 512
    else:
        img_size = jsonData["IMG_INFO"]["IMG_SIZE"]
        mask_size = img_size

    input_image = tf.image.resize(input_image, (img_size, img_size))
    input_mask = tf.image.resize(input_mask, (mask_size, mask_size))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


# classification dataset: only img
def clsDatasets(dataDir, batch_size, img_shape, val_split, aiCd, classMap, imageList):
    try:
        success, classes, classData = makeDataset(dataDir, classMap, aiCd)
        # make classesFile
        log.info("[{}] Gathering files.".format(aiCd))
        if success:
            classes = list(set(classes))
            classData = list(set(map(tuple, classData)))
            log.debug(classData)
            makeClassesFile(classData, aiCd, "C")
        else:
            raise Exception
        totalImageCnt = 0
        classes.sort()
        for imageListData in imageList:
            className = imageListData["CLASS_NAME"]
            lenList = len(imageListData["files"])
            imageCnt = 0
            for idx, fileName in enumerate(imageListData["files"]):
                baseName = os.path.basename(fileName)
                linkName = os.path.join(dataDir, className, '{}-{}'.format(idx, baseName))
                symLink(fileName, linkName)
                imageCnt += 1
                if imageCnt % 100 == 0:
                    log.debug("[{}] link files to {} ({}/{})".format(aiCd, className, imageCnt, lenList))
            log.debug("[{}] link files to {} ({}/{})".format(aiCd, className, imageCnt, lenList))
            totalImageCnt += imageCnt

        # data dir
        dataDir = pathlib.Path(dataDir)
        # make train set
        log.debug("[{}] Start train dataset.".format(aiCd))
        colorMode = 'rgb' if img_shape[2] == 3 else 'grayscale'
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            dataDir,
            validation_split=val_split if val_split != 0 else None,
            subset="training",
            seed=123,
            image_size=(img_shape[0], img_shape[1]),
            batch_size=batch_size,
            color_mode=colorMode,
            label_mode="categorical",
            class_names=classes
        )
        # make validation set
        log.debug("[{}] Start validation dataset.".format(aiCd))
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            dataDir,
            validation_split=val_split if val_split != 0 else None,
            subset="validation",
            seed=123,
            image_size=(img_shape[0], img_shape[1]),
            color_mode=colorMode,
            batch_size=batch_size,
            label_mode="categorical",
            class_names=classes
        )

        # class names
        class_names = train_ds.class_names

        # normalization
        log.debug("[{}] Normalize dataset.".format(aiCd))
        normalization_layer = preprocessing.Rescaling(1. / 255)
        normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

        # pre fetch
        log.debug("[{}] Prefetch dataset.".format(aiCd))
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        train_ds = normalized_ds.cache().prefetch(buffer_size=AUTOTUNE)

        normalized_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
        val_ds = normalized_ds.cache().prefetch(buffer_size=AUTOTUNE)

        log.info("[{}] Dataset ready. ({}, split={}).".format(aiCd, totalImageCnt, val_split))
        prcSendData(__file__, json.dumps({"TYPE": "STATE_CH", "PID": os.getpid(), "AI_STS": "LEARN", "AI_CD": aiCd}))
        # return
        return train_ds, val_ds, len(class_names)

    except Exception as e:
        log.error(traceback.format_exc())
        prcLogData(str(e))


def getExample(imgName, labelFile, classMap):
    # baseName, extension = os.path.splitext(trainImgs[i])
    extension = pathlib.Path(imgName).suffixes
    extension = extension[len(extension) - 1]
    with tf.io.gfile.GFile(imgName, 'rb') as fib:
        imageEncoded = fib.read()
    # log.debug(imgName)
    h, w = cv2.imread(imgName).shape[:2]
    key = hashlib.sha256(imageEncoded).hexdigest()
    checkLabels = []
    for data in classMap:
        checkLabels.append(classMap[data]["CLASS_NAME"])

    # annotation data
    with open(labelFile, "r") as jsonFile:
        labelData = json.load(jsonFile)
        classText = []
        positionData = [0 for _ in range(4)]
        items = []
        xMin = []
        yMin = []
        xMax = []
        yMax = []
        labels = []
        for polygonData in labelData["POLYGON_DATA"]:
            polyKey = str(polygonData["TAG_CD"])
            if polyKey not in classMap:
                continue

            classText.append(str.encode(classMap[polyKey]["CLASS_NAME"]))
            labels.append(polygonData["TAG_NAME"])
            positionData[0] = int(polygonData["POSITION"][0]['X'])
            positionData[1] = int(polygonData["POSITION"][0]['Y'])
            positionData[2] = int(polygonData["POSITION"][1]['X'])
            positionData[3] = int(polygonData["POSITION"][1]['Y'])

            if positionData[0] > positionData[2]:
                tmp = positionData[2]
                positionData[2] = positionData[0]
                positionData[0] = tmp

            if positionData[1] > positionData[3]:
                tmp = positionData[3]
                positionData[3] = positionData[1]
                positionData[1] = tmp

            items.append(positionData)

        labels = _labelEncoder.fit_transform(labels)

        log.debug("items : {}".format(items))

        for i, item in enumerate(items):
            xMin.append(float(item[0]) / w)
            yMin.append(float(item[1]) / h)
            xMax.append(float(item[2]) / w)
            yMax.append(float(item[3]) / h)

        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': _int64_feature(h),
            'image/width': _int64_feature(w),
            'image/filename': _bytes_feature(imgName.encode('utf8')),
            'image/source_id': _bytes_feature(imgName.encode('utf8')),
            'image/key/sha256': _bytes_feature(key.encode('utf8')),
            'image/encoded': _bytes_feature(imageEncoded),
            'image/format': _bytes_feature(extension.encode('utf8')),
            'image/object/bbox/xmin': _float_list_feature(xMin),
            'image/object/bbox/xmax': _float_list_feature(xMax),
            'image/object/bbox/ymin': _float_list_feature(yMin),
            'image/object/bbox/ymax': _float_list_feature(yMax),
            'image/object/class/text': _bytes_list_feature(classText),
            'image/object/class/label': _int64_list_feature(labels),
        }))
    return example


def getExampleSeg(imgName, maksName):
    with tf.io.gfile.GFile(imgName, 'rb') as fib:
        imageEncoded = fib.read()

    with tf.io.gfile.GFile(maksName, 'rb') as fib:
        maskEncoded = fib.read()

    labels = 0

    example = tf.train.Example(features=tf.train.Features(feature={
        'file_name': _bytes_feature(imgName.encode('utf8')),
        'image': _bytes_feature(imageEncoded),
        'label': _int64_feature(labels),
        'segmentation_mask': _bytes_feature(maskEncoded)
    }))
    return example


def writeTfRecord(aiCd, writer, labelPath, imgPath, classMap, thread_idx):
    failCnt = 0
    totalCnt = len(labelPath)
    tmpDir = os.path.join(aiPath, aiCd, "tmp")
    # log.debug("[{}] Defined classMap={}".format(aiCd, classMap))
    fileCnt = 0
    for i, labelFile in enumerate(labelPath):
        try:
            log.debug("{} : {}".format(i, imgPath[i]))
            example = getExample(imgPath[i], labelFile, classMap)
            writer.write(example.SerializeToString())
            fileCnt += 1
            if "ori.jpg" in imgPath[i]:
                return

            img = cv2.imread(imgPath[i])
            M = np.ones(img.shape, dtype="uint8")

            os.makedirs(tmpDir, exist_ok=True)
            brightRange = 20 if totalCnt >= 1000 else 5
            for val in range(50, 150, brightRange):
                # add Img
                M = M * val
                added = cv2.add(img, M)
                tmpFPath = os.path.join(tmpDir, '{}.jpg'.format(time.time()))
                cv2.imwrite(tmpFPath, added)

                # example = getExample(tmpFPath, labelFile, classMap)
                # writer.write(example.SerializeToString())
                # fileCnt += 1

                # sub Img
                subed = cv2.subtract(img, M)
                tmpFPath = os.path.join(tmpDir, '{}.jpg'.format(time.time()))
                cv2.imwrite(tmpFPath, subed)
                # example = getExample(tmpFPath, labelFile, classMap)
                # writer.write(example.SerializeToString())
                # fileCnt += 1

            files = glob.glob(os.path.join(tmpDir, "*.jpg"))
            for fCnt, f in enumerate(files):
                example = getExample(f, labelFile, classMap)
                writer.write(example.SerializeToString())
                if fCnt % 100 == 0:
                    log.debug("[{}] TFRecord write done. ({}/{})".format(aiCd, i, totalCnt))

            shutil.rmtree(tmpDir, ignore_errors=True)
            # os.remove(tmpFPath)
            # if fileCnt % 100 == 0:
            #     log.debug("[{}] TFRecord write done. ({}/{})".format(aiCd, i, totalCnt))

        except Exception as e:
            log.error(traceback.format_exc())
            log.warning("[{}] FILE : {}".format(aiCd, labelFile))
            prcLogData(str(e))
            failCnt += 1
            continue

    log.debug("[{}] End dataset({}/{}).".format(aiCd, totalCnt - failCnt, totalCnt))
    writer.close()



def getDataset(path, is_training, params, config):
    file_pattern = path
    if not file_pattern:
        raise ValueError('No matching files.')

    return dataloader.InputReader(
        file_pattern,
        is_training=is_training,
        use_fake_data=False,
        max_instances_per_image=config.max_instances_per_image)(params)

# yoloDataset make
def detectDataset(batch_size, img_height, img_width, val_split, aiCd, classMap, imageList):
    try:
        classes, classData = getClasses(classMap)
        if len(classes) > 0:
            classes = uniqueList(classes)
            classData = uniqueList(map(tuple, classData))
            makeClassesFile(classData, aiCd, "D")

        PATH = []
        LABELS = []
        for imageListData in imageList:
            for idx, fileName in enumerate(imageListData["files"]):
                PATH.append(fileName["PATH"])
                # LABELS.append(fileName["LABELS"])

        PATH = list(set(PATH))
        # LABELS = list(set(LABELS))

        for path in PATH:
            baseName = os.path.basename(path)
            basePath = os.path.dirname(path)
            extension = pathlib.Path(baseName).suffixes
            fName = baseName.replace(extension[len(extension) - 1], ".dat")
            LABELS.append("{}".format(os.path.join(basePath, fName)))

        lenFiles = len(LABELS)
        lenClasses = len(classes)

        # threadCnt = 20 if len(PATH) > 1000 else 1
        if len(PATH) <= 150:
            threadCnt = 1
        elif len(PATH) > 150 and len(PATH) < 1000:
            threadCnt = 5
        else:
            threadCnt = 10

        # threadCnt = 1
        # trainImgs = PATH[:int(lenFiles * (1.0 - val_split))]
        trainImgs = PATH

        n = len(trainImgs) // threadCnt
        trainImgs = [trainImgs[i:i + n] for i in range(0, len(trainImgs), n)]
        #trainLabels = LABELS[:int(lenFiles * (1.0 - val_split))]

        trainLabels = LABELS
        trainLabels = [trainLabels[i:i + n] for i in range(0, len(trainLabels), n)]

        st = time.time()
        log.info("[{}] Start train dataset.".format(aiCd))
        coord = tf.train.Coordinator()

        trainDatasetPath = os.path.join(aiPath, aiCd, "trainData")
        if os.path.isdir(trainDatasetPath):
            log.debug("[{}] Clearing link directory. path={}".format(aiCd, trainDatasetPath))
            shutil.rmtree(trainDatasetPath, ignore_errors=True)
        _ = makeDir(trainDatasetPath)

        trainDsPath = []
        trainWriter = []
        processes = []
        for i in range(threadCnt):
            trainDsPath.append('{}/train_{}.tfrecord'.format(trainDatasetPath, i))
            trainWriter.append(tf.io.TFRecordWriter(trainDsPath[i]))

            args = (aiCd, trainWriter[i], trainLabels[i], trainImgs[i], classMap, i)
            p = Process(target=writeTfRecord, args=args)
            p.start()
            processes.append(p)

        coord.join(processes, ignore_live_threads=True)
        # (aiCd, trainWriter, trainLabels, trainImgs, classMap)

        # valImgs = PATH[int(lenFiles * (1.0 - val_split)):]
        valImgs = PATH

        n = len(valImgs) // threadCnt
        valImgs = [valImgs[i * n:(i + 1) * n] for i in range((len(valImgs) + n - 1) // n)]

        # valLabels = LABELS[int(lenFiles * (1.0 - val_split)):]
        valLabels = LABELS
        valLabels = [valLabels[i * n:(i + 1) * n] for i in range((len(valLabels) + n - 1) // n)]

        log.info("[{}] Start validation dataset.".format(aiCd))

        # writeTfRecord(aiCd, valWriter, valLabels, valImgs, classMap)
        coord = tf.train.Coordinator()
        processes = []
        valDsPath = []
        valWriter = []
        for i in range(threadCnt):
            valDsPath.append('{}/val_{}.tfrecord'.format(trainDatasetPath, i))
            valWriter.append(tf.io.TFRecordWriter(valDsPath[i]))

            args = (aiCd, valWriter[i], valLabels[i], valImgs[i], classMap, i)
            p = Process(target=writeTfRecord, args=args)
            p.start()
            processes.append(p)

        coord.join(processes, ignore_live_threads=True)
        # coord.request_stop()
        # coord.join(processes, ignore_live_threads=True)
        log.debug("TIME : {}".format(time.time() - st))

        saveClassFile = os.path.join(aiPath, aiCd, "classes.json")
        saveClassFile2 = os.path.join(aiPath, aiCd, "classes.names")
        with open(saveClassFile, "r") as f:
            classInfo = json.load(f)

        if classInfo["NETWORK_NAME"] == 'YOLOV3':
            trainDs = load_tfrecord_dataset(trainDsPath, saveClassFile2, img_width)
            valDs = load_tfrecord_dataset(valDsPath, saveClassFile2, img_width)

            log.debug("[{}] Fetch train dataset".format(aiCd))
            anchors, anchor_masks = yolo_anchors, yolo_anchor_masks
            trainDs = trainDs.shuffle(buffer_size=512)
            trainDs = trainDs.batch(batch_size)
            trainDs = trainDs.map(lambda x, y: (
                transform_images(x, img_width),
                transform_targets(y, anchors, anchor_masks, img_width)))
            trainDs = trainDs.prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)

            log.debug("[{}] Fetch validation dataset".format(aiCd))
            valDs = valDs.shuffle(buffer_size=512)
            valDs = valDs.batch(batch_size)
            valDs = valDs.map(lambda x, y: (
                transform_images(x, img_width),
                transform_targets(y, anchors, anchor_masks, img_width)))
        elif classInfo["NETWORK_NAME"] == 'YOLOV4':
            trainDs = v4_load_tfrecord_dataset(trainDsPath, saveClassFile2, img_width)
            valDs = v4_load_tfrecord_dataset(valDsPath, saveClassFile2, img_width)

            log.debug("[{}] Fetch train dataset".format(aiCd))
            anchors, anchor_masks = v4_yolo_anchors, v4_yolo_anchor_masks
            trainDs = trainDs.shuffle(buffer_size=512)
            trainDs = trainDs.batch(batch_size)
            trainDs = trainDs.map(lambda x, y: (
                v4_transform_images(x, img_width),
                v4_transform_targets(y, anchors, anchor_masks, img_width)))
            trainDs = trainDs.prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)

            log.debug("[{}] Fetch validation dataset".format(aiCd))
            valDs = valDs.shuffle(buffer_size=512)
            valDs = valDs.batch(batch_size)
            valDs = valDs.map(lambda x, y: (
                v4_transform_images(x, img_width),
                v4_transform_targets(y, anchors, anchor_masks, img_width)))
        else:
            modelName = "efficientdet-d0"
            config = hparams_config.get_detection_config(modelName)

            config.num_epochs = 10
            imgSize = classInfo["IMG_INFO"]["IMG_SIZE"]
            config.image_size = "{}x{}".format(imgSize, imgSize)
            config.heads = ['object_detection']

            params = dict(
                config.as_dict(),
                model_name=modelName,
                batch_size=batch_size,
                num_shards=1
            )

            trainDs = getDataset(trainDsPath, True, params, config)
            valDs = getDataset(valDsPath, False, params, config)

        prcSendData(__file__, json.dumps({"TYPE": "STATE_CH", "PID": os.getpid(), "AI_STS": "LEARN", "AI_CD": aiCd}))
        return trainDs, valDs, lenClasses

    except Exception as e:
        log.error(traceback.format_exc())
        prcLogData(str(e))


# Video yoloDataset make
def videoDetectDataset(batch_size, img_height, img_width, val_split, aiCd, classMap, imageList):
    oriPath = os.path.join(aiPath, aiCd, 'ori.jpg')
    jsonPath = os.path.join(aiPath, aiCd, 'tmp.json')
    try:
        classes, classData = getClasses(classMap)
        if len(classes) > 0:
            classes = uniqueList(classes)
            classData = uniqueList(map(tuple, classData))
            makeClassesFile(classData, aiCd, "D")

        trainDatasetPath = os.path.join(aiPath, aiCd)
        lenClasses = len(classes)

        trainDsPath = '{}/train.tfrecord'.format(trainDatasetPath)
        valDsPath = '{}/val.tfrecord'.format(trainDatasetPath)
        trainWriter = tf.io.TFRecordWriter(trainDsPath)
        valWriter = tf.io.TFRecordWriter(valDsPath)

        PATH = []
        LABELS = []
        for imageListData in imageList:
            for idx, fileName in enumerate(imageListData["files"]):
                PATH.append(fileName["PATH"])
                LABELS.append(fileName["LABELS"])

        PATH = uniqueList(PATH)
        LABELS = uniqueList(LABELS)

        numFiles = len(PATH)
        totalfCnt = 0
        for i, videoPath in enumerate(PATH):
            labelPath = LABELS[i]
            vc = cv2.VideoCapture(videoPath)

            with open(labelPath, "r") as jsonFile:
                labelData = json.load(jsonFile)
            labelData = labelData["POLYGON_DATA"]
            labelLen = len(labelData)

            frameCnt = 0
            log.info("[{}] Video File DATASET MAKE START. File Name : {}, Files = {}/{}".format(aiCd, videoPath, i+1, numFiles))
            while vc.isOpened():
                ret, frame = vc.read()
                if ret:
                    # log.debug("FILENAME : {}, FRAME NUMBER : {}, LABELDATA : {}, labelLen : {}".format(videoPath, frameCnt, len(labelData[frameCnt]), labelLen))
                    if len(labelData[frameCnt]) > 0:
                        cv2.imwrite(oriPath, frame)
                        with open(jsonPath, "w", encoding='euc-kr') as jsonFile2:
                            tmp = {"POLYGON_DATA": labelData[frameCnt]}
                            json.dump(tmp, jsonFile2)
                        writeTfRecord(aiCd, trainWriter, [jsonPath], [oriPath], classMap, 0)
                        writeTfRecord(aiCd, valWriter, [jsonPath], [oriPath], classMap, 0)

                log.debug("[{}] Video frame DATASET MAKE DONE. File Name : {}, frames={}".format(aiCd, videoPath, frameCnt))
                frameCnt += 1
                if labelLen - 1 == frameCnt:
                    totalfCnt += frameCnt
                    break

            log.info("[{}] Video File DATASET MAKE DONE. File Name : {}, Files = {}/{}".format(
                aiCd,
                videoPath,
                i + 1,
                numFiles
            ))

        os.remove(oriPath)
        os.remove(jsonPath)

        # saveClassFile = os.path.join(aiPath, aiCd, "classes.names")
        # trainDs = load_tfrecord_dataset(trainDsPath, saveClassFile, img_width)
        # valDs = load_tfrecord_dataset(valDsPath, saveClassFile, img_width)

        # log.debug("[{}] Fetch train dataset".format(aiCd))
        # anchors, anchor_masks = yolo_anchors, yolo_anchor_masks
        # trainDs = trainDs.shuffle(buffer_size=512)
        # trainDs = trainDs.batch(batch_size)
        # trainDs = trainDs.map(lambda x, y: (
        #     transform_images(x, img_width),
        #     transform_targets(y, anchors, anchor_masks, img_width)))
        # trainDs = trainDs.prefetch(
        #     buffer_size=tf.data.experimental.AUTOTUNE)

        # log.debug("[{}] Fetch validation dataset".format(aiCd))
        # valDs = valDs.shuffle(buffer_size=512)
        # valDs = valDs.batch(batch_size)
        # valDs = valDs.map(lambda x, y: (
        #     transform_images(x, img_width),
        #     transform_targets(y, anchors, anchor_masks, img_width)))
        # prcSendData(__file__, json.dumps({"TYPE": "STATE_CH", "PID": os.getpid(), "AI_STS": "LEARN", "AI_CD": aiCd}))

        saveClassFile = os.path.join(aiPath, aiCd, "classes.json")
        saveClassFile2 = os.path.join(aiPath, aiCd, "classes.names")
        with open(saveClassFile, "r") as f:
            classInfo = json.load(f)

        if classInfo["NETWORK_NAME"] == 'YOLOV3':
            trainDs = load_tfrecord_dataset(trainDsPath, saveClassFile2, img_width)
            valDs = load_tfrecord_dataset(valDsPath, saveClassFile2, img_width)

            log.debug("[{}] Fetch train dataset".format(aiCd))
            anchors, anchor_masks = yolo_anchors, yolo_anchor_masks
            trainDs = trainDs.shuffle(buffer_size=512)
            trainDs = trainDs.batch(totalfCnt)
            trainDs = trainDs.map(lambda x, y: (
                transform_images(x, img_width),
                transform_targets(y, anchors, anchor_masks, img_width)))
            trainDs = trainDs.prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)

            log.debug("[{}] Fetch validation dataset".format(aiCd))
            valDs = valDs.shuffle(buffer_size=512)
            valDs = valDs.batch(totalfCnt)
            valDs = valDs.map(lambda x, y: (
                transform_images(x, img_width),
                transform_targets(y, anchors, anchor_masks, img_width)))
        elif classInfo["NETWORK_NAME"] == 'YOLOV4':
            trainDs = v4_load_tfrecord_dataset(trainDsPath, saveClassFile2, img_width)
            valDs = v4_load_tfrecord_dataset(valDsPath, saveClassFile2, img_width)

            log.debug("[{}] Fetch train dataset".format(aiCd))
            anchors, anchor_masks = v4_yolo_anchors, v4_yolo_anchor_masks
            trainDs = trainDs.shuffle(buffer_size=512)
            trainDs = trainDs.batch(totalfCnt)
            trainDs = trainDs.map(lambda x, y: (
                v4_transform_images(x, img_width),
                v4_transform_targets(y, anchors, anchor_masks, img_width)))
            trainDs = trainDs.prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)

            log.debug("[{}] Fetch validation dataset".format(aiCd))
            valDs = valDs.shuffle(buffer_size=512)
            valDs = valDs.batch(totalfCnt)
            valDs = valDs.map(lambda x, y: (
                v4_transform_images(x, img_width),
                v4_transform_targets(y, anchors, anchor_masks, img_width)))
        else:
            modelName = "efficientdet-d0"
            config = hparams_config.get_detection_config(modelName)

            config.num_epochs = 10
            imgSize = classInfo["IMG_INFO"]["IMG_SIZE"]
            config.image_size = "{}x{}".format(imgSize, imgSize)
            config.heads = ['object_detection']

            params = dict(
                config.as_dict(),
                model_name=modelName,
                batch_size=batch_size,
                num_shards=1
            )

            trainDs = getDataset(trainDsPath, True, params, config)
            valDs = getDataset(valDsPath, False, params, config)

        prcSendData(__file__, json.dumps({"TYPE": "STATE_CH", "PID": os.getpid(), "AI_STS": "LEARN", "AI_CD": aiCd}))

        return trainDs, valDs, lenClasses

    except Exception as e:
        log.error(traceback.format_exc())
        prcLogData(str(e))
        os.remove(oriPath)
        os.remove(jsonPath)


def retinanetDataset(imageList, dataType):
    return 0


def dataTomask(imgPaths, classMap):
    try:
        maskPaths = []
        colors = []
        for classMapData in classMap:
            color = classMap[classMapData]["COLOR"]
            colors.append(color)

        for imgPath in imgPaths:
            img = Image.open(imgPath)
            extension = pathlib.Path(imgPath).suffixes
            dataFile = imgPath.replace(extension[len(extension) - 1], ".dat")

            dirName = os.path.dirname(dataFile)
            baseName = os.path.basename(dataFile)

            extension = pathlib.Path(baseName).suffixes
            extension = extension[len(extension) - 1]
            baseName = "MASK_" + baseName.split(extension)[0] + '.png'

            maskPath = os.path.join(dirName, baseName)
            maskPaths.append(maskPath)
            # maskFile = imgPath.split(extension)[0] + '.png'
            with open(dataFile, "r") as jsonFile:
                data = json.load(jsonFile)
                if "POLYGON_DATA" not in data:
                    polygonData = None
                else:
                    polygonData = data["POLYGON_DATA"]
                    polygonData = data["POLYGON_DATA"] if len(polygonData) != 0 else None

                if "BRUSH_DATA" not in data:
                    brushData = None
                else:
                    brushData = data["BRUSH_DATA"]
                    brushData = data["BRUSH_DATA"] if len(brushData) != 0 else None

            mask = Image.new("RGB", img.size)
            if (polygonData is None) and (brushData is not None):
                prcLogData("This image is not annotation")
                continue
            else:
                if polygonData is not None:
                    for i, pdata in enumerate(polygonData):
                        positionData = []
                        draw = ImageDraw.Draw(mask)
                        # color = pdata["COLOR"]
                        if str(pdata["TAG_CD"]) in classMap:
                            color = classMap[str(pdata["TAG_CD"])]["COLOR"]
                        else:
                            continue
                        for i, position in enumerate(pdata["POSITION"]):
                            positionData.append(
                                (position["X"], position["Y"])
                            )
                        if len(positionData) <= 1:
                            continue
                        draw.polygon((positionData), fill=color)

                if brushData is not None:
                    for bdata in brushData:
                        positionData = []
                        draw = ImageDraw.Draw(mask)
                        if str(pdata["TAG_CD"]) in classMap:
                            color = classMap[str(pdata["TAG_CD"])]["COLOR"]
                        else:
                            continue
                        lineWidth = bdata["LINE_WIDTH"]
                        position = bdata["POINTS"]
                        mode = bdata["MODE"]
                        for i in range(0, len(position), 2):
                            x = position[i]
                            y = position[i + 1]
                            positionData.append(
                                (x, y)
                            )
                        # positionData = sorted(positionData, key=lambda x: x[1])
                        if mode == 'source-over':
                            draw.line((positionData), fill=color, joint='curve', width=lineWidth)
                        elif mode == 'destination-out':
                            draw.line((positionData), fill='#000000', joint='curve', width=lineWidth)

                # mask = ImageOps.grayscale(mask)

                # mask = mask.resize((128, 128))
                mask.save(maskPath)

        colors = uniqueList(colors)

        log.info("Change Mask Image to Label Mask for Train")
        for idx, maskPath in enumerate(maskPaths):
            # color mask -> label mask
            labelMask = Image.open(maskPath)
            w, h = labelMask.size

            maskPix = labelMask.load()

            for i, color in enumerate(colors):
                color = hex2rgb(color)
                for x in range(0, w):
                    for y in range(0, h):
                        if maskPix[x, y] == color:
                            maskPix[x, y] = (i + 1, i + 1, i + 1)
                        else:
                            maskPix[x, y] = 0

            if idx % 100 == 0:
                log.info("Change Files. ({} / {})".format(idx + 1, len(maskPaths)))

            labelMask = ImageOps.grayscale(labelMask)
            labelMask.save(maskPath)

        return maskPaths

    except Exception as e:
        # log.error(traceback.format_exc())
        log.warning("[{}] FILE : {}".format(dataFile))
        prcLogData(str(e))


def segDataset(batch_size, img_height, img_width, val_split, aiCd, classMap, imageList):
    try:
        log.info("[{}] Start datasets.".format(aiCd))
        classes, classData = getClasses(classMap)
        if len(classes) > 0:
            classes = uniqueList(classes)
            classData = uniqueList(map(tuple, classData))
            makeClassesFile(classData, aiCd, "S")

        trainDatasetPath = os.path.join(aiPath, aiCd)
        trainDsPath = '{}/train.tfrecord'.format(trainDatasetPath)
        valDsPath = '{}/val.tfrecord'.format(trainDatasetPath)

        trainWriter = tf.io.TFRecordWriter(trainDsPath)
        valWriter = tf.io.TFRecordWriter(valDsPath)

        dataPath = os.path.join(aiPath, aiCd, "{}_TEMP".format(aiCd))
        makeDir(dataPath)
        log.debug("[{}] Clearing link directory. path={}".format(aiCd, dataPath))
        shutil.rmtree(dataPath, ignore_errors=True)

        trainDataPathImg = os.path.join(aiPath, aiCd, "{}_TEMP".format(aiCd), "train/images")
        makeDir(trainDataPathImg)

        trainDataPathMask = os.path.join(aiPath, aiCd, "{}_TEMP".format(aiCd), "train/masks")
        makeDir(trainDataPathMask)

        valDataPathImg = os.path.join(aiPath, aiCd, "{}_TEMP".format(aiCd), "validation/images")
        makeDir(valDataPathImg)

        valDataPathMask = os.path.join(aiPath, aiCd, "{}_TEMP".format(aiCd), "validation/masks")
        makeDir(valDataPathMask)

        imgPaths = []

        for idx, fileName in enumerate(imageList[0]["files"]):
            imgPaths.append(fileName["PATH"])

        maskPaths = dataTomask(imgPaths, classMap)

        lenFiles = len(maskPaths)
        lenClasses = len(classes)

        log.info("[{}] Gathering image files.".format(aiCd))
        trainImgs = imgPaths[:int(lenFiles * (1.0 - val_split))]
        valImgs = imgPaths[int(lenFiles * (1.0 - val_split)):]

        log.info("[{}] Gathering label files.".format(aiCd))
        trainLabels = maskPaths[:int(lenFiles * (1.0 - val_split))]
        valLabels = maskPaths[int(lenFiles * (1.0 - val_split)):]

        # trainGen = makeGen(trainImgs, trainLabels, trainDataPathImg, trainDataPathMask, img_width, batch_size, "training", "I")
        # ValiGen = makeGen(valImgs, valLabels, valDataPathImg, valDataPathMask, img_width, batch_size, "training", "I")

        log.info("[{}] Start train dataset.".format(aiCd))
        makeGen(trainImgs, trainLabels, trainDataPathImg, trainDataPathMask, batch_size, "I", trainWriter)
        log.info("[{}] End train dataset.".format(aiCd))

        log.info("[{}] Start validation dataset.".format(aiCd))
        makeGen(valImgs, valLabels, valDataPathImg, valDataPathMask, batch_size, "I", valWriter)
        log.info("[{}] End validation dataset.".format(aiCd))

        log.debug("[{}] Fetch train dataset".format(aiCd))
        train = tf.data.TFRecordDataset(trainDsPath)
        train = train.map(loadSegTrain)
        trainDs = train.cache().shuffle(buffer_size=1).batch(batch_size).repeat()
        trainDs = trainDs.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)

        log.debug("[{}] Fetch validation dataset".format(aiCd))
        # valDs = get_dataset(valDsPath, batch_size, img_width)
        valid = tf.data.TFRecordDataset(valDsPath)
        valid = valid.map(loadSegVal)
        valDs = valid.batch(len(valImgs))

        prcSendData(__file__, json.dumps({"TYPE": "STATE_CH", "PID": os.getpid(), "AI_STS": "LEARN", "AI_CD": aiCd}))

        return trainDs, valDs, lenClasses + 1
        # return 0

    except Exception as e:
        log.error(traceback.format_exc())
        prcLogData(str(e))


def videoDataTomask(imgFile, polygonData, maskPath, classMap):
    try:
        img = Image.open(imgFile)

        mask = Image.new("RGB", img.size)
        colors = []
        for classMapData in classMap:
            color = classMap[classMapData]["COLOR"]
            colors.append(color)

        if polygonData is not None:
            for pdata in polygonData:
                positionData = []
                draw = ImageDraw.Draw(mask)
                if str(pdata["TAG_CD"]) in classMap:
                    color = classMap[str(pdata["TAG_CD"])]["COLOR"]
                else:
                    continue
                for i, position in enumerate(pdata["POSITION"]):
                    positionData.append(
                        (position["X"], position["Y"])
                    )
                if len(positionData) <= 1:
                    continue
                draw.polygon((positionData), fill=color)
            # mask = ImageOps.grayscale(mask)
            # mask = mask.resize((128, 128))
            mask.save(maskPath)

        labelMask = Image.open(maskPath)
        w, h = labelMask.size
        maskPix = labelMask.load()

        for i, color in enumerate(colors):
            color = hex2rgb(color)
            for x in range(0, w):
                for y in range(0, h):
                    if maskPix[x, y] == color:
                        maskPix[x, y] = (i + 1, i + 1, i + 1)
                    else:
                        maskPix[x, y] = 0

        labelMask = ImageOps.grayscale(labelMask)
        labelMask.save(maskPath)

    except Exception as e:
        log.error(traceback.format_exc())
        prcLogData(str(e))


def videoSegDataset(batch_size, img_height, img_width, val_split, aiCd, classMap, imageList):
    try:
        log.info("[{}] Start datasets.".format(aiCd))
        classes, classData = getClasses(classMap)
        if len(classes) > 0:
            classes = list(set(classes))
            classData = list(set(map(tuple, classData)))
            makeClassesFile(classData, aiCd, "S")
        lenClasses = len(classes)

        trainDatasetPath = os.path.join(aiPath, aiCd)
        trainDsPath = '{}/train.tfrecord'.format(trainDatasetPath)
        valDsPath = '{}/val.tfrecord'.format(trainDatasetPath)
        trainWriter = tf.io.TFRecordWriter(trainDsPath)
        valWriter = tf.io.TFRecordWriter(valDsPath)

        dataPath = os.path.join(aiPath, aiCd, "{}_TEMP".format(aiCd))
        makeDir(dataPath)
        log.debug("[{}] Clearing link directory. path={}".format(aiCd, dataPath))
        shutil.rmtree(dataPath, ignore_errors=True)

        trainDataPathImg = os.path.join(aiPath, aiCd, "{}_TEMP".format(aiCd), "train/images")
        makeDir(trainDataPathImg)

        trainDataPathMask = os.path.join(aiPath, aiCd, "{}_TEMP".format(aiCd), "train/masks")
        makeDir(trainDataPathMask)

        valDataPathImg = os.path.join(aiPath, aiCd, "{}_TEMP".format(aiCd), "validation/images")
        makeDir(valDataPathImg)

        valDataPathMask = os.path.join(aiPath, aiCd, "{}_TEMP".format(aiCd), "validation/masks")
        makeDir(valDataPathMask)

        PATH = []
        LABELS = []

        for imageListData in imageList:
            for idx, fileName in enumerate(imageListData["files"]):
                PATH.append(fileName["PATH"])
                LABELS.append(fileName["LABELS"])

        PATH = uniqueList(PATH)
        LABELS = uniqueList(LABELS)

        numFiles = len(PATH)
        saveIdx = 0
        totalfCnt = 0
        for i, videoPath in enumerate(PATH):
            labelPath = LABELS[i]
            vc = cv2.VideoCapture(videoPath)
            with open(labelPath, "r") as jsonFile:
                labelData = json.load(jsonFile)
            labelData = labelData["POLYGON_DATA"]
            labelLen = len(labelData)
            frameCnt = 0
            log.info("[{}] Video File DATASET MAKE START. File Name : {}, Files = {}/{}".format(
                aiCd,
                videoPath,
                i + 1,
                numFiles
            ))

            while vc.isOpened():
                ret, frame = vc.read()
                if ret:
                    # log.debug("FILENAME : {}, FRAME NUMBER : {}, LABELDATA : {}, labelLen : {}".format(videoPath, frameCnt, len(labelData[frameCnt]), labelLen))
                    if len(labelData[frameCnt]) > 0:
                        imgName = os.path.join(trainDataPathImg, '{}_img.jpg'.format(saveIdx))
                        maskName = os.path.join(trainDataPathMask, '{}_mask.png'.format(saveIdx))
                        cv2.imwrite(imgName, frame)
                        polygonData = labelData[frameCnt]
                        videoDataTomask(imgName, polygonData, maskName, classMap)

                        imgName = os.path.join(valDataPathImg, '{}_img.jpg'.format(saveIdx))
                        maskName = os.path.join(valDataPathMask, '{}_mask.png'.format(saveIdx))
                        cv2.imwrite(imgName, frame)
                        videoDataTomask(imgName, polygonData, maskName, classMap)
                log.debug("[{}] Video frame DATASET MAKE DONE. File Name : {}, frames={}".format(aiCd, videoPath, frameCnt))
                frameCnt += 1
                saveIdx += 1
                if labelLen - 1 == frameCnt:
                    totalfCnt += frameCnt
                    break

            log.info("[{}] Video File DATASET MAKE DONE. File Name : {}, Files = {}/{}".format(
                aiCd,
                videoPath,
                i + 1,
                numFiles
            ))

        log.info("[{}] Start train dataset.".format(aiCd))
        makeGen(None, None, trainDataPathImg, trainDataPathMask, batch_size, "V", trainWriter)
        log.info("[{}] End train dataset.".format(aiCd))

        log.info("[{}] Start validation dataset.".format(aiCd))
        makeGen(None, None, valDataPathImg, valDataPathMask, batch_size, "V", valWriter)
        log.info("[{}] End validation dataset.".format(aiCd))

        log.debug("[{}] Fetch train dataset".format(aiCd))
        train = tf.data.TFRecordDataset(trainDsPath)
        train = train.map(loadSegTrain)
        trainDs = train.cache().shuffle(buffer_size=512).batch(batch_size).repeat()
        trainDs = trainDs.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)

        log.debug("[{}] Fetch validation dataset".format(aiCd))
        # valDs = get_dataset(valDsPath, batch_size, img_width)
        valid = tf.data.TFRecordDataset(valDsPath)
        valid = valid.map(loadSegVal)
        valDs = valid.batch(totalfCnt)

        return trainDs, valDs, lenClasses + 1

    except Exception as e:
        log.error(traceback.format_exc())
        prcLogData(str(e))


def makeGen(images, masks, datasetImgPath, datasetMaskPath, batchSize, mode, writer):
    try:
        if mode == "I":
            for idx, name in enumerate(images):
                baseName = os.path.basename(name)
                linkName = os.path.join(datasetImgPath, '{}-{}'.format(idx, baseName))
                symLink(name, linkName)

            for idx, name in enumerate(masks):
                baseName = os.path.basename(name)
                linkName = os.path.join(datasetMaskPath, '{}-{}'.format(idx, baseName))
                symLink(name, linkName)

        elif mode == "V":
            datasetImgPath = datasetImgPath
            datasetMaskPath = datasetMaskPath

        datasetImg = os.listdir(datasetImgPath)
        datasetMask = os.listdir(datasetMaskPath)

        for i, imgName in enumerate(datasetImg):
            imgPath = os.path.join(datasetImgPath, imgName)
            maskPath = os.path.join(datasetMaskPath, datasetMask[i])

            example = getExampleSeg(imgPath, maskPath)
            writer.write(example.SerializeToString())

    except Exception as e:
        log.error(traceback.format_exc())
        prcLogData(e)


def predictDataset(dataType, modelName, objectType, imgPath, inputShape):
    try:
        if dataType == "I":
            if objectType == "C":
                if inputShape[2] == 3:
                    img = cv2.imread(imgPath)
                elif inputShape[2] == 1:
                    img = cv2.imread(imgPath, 0)
                oriH, oriW = img.shape[:2]

                img = cv2.resize(
                    img,
                    (inputShape[0], inputShape[1]),
                    interpolation=cv2.INTER_CUBIC
                )
                img = img.reshape(1, inputShape[0], inputShape[1], inputShape[2])
                img = img / 255.
                resizeH, resizeW = img.shape[:2]
                return img, oriH, oriW, resizeH, resizeW

            elif objectType == "D":
                if modelName == "YOLOV3":
                    img_raw = tf.image.decode_image(
                        open(imgPath, "rb").read(),
                        channels=3
                    )
                    oriH, oriW = img_raw.shape[:2]
                    img = tf.expand_dims(img_raw, 0)
                    img = transform_images(img, 416)
                    resizeH, resizeW = 416, 416

                elif modelName == "YOLOV4":
                    img_raw = tf.image.decode_image(
                        open(imgPath, "rb").read(),
                        channels=3
                    )
                    oriH, oriW = img_raw.shape[:2]
                    img = tf.expand_dims(img_raw, 0)
                    img = v4_transform_images(img, 416)
                    resizeH, resizeW = 416, 416

                elif modelName == "EFFICIENTDET":
                    log.info(modelName)
                    img = cv2.imread(imgPath)
                    oriH, oriW = img.shape[:2]
                    resizeH, resizeW = 512, 512

                    img = [np.array(Image.open(imgPath).convert("RGB"))]
                    img = tf.convert_to_tensor(img, dtype=tf.uint8)

                return img, oriH, oriW, resizeH, resizeW

            elif objectType == "S":
                if modelName == 'U-NET':
                    img_raw = tf.image.decode_image(open(imgPath, "rb").read(), channels=3)
                    oriH, oriW = img_raw.shape[:2]
                    img = tf.expand_dims(img_raw, 0)
                    img = transform_images(img, inputShape[0])
                    resizeH, resizeW = inputShape[0], inputShape[1]
                else:
                    # img = np.array(Image.open(imgPath))
                    img = cv2.imread(imgPath)
                    # img = cv2.imread(imgPath)
                    oriH, oriW, _ = img.shape
                    resizeH, resizeW = inputShape[0], inputShape[1]

                return img, oriH, oriW, resizeH, resizeW

        elif dataType == "V":
            if objectType == "C":
                img = imgPath
                oriH, oriW = img.shape[:2]
                img = cv2.resize(
                    img,
                    (inputShape[0], inputShape[1]),
                    interpolation=cv2.INTER_CUBIC
                )
                img = img.reshape(1, inputShape[0], inputShape[1], inputShape[2])
                img = img / 255.
                resizeH, resizeW = img.shape[:2]

                return img, oriH, oriW, resizeH, resizeW

            elif objectType == "D":
                # img_raw = tf.image.decode_image(imgPath, channels=3)
                if modelName == "YOLOV3":
                    img_raw = imgPath
                    oriH, oriW = img_raw.shape[:2]
                    img = tf.expand_dims(img_raw, 0)
                    img = transform_images(img, 416)
                    resizeH, resizeW = 416, 416

                elif modelName == "YOLOV4":
                    img_raw = imgPath
                    oriH, oriW = img_raw.shape[:2]
                    img = tf.expand_dims(img_raw, 0)
                    img = v4_transform_images(img, 416)
                    resizeH, resizeW = 416, 416

                elif modelName == "EFFICIENTDET":
                    # cv2.imwrite("./tmp.jpg", imgPath)
                    # img = cv2.imread("./tmp.jpg")
                    img = imgPath
                    oriH, oriW = img.shape[:2]
                    resizeH, resizeW = 512, 512
                    img = [img]
                    img = tf.convert_to_tensor(img, dtype=tf.uint8)

                return img, oriH, oriW, resizeH, resizeW

            elif objectType == "S":
                if modelName == 'U-NET':
                    img_raw = tf.image.decode_image(imgPath)
                    oriH, oriW = img_raw.shape[:2]
                    img = tf.expand_dims(img_raw, 0)
                    img = transform_images(img, inputShape[0])
                    resizeH, resizeW = inputShape[0], inputShape[1]
                else:
                    img = imgPath
                    # img = cv2.imread(imgPath)
                    oriH, oriW, _ = img.shape
                    resizeH, resizeW = 512, 512

                return img, oriH, oriW, resizeH, resizeW

    except Exception as e:
        log.error(traceback.format_exc())
        prcLogData(e)
