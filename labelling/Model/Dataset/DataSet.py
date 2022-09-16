# -*- coding:utf-8 -*-
'''
OBJECT_TYPE 별 데이터셋 만드는 함수 : data argumentation 추가 해야함.
1. classificationDataset() : classification dataset
2. yoloDataset() : yolo dataset
3. retinanetDataset() : yolo dataset
4. segmentationDataset() : segmentation dataset
5. makeClassesFile() : class.txt 파일 만드는 함수
'''

import cv2
import os
import sys
import traceback
import hashlib
import tensorflow as tf
import pandas as pd
import numpy as np
import random


# Model 폴더 찾기 위한 방법
basePath = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir, "../")))
# Model Path 등록
sys.path.append(basePath)

from Common.Logger.Logger import logger
from Common.Utils.Utils import getConfig, makeDir
from Output.Output import sendMsg
import Dataset.ImageProcessing as ImageProcessing
from Network.TF.YOLOv3.dataset import transform_images

log = logger("log")
srvIp, srvPort, logPath, tempPath, datasetPath, aiPath = getConfig()


# classification dataset : only img
def classificationDataset(imageList, aiCd):
    try:
        trainXTmp = []
        trainY = []
        classNo = []
        className = []
        classes = []

        trainFiles = []
        log.info("DATASET MAKE START")
        for data in imageList:
            className.append(data["CLASS_NAME"])
            classNo.append(data["CLASS_NO"])

            for imageFileName in data["files"]:
                trainFiles.append(imageFileName)

        random.shuffle(trainFiles)
        random.shuffle(trainFiles)

        # log.info(trainFiles)
        for imgName in trainFiles:
            log.debug(imgName)

            img = cv2.imread(imgName, 1)
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
            trainXTmp.append(img)
            for j in range(len(className)):
                if className[j] in imgName:
                    trainY.append(j)

            # data argumentation
            # 1. rotation
            for i in range(0, 360, 10):
                dst = ImageProcessing.rotate(img, i)
                trainXTmp.append(dst)
                for j in range(len(className)):
                    if className[j] in imgName:
                        trainY.append(j)

            # 2. brightness
            for i in range(100, 200, 10):
                dst = ImageProcessing.adjustBrightness(img, i)
                trainXTmp.append(dst)
                for j in range(len(className)):
                    if className[j] in imgName:
                        trainY.append(j)

            # 3. H flip
            dst = ImageProcessing.flip(img, "H")
            trainXTmp.append(dst)
            for i in range(len(className)):
                if className[i] in imgName:
                    trainY.append(i)

            # 4. V flip
            dst = ImageProcessing.flip(img, "V")
            trainXTmp.append(dst)
            for i in range(len(className)):
                if className[i] in imgName:
                    trainY.append(i)

        trainX = np.array(trainXTmp)
        trainY = np.array(trainY)

        # classes = [classNo, className]
        # print(classes)
        for i in range(len(classNo)):
            tmp = "{},{}".format(classNo[i], className[i])
            classes.append(tmp)
        print(classes)
        lenClasses = makeClassesFile(classes, aiCd)

        log.info("DATASET MAKE FINISH")
        return trainX, trainY, className, lenClasses

    except Exception as e:
        log.error(e)
        log.error(traceback.format_exc())
        output = {"STATUS": 0, "code": str(e)}
        _ = sendMsg("http://{}:{}/api/binary/trainBinLog".format(srvIp, srvPort), output)


# yoloDataset make
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


def getExample(line, image_path):
    labelCnt = -1
    class_text = []
    labels = []
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    truncated = []
    poses = []
    difficult_obj = []

    # read the image
    with tf.io.gfile.GFile(image_path, 'rb') as fib:
        image_encoded = fib.read()
    h, w, _ = cv2.imread(image_path).shape

    key = hashlib.sha256(image_encoded).hexdigest()
    baseName, extension = os.path.splitext(image_path)

    extension = extension.lstrip('.')
    # read the parameters
    item = line.split(',')

    if item[len(item) - 1] not in class_text:
        labelCnt += 1

    class_text.append(str.encode(item[len(item) - 1]))
    labels.append(labelCnt)

    # xmin, ymin, xmax, ymax
    # item[0], item[1], item[2], item[3]

    if float(item[0]) > float(item[2]):
        tmp = item[2]
        item[2] = item[0]
        item[0] = tmp

    if float(item[1]) > float(item[3]):
        tmp = item[3]
        item[3] = item[1]
        item[1] = tmp

    item[0] = int(float(item[0]))
    item[1] = int(float(item[1]))
    item[2] = int(float(item[2]))
    item[3] = int(float(item[3]))

    xmin.append(float(item[0]) / w)
    ymin.append(float(item[1]) / h)
    xmax.append(float(item[2]) / w)
    ymax.append(float(item[3]) / h)

    truncated.append(0)
    difficult_obj.append(0)
    poses.append("Unspecified".encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(h),
        'image/width': _int64_feature(w),
        'image/filename': _bytes_feature(image_path.encode('utf8')),
        'image/source_id': _bytes_feature(image_path.encode('utf8')),
        'image/key/sha256': _bytes_feature(key.encode('utf8')),
        'image/encoded': _bytes_feature(image_encoded),
        'image/format': _bytes_feature(extension.encode('utf8')),
        'image/object/bbox/xmin': _float_list_feature(xmin),
        'image/object/bbox/xmax': _float_list_feature(xmax),
        'image/object/bbox/ymin': _float_list_feature(ymin),
        'image/object/bbox/ymax': _float_list_feature(ymax),
        'image/object/class/text': _bytes_list_feature(class_text),
        'image/object/class/label': _int64_list_feature(labels),
    }))

    return example


def yoloDataset(imageList, dataType, aiCd):
    try:
        classes = []
        labelData = []
        imgPath = []
        dataType = dataType.upper()

        # Create a TFRecordWriter, we will save the outputs in the following file
        trainPath = os.path.join(aiPath, aiCd)
        makeDir(os.path.join(aiPath, aiCd))

        print(trainPath)
        csvPath = os.path.join(trainPath, "{}.csv".format(aiCd))
        log.info("START MAKE DATASET")
        # dataset argumentation -> 90 deg rotation, brightness
        if dataType == "I":
            for data in imageList:
                classes.append(data["className"])
                for files in data["files"]:
                    labelData.append(files["LABELS"].split(";"))
                    imgPath.append(files["PATH"])

            labelData = [v for v in labelData if v]

            with open(csvPath, "w") as f:
                for i in range(0, 3000):
                    for idx, label in enumerate(labelData):
                        buf = label[0].split(",")
                        buf2 = "{},{},{},{},{},{}\n".format(imgPath[idx], int(float(buf[0])), int(float(buf[1])),
                                                            int(float(buf[2])), int(float(buf[3])), buf[4])
                        f.write(buf2)

            df = pd.read_csv(csvPath, header=None)
            df = df.sample(frac=1)
            df.to_csv(csvPath, header=None, index=None)

            with open(csvPath, "r") as f:
                lines = f.read().splitlines()
                lenLines = len(lines)
                train = int(lenLines * 0.7)
                trainLines = lines[:train - 1]
                testLines = lines[train:]

            log.info("START MAKE TRAIN DATASET")
            train_writer = tf.io.TFRecordWriter('{}/train.tfrecord'.format(trainPath))
            val_writer = tf.io.TFRecordWriter('{}/val.tfrecord'.format(trainPath))
            labelList = []
            for line in trainLines:
                tmp = line.split(",")
                imgPath = tmp[0]
                try:
                    img = cv2.imread(imgPath, 1)
                    img = cv2.resize(img, (416, 416), interpolation=cv2.INTER_CUBIC)
                    labelList.append(tmp[len(tmp) - 1])

                    tmp1 = line.split(imgPath)
                    newLine = tmp1[1].lstrip(',')
                    example = getExample(newLine, imgPath)
                    if example != 0:
                        train_writer.write(example.SerializeToString())
                except Exception:
                    continue

            log.info("START MAKE VALIDATION DATASET")
            for line in testLines:
                tmp = line.split(",")
                imgPath = tmp[0]

                try:
                    img = cv2.imread(imgPath, 1)
                    img = cv2.resize(img, (416, 416), interpolation=cv2.INTER_CUBIC)
                    labelList.append(tmp[len(tmp) - 1])
                    tmp1 = line.split(imgPath)
                    newLine = tmp1[1].lstrip(',')
                    example = getExample(newLine, imgPath)
                    if example != 0:
                        val_writer.write(example.SerializeToString())
                except Exception:
                    continue

        # elif dataType == "V" :

        else:
            log.error("DataType only Image or Video!")
            output = {"STATUS": 0, "code": str("DataType only Image or Video!")}
            _ = sendMsg("http://{}:{}/api/binary/trainBinLog".format(srvIp, srvPort), output)
            # res = requests.post(server, headers=headers, data=json.dumps(output))

        trainDataPath = os.path.join(trainPath, "train.tfrecord")
        valDataPath = os.path.join(trainPath, "val.tfrecord")
        lenClasses = makeClassesFile(classes, aiCd)
        with open(os.path.join(aiPath, aiCd, "{}_classes.txt".format(aiCd)), "w") as f:
            for tmp in classes:
                f.write("{}\n".format(tmp))

        log.info("DONE MAKE DATASET")

        return trainDataPath, valDataPath, classes, lenClasses

    except tf.errors.ResourceExhaustedError as err:
        log.error(err)
        output = {"STATUS": 0, "code": str(err)}
        _ = sendMsg("http://{}:{}/api/binary/trainBinLog".format(srvIp, srvPort), output)

    except Exception as e:
        log.error(e)
        log.error(traceback.format_exc())
        output = {"STATUS": 0, "code": str(e)}
        _ = sendMsg("http://{}:{}/api/binary/trainBinLog".format(srvIp, srvPort), output)


def retinanetDataset(imageList, dataType):
    return 0


def deeplabDataset(imageList, classesData, dataType, aiCd, batchSize):
    try:
        dataType = dataType.upper()
        classes = []
        makeDir(os.path.join(aiPath, aiCd))
        if dataType == "I":
            tmpPath = []
            imageFiles = []
            maskFiles = []
            for t in classesData:
                classes.append(t["CLASS_CD"])

            for data in imageList:
                tmpPath.append(data["IMAGE_PATH"])

            lenClasses = len(classes)
            with open(os.path.join(aiPath, aiCd, "{}_classes.txt".format(aiCd)), "w") as f:
                for tmp in classes:
                    f.write("{}\n".format(tmp))

            for ttmp in tmpPath:
                for i in range(3000):
                    imageFiles.append(ttmp)
                    imgPath = os.path.dirname(ttmp)
                    maskName = os.path.basename(ttmp)
                    maskName = 'MASK_{}'.format(maskName)

                    fileName, fileExtension = os.path.splitext(maskName)
                    maskName = maskName.replace(fileExtension, ".png")

                    maskPath = os.path.join(imgPath, maskName)
                    maskFiles.append(maskPath)

            fileCnt = len(imageFiles)
            train = int(len(imageFiles) * 0.7)
            trainImageFiles = imageFiles[:train - 1]
            testImageFiles = imageFiles[train:]

            trainMaskFiles = maskFiles[:train - 1]
            testMaskFiles = maskFiles[train:]

            trainGen = makeGen(batchSize, trainImageFiles, trainMaskFiles)
            ValiGen = makeGen(batchSize, testImageFiles, testMaskFiles)

            return trainGen, ValiGen, classes, lenClasses, fileCnt

    except Exception as e:
        log.error(traceback.format_exc())
        output = {"STATUS": 0, "code": str(e)}
        _ = sendMsg("http://{}:{}/api/binary/trainBinLog".format(srvIp, srvPort), output)


def makeGen(batchSize, imageFiles, maskFiles):
    c = 0
    while True:
        img = np.zeros((batchSize, 512, 512, 3)).astype('float')
        mask = np.zeros((batchSize, 512, 512, 1)).astype('float')

        for i in range(c, c + batchSize):
            train_img = cv2.imread(imageFiles[i]) / 255.
            train_img = cv2.resize(train_img, (512, 512))

            img[i - c] = train_img

            train_mask = cv2.imread(maskFiles[i], cv2.IMREAD_GRAYSCALE) / 255.
            train_mask = cv2.resize(train_mask, (512, 512))
            train_mask = train_mask.reshape(512, 512, 1)

            mask[i - c] = train_mask

        c += batchSize
        if(c + batchSize >= len(imageFiles)):
            c = 0

        yield img, mask


def makeClassesFile(classes, aiCd):
    makeDir(os.path.join(aiPath, aiCd))
    saveClassFile = os.path.join(aiPath, aiCd, "{}_classes.txt".format(aiCd))
    with open(saveClassFile, "w") as f:
        try:
            for i, label in enumerate(classes):
                tmp = "{}\n".format(label)
                f.write(tmp)

        except Exception as e:
            log.error(e)
            log.error(traceback.format_exc())
            output = {"STATUS": 0, "code": str(e)}
            _ = sendMsg("http://{}:{}/api/binary/trainBinLog".format(srvIp, srvPort), output)

    return len(classes)


def predictDataset(dataType, objectType, imgPath):
    try:
        if dataType == "I":
            if objectType == "C":
                img = cv2.imread(imgPath)
                oriH, oriW = img.shape[:2]

                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
                img = img.reshape(1, 512, 512, 3)
                resizeH, resizeW = img.shape[:2]
                return img, oriH, oriW, resizeH, resizeW

            elif objectType == "D":
                img_raw = tf.image.decode_image(open(imgPath, "rb").read(), channels=3)
                oriH, oriW = img_raw.shape[:2]
                img = tf.expand_dims(img_raw, 0)
                img = transform_images(img, 416)
                resizeH, resizeW = 416, 416

                return img, oriH, oriW, resizeH, resizeW

            elif objectType == "S":
                # img = np.array(Image.open(imgPath))
                img = cv2.imread(imgPath)
                # img = cv2.imread(imgPath)
                oriH, oriW, _ = img.shape
                resizeH, resizeW = 512, 512

                return img, oriH, oriW, resizeH, resizeW

        elif dataType == "V":
            if objectType == "C":
                img = imgPath
                oriH, oriW = img.shape[:2]
                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
                img = img.reshape(1, 512, 512, 3)
                resizeH, resizeW = img.shape[:2]

                return img, oriH, oriW, resizeH, resizeW

            elif objectType == "D":
                # img_raw = tf.image.decode_image(imgPath, channels=3)
                img_raw = imgPath
                oriH, oriW = img_raw.shape[:2]
                img = tf.expand_dims(img_raw, 0)
                img = transform_images(img, 416)
                resizeH, resizeW = 416, 416
                return img, oriH, oriW, resizeH, resizeW

            elif objectType == "S":
                img = imgPath
                # img = cv2.imread(imgPath)
                oriH, oriW, _ = img.shape
                resizeH, resizeW = 512, 512

                return img, oriH, oriW, resizeH, resizeW

    except Exception as e:
        log.error(traceback.format_exc())
        output = {"STATUS": 0, "code": str(e)}
        _ = sendMsg("http://{}:{}/api/binary/trainBinLog".format(srvIp, srvPort), output)

# if __name__ == "__main__":
#     imageList = []
#     datas = {
#         "CLASS_NO":
#         12,
#         "CLASS_NAME":
#         "car",
#         "TAGS": {},
#         "files": [
#             "/Users/upload/DataSets/CI200004/car/IMG_3727.jpeg",
#             "/Users/upload/DataSets/CI200004/car/IMG_3726.jpeg",
#             "/Users/upload/DataSets/CI200004/car/IMG_3724.jpeg"
#         ]
#     }
#     imageList.append(datas)
#     datas = {
#         "CLASS_NO":
#         13,
#         "CLASS_NAME":
#         "person",
#         "TAGS": {},
#         "files": [
#             "/Users/upload/DataSets/CI200004/person/IMG_3907.jpeg",
#             "/Users/upload/DataSets/CI200004/person/IMG_3996.jpeg",
#             "/Users/upload/DataSets/CI200004/person/IMG_3882.jpeg"
#         ]
#     }
#     imageList.append(datas)
#     aiCd = 0000000

#     classificationDataset(imageList, aiCd)
