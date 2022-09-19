#-*- coding:utf-8 -*-
'''
이미지 프로세싱 관련 함수들
1. rotate(img, angle) : 이미지 로테이션
2. adjustBrightness(img, value) : 이미지 밝기 조절
3. colorConvert(img, colorType) : 이미지 컬러 정보 변경 / RGB, GRAY, HSV, YCRCB
4. colorReverse(img) : 색상 반전
5. gammaCorection(img, gamma) : 감마 보정(1이 기준, 1보다 작을수록 어두워지고, 높을수록 밝아짐)
6. histogramEqul(img) : 히스토그램 평활화
7. noiseFiltering(img) : MeanDenoising 기반 노이즈 필터링
8. blurring(img, blurType) : 이미지 블러링 / gaussian, bilateral, median, average
9. sharpening(img) : 이미지 샤프닝
10. binarization(img, value) : 영상 이진화
11. resize(img, factor) : 영상 크기 조절(1.0이 기준, 1.0보다 작아질수록 이미지는 작아지고, 커질수록 이미지는 커짐)
12. flip(img, flipType) : 영상 반전(h : horizontal, v : vertical)
13. crop(img, cropSize=100) : 영상 크롭
14. getThumbnail() : Thumbnail 만드는 함수
   - image : 200,200 리사이즈, 이미지 저장
   - video : 100번째 프레임 200,200 리사이즈, 이미지 저장, fps 전달
'''

import cv2
import os
import sys
import numpy as np
import traceback

# Model 폴더 찾기 위한 방법
basePath = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# Model Path 등록
sys.path.append(basePath)

from Common.Logger.Logger import logger

log = logger("log")


# 이미지 회전
def rotate(img, angle):
    try:
        h, w, c = img.shape
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        dst = cv2.warpAffine(img, matrix, (w, h))
        return dst

    except Exception as e:
        log.error(traceback.format_exc())
        log.error(e)


# 밝기 조절
def adjustBrightness(img, value=0):
    try:
        num_channels = 1 if len(img.shape) < 3 else 1 if img.shape[-1] == 1 else 3
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if num_channels == 1 else img
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        if value >= 0:
            lim = 255 - value
            v[v > lim] = 255
            # v[v <= lim] += value
            np.add(v[v <= lim], value, out=v[v <= lim], casting="unsafe")

        else:
            value = int(-value)
            lim = 0 + value
            v[v < lim] = 0
            v[v >= lim] -= value
            np.add(v[v >= lim], value, out=v[v >= lim], casting="unsafe")

        final_hsv = cv2.merge((h, s, v))

        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if num_channels == 1 else img
        return img

    except Exception as e:
        log.error(traceback.format_exc())
        log.error(e)


# 이미지 컬러 정보 변경(RGB, GRAY, HSV, YCbCr)
def colorConvert(img, colorType):
    colorType = colorType.upper()
    try:
        # get image shape
        h, w, c = img.shape
        if c == 3:
            if colorType == 'RGB':
                log.warning("Color is already RGB")
                dst = img
            elif colorType == 'GRAY':
                dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif colorType == "HSV":
                dst = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            elif colorType == "YCBCR":
                dst = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            else:
                log.error("Color Type ERROR!")
                return

        elif c == 1:
            if colorType == 'RGB':
                dst = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif colorType == 'GRAY':
                log.warning("Color is already GRAY")
                dst = img
            else:
                log.error("Color Type ERROR. Grayscale cannot convert HSV and YCbCr")
                return

        else:
            log.error("Cannot read color space")
            return

        return dst

    except Exception as e:
        log.error(traceback.format_exc())
        log.error(e)


# 색상 반전(grayscale)
def colorReverse(img):
    try:
        # get image shape
        h, w, c = img.shape
        dst = ~img
        return dst

    except Exception as e:
        log.error(traceback.format_exc())
        log.error(e)


# 감마보정
def gammaCorection(img, gamma=1.0):
    try:
        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        dst = cv2.LUT(img, lookUpTable)
        return dst

    except Exception as e:
        log.error(traceback.format_exc())
        log.error(e)


# 히스토그램 평활화
def histogramEqul(img):
    try:
        # get image shape
        h, w, c = img.shape
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        if c == 1:
            dst = clahe.append(img)
        else:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            labPlanes = cv2.split(lab)
            labPlanes[0] = clahe.apply(labPlanes[0])
            lab = cv2.merge(labPlanes)
            dst = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return dst

    except Exception as e:
        log.error(traceback.format_exc())
        log.error(e)


# 노이즈 필터링
def noiseFiltering(img):
    try:
        # get image shape
        h, w, c = img.shape
        if c == 1:
            dst = cv2.fastNlMeansDenoising(img, None, 10, 10, 7, 21)
        else:
            dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        return dst

    except Exception as e:
        log.error(traceback.format_exc())
        log.error(e)


# 가우시안 블러
def blurring(img, blurType="gaussian"):
    try:
        blurType = blurType.lower()
        if blurType == 'gaussian':
            dst = cv2.GaussianBlur(img, (5, 5), 0)
        elif blurType == 'bilateral':
            dst = cv2.bilateralFilter(img, 9, 75, 75)
        elif blurType == 'median':
            dst = cv2.medianBlur(img, 5)
        elif blurType == 'avgerage':
            dst = cv2.blur(img, (5, 5))

        return dst
    except Exception as e:
        log.error(traceback.format_exc())
        log.error(e)


def sharpening(img):
    try:
        filter = np.array([[-1, -1, -1, -1, -1],
                           [-1, 2, 2, 2, -1],
                           [-1, 2, 9, 2, -1],
                           [-1, 2, 2, 2, -1],
                           [-1, -1, -1, -1, -1]]) / 9.0

        dst = cv2.filter2D(img, -1, filter)
        return dst

    except Exception as e:
        log.error(traceback.format_exc())
        log.error(e)


# 영상 이진화
def binarization(img, value=127):
    try:
        # get image shape
        h, w, c = img.shape
        # image channel 3 is COLOR ==> change grayscale
        if c == 3:
            tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, dst = cv2.threshold(tmp, value, 255, cv2.THRESH_BINARY)
        return dst

    except Exception as e:
        log.error(traceback.format_exc())
        log.error(e)


# 영상 크기 조절
def resize(img, factor=1.0):
    try:
        if factor < 0.0:
            log.error("영상 크기는 음수를 사용할 수 없습니다.")
            return
        elif factor > 1.0:
            interpolation = cv2.INTER_CUBIC
        elif factor <= 1.0:
            interpolation = cv2.INTER_AREA

        dst = cv2.resize(img, dsize=(0, 0), fx=factor, fy=factor, interpolation=interpolation)
        return dst
    except Exception as e:
        log.error(traceback.format_exc())
        log.error(e)


# 영상 반전
def flip(img, flipType='h'):
    try:
        flipType = flipType.lower()

        if flipType == 'h':
            dst = cv2.flip(img, 0)
        elif flipType == 'v':
            dst = cv2.flip(img, 1)
        else:
            log.error("Chack Flip Code (V : Vertical, H : Horizontal)")
            return
        return dst

    except Exception as e:
        log.error(traceback.format_exc())
        log.error(e)


def crop(img, cropSize=100):
    try:
        h, w, c = img.shape
        dst = img[int(h / 2) - int(cropSize / 2):int(h / 2) + int(cropSize / 2),
                  int(w / 2) - int(cropSize / 2):int(w / 2) + int(cropSize / 2)]

        return dst
    except Exception as e:
        log.error(traceback.format_exc())
        log.error(e)


def getThumbnail(MDL_TYPE, imgPath, savePath):
    err = {"MSG": None, "ERROR_FILE": None}
    frameRate = 0
    try:
        if MDL_TYPE == "i":
            img = cv2.imread(imgPath)
            img = cv2.resize(img, (200, 200), cv2.INTER_CUBIC)
            cv2.imwrite(savePath, img)

        elif MDL_TYPE == 'v':
            vc = cv2.VideoCapture(imgPath)
            frameRate = vc.get(cv2.CAP_PROP_FPS)

            vc.set(cv2.CAP_PROP_POS_FRAMES, 100)
            ok, frame = vc.read()

            if ok is False:
                vc.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = vc.read()

            frame = cv2.resize(frame, (200, 200), cv2.INTER_CUBIC)
            cv2.imwrite(savePath, frame)
        return frameRate, err

    except Exception:
        err = {"MSG": "Thumbnail Error", "ERROR_FILE": imgPath}
        log.error(traceback.format_exc())
        return frameRate, err
