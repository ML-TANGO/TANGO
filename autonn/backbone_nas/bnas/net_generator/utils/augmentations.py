# References:
# [1] YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Image augmentation functions
"""

import random

import cv2
import numpy as np

from .metrics import bbox_ioa


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    '''HSV color-space augmentation'''
    if hgain or sgain or vgain:
        _r = np.random.uniform(-1, 1, 3) * \
            [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype  # uint8

        _x = np.arange(0, 256, dtype=_r.dtype)
        lut_hue = ((_x * _r[0]) % 180).astype(dtype)
        lut_sat = np.clip(_x * _r[1], 0, 255).astype(dtype)
        lut_val = np.clip(_x * _r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(
            sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def hist_equalize(img, clahe=True, bgr=False):
    '''
    Equalize histogram on BGR image 'im'
    with im.shape(n,m,3) and range 0-255
    '''
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        _c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = _c.apply(yuv[:, :, 0])
    else:
        # equalize Y channel histogram
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    # convert YUV image to RGB
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)


def replicate(img, labels):
    '''Replicate labels'''
    _h, _w = img.shape[:2]
    boxes = labels[:, 1:].astype(int)
    _x1, _y1, _x2, _y2 = boxes.T
    _s = ((_x2 - _x1) + (_y2 - _y1)) / 2  # side length (pixels)
    for i in _s.argsort()[:round(_s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        _bh, _bw = y2b - y1b, x2b - x1b
        _yc, _xc = int(random.uniform(0, _h - _bh)
                       ), int(random.uniform(0, _w - _bw))  # offset x, y
        x1a, y1a, x2a, y2a = [_xc, _yc, _xc + _bw, _yc + _bh]
        # im4[ymin:ymax, xmin:xmax]
        img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        labels = np.append(
            labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return img, labels


def letterbox(img, new_shape=(640, 640),
              color=(114, 114, 114), auto=True,
              scale_fill=False, scaleup=True,
              stride=32):
    '''
    Resize and pad image while meeting stride-multiple constraints
    '''
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    _r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        _r = min(_r, 1.0)

    # Compute padding
    ratio = _r, _r  # width, height ratios
    new_unpad = int(round(shape[1] * _r)), int(round(shape[0] * _r))
    _dw, _dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        _dw, _dh = np.mod(_dw, stride), np.mod(_dh, stride)  # wh padding
    elif scale_fill:  # stretch
        _dw, _dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / \
            shape[0]  # width, height ratios

    _dw /= 2  # divide padding into 2 sides
    _dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(_dh - 0.1)), int(round(_dh + 0.1))
    left, right = int(round(_dw - 0.1)), int(round(_dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (_dw, _dh)


def copy_paste(img, labels, segments, _p=0.5):
    '''
    Implement Copy-Paste augmentation
    https://arxiv.org/abs/2012.07177,
    labels as nx5 np.array(cls, xyxy)
    '''
    _n = len(segments)
    if _p and _n:
        _, _w, _ = img.shape  # height, width, channels
        im_new = np.zeros(img.shape, np.uint8)
        for j in random.sample(range(_n), k=round(_p * _n)):
            _l, _s = labels[j], segments[j]
            box = _w - _l[3], _l[2], _w - _l[1], _l[4]
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                labels = np.concatenate((labels, [[_l[0], *box]]), 0)
                segments.append(np.concatenate(
                    (_w - _s[:, 0:1], _s[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[j].astype(
                    np.int32)], -1, (255, 255, 255), cv2.FILLED)

        result = cv2.bitwise_and(src1=img, src2=im_new)
        result = cv2.flip(result, 1)  # augment segments (flip left-right)
        i = result > 0  # pixels to replace
        # i[:, :] = result.max(2).reshape(h, w, 1)  # act over ch
        img[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug

    return img, labels, segments


def cutout(img, labels, _p=0.5):
    '''
    Applies image cutout augmentation
    https://arxiv.org/abs/1708.04552
    '''
    if random.random() < _p:
        _h, _w = img.shape[:2]
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + \
            [0.0625] * 8 + [0.03125] * 16  # image size fraction
        for _s in scales:
            mask_h = random.randint(1, int(_h * _s))  # create random masks
            mask_w = random.randint(1, int(_w * _s))

            # box
            xmin = max(0, random.randint(0, _w) - mask_w // 2)
            ymin = max(0, random.randint(0, _h) - mask_h // 2)
            xmax = min(_w, xmin + mask_w)
            ymax = min(_h, ymin + mask_h)

            # apply random color mask
            img[ymin:ymax, xmin:xmax] = [
                random.randint(64, 191) for _ in range(3)]

            # return unobscured labels
            if labels and _s > 0.03:
                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
                labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def mixup(img, labels, im2, labels2):
    '''
    Applies MixUp augmentation
    https://arxiv.org/pdf/1710.09412.pdf
    '''
    # mixup ratio, alpha=beta=32.0
    _r = np.random.beta(32.0, 32.0)
    img = (img * _r + im2 * (1 - _r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return img, labels


# box1(4,n), box2(4,n)
def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
    '''
    Compute candidate boxes: box1 before augment,
    box2 after augment, wh_thr (pixels),
    aspect_ratio_thr, area_ratio
    '''
    _w1, _h1 = box1[2] - box1[0], box1[3] - box1[1]
    _w2, _h2 = box2[2] - box2[0], box2[3] - box2[1]
    _ar = np.maximum(_w2 / (_h2 + eps), _h2 / (_w2 + eps))  # aspect ratio
    # candidates
    return ((_w2 > wh_thr) &
            (_h2 > wh_thr) &
            (_w2 * _h2 / (_w1 * _h1 + eps) > area_thr) &
            (_ar < ar_thr))
