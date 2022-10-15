# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""

import math
import warnings
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from PIL import Image
from torch.cuda import amp

from datasets import letterbox
from utils.general import (LOGGER, increment_path, make_divisible,
                             non_max_suppression, scale_coords, xyxy2xywh)
from utils.plots import Annotator, save_one_box
from utils.torch_utils import copy_attr, time_sync


def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of Pillow exif_transpose()
    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orient = exif.get(0x0112, 1)  # default 1
    if orient > 1:
        method = {
            2: Image.FLIP_LEFT_RIGHT,
            3: Image.ROTATE_180,
            4: Image.FLIP_TOP_BOTTOM,
            5: Image.TRANSPOSE,
            6: Image.ROTATE_270,
            7: Image.TRANSVERSE,
            8: Image.ROTATE_90, }.get(orient)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image

def autopad(k, _p=None):  # kernel, padding
    '''
    autopad
    '''
    # Pad to 'same'
    if _p is None:
        _p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return _p


class Conv(nn.Module):
    '''
    Standard convolution
    ch_in, ch_out, kernel, stride, padding, groups
    '''
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p),
                              groups=g, bias=False)
        self._bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, _x):
        '''
        forward
        '''
        return self.act(self._bn(self.conv(_x)))

    def forward_fuse(self, _x):
        '''
        forward fuse
        '''
        return self.act(self.conv(_x))


class DWConv(Conv):
    '''
    Depth-wise convolution class
    ch_in, ch_out, kernel, stride, padding, groups
    '''
    def __init__(self, c1, c2, k=1, s=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Bottleneck(nn.Module):
    '''
    Standard bottleneck
    ch_in, ch_out, shortcut, groups, expansion
    '''
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        _c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, _c, 1, 1)
        self.cv2 = Conv(_c, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, _x):
        '''
        forward
        '''
        return (_x + self.cv2(self.cv1(_x))
                if self.add else self.cv2(self.cv1(_x)))


class C3(nn.Module):
    '''
    CSP Bottleneck with 3 convolutions
    ch_in, ch_out, number, shortcut, groups, expansion
    '''
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        _c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, _c, 1, 1)
        self.cv2 = Conv(c1, _c, 1, 1)
        self.cv3 = Conv(2 * _c, c2, 1)  # optional act=FReLU(c2)
        self._m = nn.Sequential(
            *(Bottleneck(_c, _c, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, _x):
        '''
        forward
        '''
        return self.cv3(torch.cat((self._m(self.cv1(_x)), self.cv2(_x)), 1))


class SPPF(nn.Module):
    '''
    Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    '''

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        _c = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, _c, 1, 1)
        self.cv2 = Conv(_c * 4, c2, 1, 1)
        self._m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, _x):
        '''
        forward
        '''
        _x = self.cv1(_x)
        with warnings.catch_warnings():
            # suppress torch 1.9.0 max_pool2d() warning
            warnings.simplefilter('ignore')
            _y1 = self._m(_x)
            _y2 = self._m(_y1)
            return self.cv2(torch.cat((_x, _y1, _y2, self._m(_y2)), 1))


class Concat(nn.Module):
    '''
    Concatenate a list of tensors along dimension
    '''

    def __init__(self, dimension=1):
        super().__init__()
        self._d = dimension

    def forward(self, _x):
        '''
        forward
        '''
        return torch.cat(_x, self._d)


class AutoShape(nn.Module):
    '''
    YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs.
    Includes preprocessing, inference and NMS
    '''
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    agnostic = False  # NMS class-agnostic
    multi_label = False  # NMS multiple labels per box
    # (optional list) filter by class,
    # i.e. = [0, 15, 16] for COCO persons, cats and dogs
    classes = None
    max_det = 1000  # maximum number of detections per image
    amp = False  # Automatic Mixed Precision (AMP) inference

    def __init__(self, model, verbose=True):
        super().__init__()
        if verbose:
            LOGGER.info('Adding AutoShape... ')
        copy_attr(self, model,
                  include=('yaml', 'nc', 'hyp', 'names',
                           'stride', 'abc'),
                  exclude=())  # copy attributes
        # self.dmb = isinstance(model, DetectMultiBackend)
        # DetectMultiBackend() instance
        self.dmb = False
        self._pt = True  # PyTorch model
        self.model = model.eval()

    def _apply(self, fn):
        '''
        Apply to(), cpu(), cuda(), half() to model tensors
        that are not parameters or registered buffers
        '''
        # self = super()._apply(fn)
        super()._apply(fn)
        if self._pt:
            # Detect()
            _m = (self.model.model.model[-1]
                  if self.dmb else self.model.model[-1])
            _m.stride = fn(_m.stride)
            _m.grid = list(map(fn, _m.grid))
            if isinstance(_m.anchor_grid, list):
                _m.anchor_grid = list(map(fn, _m.anchor_grid))
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        '''
        Inference from various sources.
        For height=640, width=1280, RGB images example inputs are:
          file:       imgs = 'data/images/zidane.jpg'  # str or PosixPath
          URI:             = 'https://ultralytics.com/images/zidane.jpg'
          OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]
                            # HWC BGR to RGB x(640,1280,3)
          PIL:             = Image.open('image.jpg') or ImageGrab.grab()
                            # HWC x(640,1280,3)
          numpy:           = np.zeros((640,1280,3))  # HWC
          torch:           = torch.zeros(16,3,320,640)
                            # BCHW (scaled to size=640, 0-1 values)
          multiple:        = [Image.open('image1.jpg'),
                              Image.open('image2.jpg'), ...]
                              # list of images
        '''
        vdict = edict()

        vdict.t = [time_sync()]
        vdict.p = next(self.model.parameters()) if self._pt else torch.zeros(
            1, device=self.model.device)  # for device, type
        # Automatic Mixed Precision (AMP) inference
        autocast = self.amp and (vdict.p.device.type != 'cpu')
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(autocast):
                # inference
                return self.model(
                    imgs.to(vdict.p.device).type_as(vdict.p),
                    augment, profile)

        # Pre-process
        vdict.n, imgs = (len(imgs), list(imgs)) if isinstance(
            imgs, (list, tuple)) else (1, [imgs])  # number, list of images
        # image and inference shapes, filenames
        vdict.shape0, vdict.shape1, files = [], [], []
        for i, img in enumerate(imgs):
            _f = f'image{i}'  # filename
            if isinstance(img, (str, Path)):  # filename or uri
                img, _f = Image.open(requests.get(img, stream=True).raw if str(
                    img).startswith('http') else img), img
                img = np.asarray(exif_transpose(img))
            elif isinstance(img, Image.Image):  # PIL Image
                img, _f = np.asarray(exif_transpose(img)), getattr(
                    img, 'filename', _f) or _f
            files.append(Path(_f).with_suffix('.jpg').name)
            if img.shape[0] < 5:  # image in CHW
                # reverse dataloader .transpose(2, 0, 1)
                img = img.transpose((1, 2, 0))
            img = img[..., :3] if img.ndim == 3 else np.tile(
                img[..., None], 3)  # enforce 3ch input
            _s = img.shape[:2]  # HWC
            vdict.shape0.append(_s)  # image shape
            _g = (size / max(_s))  # gain
            vdict.shape1.append([_y * _g for _y in _s])
            imgs[i] = img if img.data.contiguous else np.ascontiguousarray(
                img)  # update
        vdict.shape1 = [make_divisible(x, self.stride)
                        if self._pt
                        else size
                        for x in np.array(
                            vdict.shape1).max(0)]  # inf shape
        vdict.x = [letterbox(img, vdict.shape1, auto=False)[0]
                   for img in imgs]  # pad
        vdict.x = np.ascontiguousarray(np.array(vdict.x).transpose(
            (0, 3, 1, 2)))  # stack and BHWC to BCHW
        vdict.x = torch.from_numpy(vdict.x)
        vdict.x = vdict.x.to(vdict.p.device).type_as(vdict.p) / \
            255  # uint8 to fp16/32
        vdict.t.append(time_sync())

        with amp.autocast(autocast):
            # Inference
            _y = self.model(vdict.x, augment, profile)  # forward
            vdict.t.append(time_sync())

            # Post-process
            _y = non_max_suppression(_y if self.dmb else _y[0],
                                     self.conf,
                                     self.iou,
                                     self.classes,
                                     self.agnostic,
                                     self.multi_label,
                                     max_det=self.max_det)  # NMS
            for i in range(vdict.n):
                scale_coords(vdict.shape1, _y[i][:, :4], vdict.shape0[i])

            vdict.t.append(time_sync())
            det_args = edict()
            det_args.times = vdict.t
            det_args.names = self.names
            det_args.shape = vdict.x.shape
            return Detections(imgs, _y, files, det_args)


class Detections:
    '''
    YOLOv5 detections class for inference results
    '''
    def __init__(self, imgs, pred, files, args):
        super().__init__()
        self.args = args
        _d = pred[0].device  # device
        _gn = [torch.Tensor(
            [*(img.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=_d)
               for img in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.files = files  # image filenames
        self.args.xyxy = pred  # xyxy pixels
        self.args.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.args.xyxyn = [x / g for x,
                           g in zip(self.args.xyxy, _gn)]  # xyxy normalized
        self.args.xywhn = [x / g for x,
                           g in zip(self.args.xywh, _gn)]  # xywh normalized
        self.args.n = len(self.pred)  # number of images (batch size)
        self.args.t = tuple((args.times[i + 1] - args.times[i]) * 1000 /
                            self.args.n for i in range(3))  # timestamps (ms)

        self.dsp_args = edict()
        self.dsp_args.pprint = False
        self.dsp_args.show = False
        self.dsp_args.save = False
        self.dsp_args.crop = False
        self.dsp_args.render = False
        self.dsp_args.labels = True
        self.dsp_args.save_dir = Path('')

    def display(self):
        '''
        display
        '''
        crops = []
        for i, (img, pred) in enumerate(zip(self.imgs, self.pred)):
            # string
            _s = f'image {i + 1}/{len(self.pred)}: \
                {img.shape[0]}x{img.shape[1]} '
            if pred.shape[0]:
                for _c in pred[:, -1].unique():
                    _n = (pred[:, -1] == _c).sum()  # detections per class
                    # add to string
                    _s += f"{_n} {self.args.names[int(_c)]}{'s' * (_n > 1)}, "
                if ((self.dsp_args.show or
                     self.dsp_args.save or
                     self.dsp_args.render or
                     self.dsp_args.crop)):
                    annotator = Annotator(img, example=str(self.args.names))
                    # xyxy, confidence, class
                    for *box, conf, cls in reversed(pred):
                        label = f'{self.args.names[int(cls)]} {conf:.2f}'
                        if self.dsp_args.crop:
                            file = self.dsp_args.save_dir / 'crops' / \
                                self.args.names[int(cls)] / \
                                self.files[i] if self.dsp_args.save else None
                            crops.append({
                                'box': box,
                                'conf': conf,
                                'cls': cls,
                                'label': label,
                                'im': save_one_box(box, img, file=file,
                                                   save=self.dsp_args.save)})
                        else:  # all others
                            annotator.box_label(
                                box, label if self.dsp_args.labels else '')
                    img = annotator.img
            else:
                _s += '(no detections)'

            img = Image.fromarray(img.astype(np.uint8)) if isinstance(
                img, np.ndarray) else img  # from np
            if self.dsp_args.pprint:
                print(_s.rstrip(', '))
            if self.dsp_args.show:
                img.show(self.files[i])  # show
            if self.dsp_args.save:
                _f = self.files[i]
                img.save(self.dsp_args.save_dir / _f)  # save
                # if i == self.n - 1:
                #    LOGGER.info(
                #        f"Saved {self.n} image{'s' * (self.n > 1)} \
                #          to {colorstr('bold', self.dsp_args.save_dir)}")
            if self.dsp_args.render:
                self.imgs[i] = np.asarray(img)
        return crops

    def print(self):
        '''
        print
        '''
        self.dsp_args.pprint = True
        self.display()  # print results
        self.dsp_args.pprint = False
        print(
            f'Speed: %.1fms pre-process, %.1fms inference, \
                %.1fms NMS per image at shape \
                    {tuple(self.args.shape)}' % self.args.t)

    def show(self, labels=True):
        '''
        show
        '''
        self.dsp_args.show = True
        self.dsp_args.labels = labels
        self.display()  # show results
        self.dsp_args.show = False
        self.dsp_args.labels = True

    def save(self, labels=True, save_dir='runs/detect/exp'):
        '''
        save
        '''
        self.dsp_args.show = True
        self.dsp_args.labels = labels
        self.dsp_args.save_dir = increment_path(
            save_dir, exist_ok=save_dir != 'runs/detect/exp',
            mkdir=True)  # increment save_dir
        self.display()  # save results
        self.dsp_args.show = False
        self.dsp_args.labels = True
        self.dsp_args.save_dir = Path('')

    def crop(self, save=True, save_dir='runs/detect/exp'):
        '''
        cropping
        '''
        self.dsp_args.crop = True
        self.dsp_args.save = save
        self.dsp_args.save_dir = increment_path(
            save_dir, exist_ok=save_dir != 'runs/detect/exp',
            mkdir=True) if save else None
        # crop results
        crops = self.display()
        self.dsp_args.crop = False
        self.dsp_args.save = False
        self.dsp_args.save_dir = Path('')
        return crops

    def render(self, labels=True):
        '''
        rendering
        '''
        self.dsp_args.render = True
        self.dsp_args.labels = labels
        self.display()  # render results
        self.dsp_args.render = False
        self.dsp_args.labels = True
        return self.imgs

    def pandas(self):
        '''
        return detections as pandas DataFrames,
        i.e. print(results.pandas().xyxy[0])
        '''
        new = copy(self)  # return copy
        _ca = ('xmin', 'ymin', 'xmax', 'ymax',
               'confidence', 'class', 'name')  # xyxy columns
        _cb = ('xcenter', 'ycenter', 'width', 'height',
               'confidence', 'class', 'name')  # xywh columns
        for _k, _c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'],
                          [_ca, _ca, _cb, _cb]):
            # update
            _a = [[_x[:5] + [int(_x[5]),
                             self.args.names[int(_x[5])]]
                   for _x in _x.tolist()] for _x in getattr(self, _k)]
            setattr(new, _k, [pd.DataFrame(_x, columns=_c) for _x in _a])
        return new

    def tolist(self):
        '''
        return a list of Detections objects,
        i.e. 'for result in results.tolist():'
        '''
        _r = range(self.args.n)  # iterable
        det_args = edict()
        det_args.times = self.args.times
        det_args.names = self.args.names
        det_args.shape = self.args.shape
        _x = [Detections([self.imgs[i]], [self.pred[i]], [
            self.files[i]], det_args) for i in _r]
        # for d in x:
        #    for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
        #        setattr(d, k, getattr(d, k)[0])  # pop out of list
        return _x

    def __len__(self):
        return self.args.n  # override len(results)

    def __str__(self):
        self.print()  # override print(results)
        return ''


class Classify(nn.Module):
    '''
    Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    ch_in, ch_out, kernel, stride, padding, groups
    '''

    def __init__(self, kernel_size=1, padding=None, **kwargs):
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(padding=autopad(kernel_size, padding), **kwargs)
        # self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p),
        #                      groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, _x):
        '''
        forward
        '''
        _z = torch.cat([self.aap(_y) for _y in (
            _x if isinstance(_x, list) else [_x])], 1)  # cat if list
        return self.flat(self.conv(_z))  # flatten to x(b,c2)
