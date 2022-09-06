# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules
Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import math
from ast import literal_eval
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from easydict import EasyDict as edict

from ..utils.autoanchor import check_anchor_order
from ..utils.general import LOGGER, check_version, make_divisible
from ..utils.plots import feature_visualization
from ..utils.torch_utils import (fuse_conv_and_bn,  # , time_sync
                                 initialize_weights)

from .common import C3, Bottleneck, Concat, Conv, DWConv

# try:
#     import thop  # for FLOPs computation
# except ImportError:
#     thop = None


class Detect(nn.Module):
    """
    head of yolo
    """
    stride = []  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter
    export = False  # export mode

    def __init__(self, _nc=80, anchors=(), _ch=(), inplace=True):
        super().__init__()
        det_dict = edict()
        det_dict.nc = _nc  # number of classes
        det_dict.no = _nc + 5  # number of outputs per anchor
        det_dict.nl = len(anchors)  # number of detection layers
        det_dict.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * det_dict.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * det_dict.nl  # init anchor grid
        self.register_buffer('anchors', torch.Tensor(
            anchors).float().view(det_dict.nl, -1, 2))  # shape(nl,na,2)
        self._m = nn.ModuleList(nn.Conv2d(x, det_dict.no * det_dict.na, 1)
                                for x in _ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)
        self.det_dict = det_dict

    def forward(self, _x):
        """
        forward
        """
        _z = []  # inference output
        for i in range(self.det_dict.nl):
            _x[i] = self._m[i](_x[i])  # conv
            # x(bs,255,20,20) to x(bs,3,20,20,85)
            _bs, _, _ny, _nx = _x[i].shape
            _x[i] = _x[i].view(_bs, self.det_dict.na,
                               self.det_dict.no, _ny,
                               _nx).permute(0, 1, 3,
                                            4, 2).contiguous()

            if not self.training:  # inference
                if ((self.onnx_dynamic or
                     self.grid[i].shape[2:4] != _x[i].shape[2:4])):
                    self.grid[i], self.anchor_grid[i] = self._make_grid(
                        _nx, _ny, i)

                _y = _x[i].sigmoid()
                if self.inplace:
                    _y[..., 0:2] = (_y[..., 0:2] * 2 + self.grid[i]) \
                        * self.stride[i]  # xy
                    _y[..., 2:4] = (_y[..., 2:4] * 2) ** 2 * \
                        self.anchor_grid[i]  # wh
                else:
                    # for YOLOv5 on AWS Inferentia
                    # https://github.com/ultralytics/yolov5/pull/2953
                    # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    _xy, _wh, conf = _y.split((2, 2, self.det_dict.nc + 1), 4)
                    _xy = (_xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    _wh = (_wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    _y = torch.cat((_xy, _wh, conf), 4)
                _z.append(_y.view(_bs, -1, self.det_dict.no))

        return (_x
                if self.training
                else (torch.cat(_z, 1),)
                if self.export
                else (torch.cat(_z, 1), _x))

    def _make_grid(self, _nx=20, _ny=20, i=0):
        """
        _make_grid 
        YOLOv5 model
        model, input channels, number of classes
        """
        _d = self.anchors[i].device
        _t = self.anchors[i].dtype
        shape = 1, self.det_dict.na, _ny, _nx, 2  # grid shape
        _y, _x = torch.arange(_ny, device=_d, dtype=_t), torch.arange(
            _nx, device=_d, dtype=_t)
        # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
        if check_version(torch.__version__, '1.10.0'):
            _yv, _xv = torch.meshgrid(_y, _x, indexing='ij')
        else:
            _yv, _xv = torch.meshgrid(_y, _x)
        # add grid offset, i.e. y = 2.0 * x - 0.5
        grid = torch.stack((_xv, _yv), 2).expand(shape) - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]
                       ).view((1, self.det_dict.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Model(nn.Module):
    """
    YOLOv5 model
    model, input channels, number of classes
    """
    def __init__(self, cfg='yolov5s.yaml', _ch=3, _nc=None, anchors=None):
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as _f:
                self.yaml = yaml.safe_load(_f)  # model dict

        # Define model
        _ch = self.yaml['ch'] = self.yaml.get('ch', _ch)  # input channels
        if _nc and _nc != self.yaml['nc']:
            LOGGER.info(
                "Overriding model.yaml nc=%s with nc=%s",
                self.yaml['nc'], _nc)
            self.yaml['nc'] = _nc  # override yaml value
        if anchors:
            LOGGER.info(
                "Overriding model.yaml anchors with anchors=%s",
                anchors)
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(
            deepcopy(self.yaml), _ch=[_ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        _m = self.model[-1]  # Detect()
        if isinstance(_m, Detect):
            _s = 256  # 2x min stride
            _m.inplace = self.inplace
            _m.stride = torch.Tensor(
                [_s / x.shape[-2]
                 for x
                 in self.forward(torch.zeros(1, _ch, _s, _s))])  # forward
            # must be in pixel-space (not grid-space)
            _m = check_anchor_order(_m)
            _m.anchors /= _m.stride.view(-1, 1, 1)
            self.stride = _m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        # self.info()
        # LOGGER.info('')

    def forward(self, _x, visualize=False):
        """
        single-scale inference, train
        """
        return self._forward_once(_x, visualize)

    def _forward_once(self, _x, visualize=False):
        """
        _forward_once
        """
        # y, dt = [], []  # outputs
        _y = []
        for _m in self.model:
            if _m.f != -1:  # if not from previous layer
                _x = (_y[_m.f]  # from earlier layers
                      if isinstance(_m.f, int)
                      else [_x if j == -1 else _y[j] for j in _m.f])
            # if profile:
            #     self._profile_one_layer(m, x, dt)
            _x = _m(_x)  # run
            _y.append(_x if _m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(_x, _m.type, _m.i, save_dir=visualize)
        return _x

    def _descale_pred(self, _p, flips, scale, img_size):
        """
        _descale_pred
        """
        # de-scale predictions following augmented inference
        # (inverse operation)
        if self.inplace:
            _p[..., :4] /= scale  # de-scale
            if flips == 2:
                _p[..., 1] = img_size[0] - _p[..., 1]  # de-flip ud
            elif flips == 3:
                _p[..., 0] = img_size[1] - _p[..., 0]  # de-flip lr
        else:
            _x, _y, _wh = _p[..., 0:1] / scale, _p[..., 1:2] / \
                scale, _p[..., 2:4] / scale  # de-scale
            if flips == 2:
                _y = img_size[0] - _y  # de-flip ud
            elif flips == 3:
                _x = img_size[1] - _x  # de-flip lr
            _p = torch.cat((_x, _y, _wh, _p[..., 4:]), -1)
        return _p

    def _clip_augmented(self, _y):
        """
        _clip_autmented
        """
        # Clip YOLOv5 augmented inference tails
        _nl = self.model[-1].nl  # number of detection layers (P3-P5)
        _g = sum(4 ** x for x in range(_nl))  # grid points
        _e = 1  # exclude layer count
        _i = (_y[0].shape[1] // _g) * sum(4 ** x for x in range(_e))  # indices
        _y[0] = _y[0][:, :-_i]  # large
        _i = (_y[-1].shape[1] // _g) * \
            sum(4 ** (_nl - 1 - x)
                for x in range(_e))  # indices
        _y[-1] = _y[-1][:, _i:]  # small
        return _y

    # def _profile_one_layer(self, m, x, dt):
    #     c = isinstance(m, Detect)
    #       # is final layer, copy input as inplace fix
    #     o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[
    #         0] / 1E9 * 2 if thop else 0  # FLOPs
    #     t = time_sync()
    #     for _ in range(10):
    #         m(x.copy() if c else x)
    #     dt.append((time_sync() - t) * 100)
    #     if m == self.model[0]:
    #         LOGGER.info(
    #             f"{'time (ms)':>10s} {'GFLOPs':>10s} \
    #               {'params':>10s}  module")
    #     LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
    #     if c:
    #         LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _initialize_biases(self, _cf=None):
        '''
        initialize biases into Detect(), cf is class frequency
        https://arxiv.org/abs/1708.02002 section 3.3
        '''
        _m = self.model[-1]  # Detect() module
        for _mi, _s in zip(_m.m, _m.stride):  # from
            _b = _mi.bias.view(_m.na, -1)  # conv.bias(255) to (3,85)
            # obj (8 objects per 640 image)
            _b.data[:, 4] += math.log(8 / (640 / _s) ** 2)
            _b.data[:, 5:] += (math.log(0.6 / (_m.nc - 0.999999))
                               if _cf is None
                               else torch.log(_cf / _cf.sum()))  # cls
            _mi.bias = torch.nn.Parameter(_b.view(-1), requires_grad=True)

    # def _print_biases(self):
    #     m = self.model[-1]  # Detect() module
    #     for mi in m.m:  # from
    #         b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
    #         LOGGER.info(
    #             ('%6g Conv2d.bias:' + '%10.3g' * 6) % \
    #             (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))
    #               # shortcut weights

    def fuse(self):
        '''
        fuse model Conv2d() + BatchNorm2d() layers
        '''
        LOGGER.info('Fusing layers... ')
        for _m in self.model.modules():
            if isinstance(_m, (Conv, DWConv)) and hasattr(_m, 'bn'):
                _m.conv = fuse_conv_and_bn(_m.conv, _m.bn)  # update conv
                delattr(_m, 'bn')  # remove batchnorm
                _m.forward = _m.forward_fuse  # update forward
        self.info()
        return self

    def _apply(self, fn):
        '''
        Apply to(), cpu(), cuda(), half() to model tensors
        that are not parameters or registered buffers
        '''
        super()._apply(fn)
        _md = self.model[-1]  # Detect()
        if isinstance(_md, Detect):
            _md.stride = fn(_md.stride)
            _md.grid = list(map(fn, _md.grid))
            if isinstance(_md.anchor_grid, list):
                _md.anchor_grid = list(map(fn, _md.anchor_grid))
        return self


def parse_model(_d, _ch):
    '''
    model_dict, input_channels(3)
    '''
    vdicts = edict()
    anchors, vdicts.nc, vdicts.gd, vdicts.gw = \
        _d['anchors'], _d['nc'], _d['depth_multiple'], _d['width_multiple']
    vdicts.na = ((len(anchors[0]) // 2)
                 if isinstance(anchors, list)
                 else anchors)  # number of anchors
    # number of outputs = anchors * (classes + 5)
    vdicts.no = vdicts.na * (vdicts.nc + 5)

    # layers, savelist, ch out
    layers, vdicts.save, vdicts.c2 = [], [], _ch[-1]
    # from, number, module, args
    for i, (_f, _n, _m, args) in enumerate(_d['backbone'] + _d['head']):
        _m = literal_eval(_m) if isinstance(_m, str) else _m  # eval strings
        for j, _a in enumerate(args):
            try:
                args[j] = (literal_eval(_a)
                           if isinstance(_a, str) else _a)  # eval strings
            except NameError:
                pass

        _n = max(round(_n * vdicts.gd), 1) if _n > 1 else _n  # depth gain
        if _m in (Conv, Bottleneck, DWConv,
                  C3, nn.ConvTranspose2d):
            vdicts.c1, vdicts.c2 = _ch[_f], args[0]
            if vdicts.c2 != vdicts.no:  # if not output
                vdicts.c2 = make_divisible(vdicts.c2 * vdicts.gw, 8)

            args = [vdicts.c1, vdicts.c2, *args[1:]]
            if _m in [C3]:
                args.insert(2, _n)  # number of repeats
                _n = 1
        elif _m is nn.BatchNorm2d:
            args = [_ch[_f]]
        elif _m is Concat:
            vdicts.c2 = sum(_ch[x] for x in _f)
        elif _m is Detect:
            args.append([_ch[x] for x in _f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(_f)
        else:
            vdicts.c2 = _ch[_f]

        _ms = (nn.Sequential(*(_m(*args) for _ in range(_n)))
               if _n > 1 else _m(*args))  # module
        vdicts.t = str(_m)[8:-2].replace('__main__.', '')  # module type
        _np = sum(x.numel() for x in _ms.parameters())  # number params
        # attach index, 'from' index, type, number params
        _ms.i, _ms.f, _ms.type, _ms.np = i, _f, vdicts.t, _np
        vdicts.save.extend(x % i for x in ([_f] if isinstance(
            _f, int) else _f) if x != -1)  # append to savelist
        layers.append(_ms)
        if i == 0:
            _ch = []
        _ch.append(vdicts.c2)
    return nn.Sequential(*layers), sorted(vdicts.save)
