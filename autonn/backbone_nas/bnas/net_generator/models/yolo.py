# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules
Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import math
import contextlib
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, make_divisible
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn,  # , time_sync
                                 initialize_weights)

from .common import C3, Bottleneck, Concat, Conv, DWConv, SPPF

# try:
#     import thop  # for FLOPs computation
# except ImportError:
#     thop = None


class Detect(nn.Module):
    """
    head of yolo
    """
    stride = []  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer('anchors', torch.Tensor(
            anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1)
                                for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Model(nn.Module):
    """
    YOLOv5 model
    model, input channels, number of classes
    """
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as _f:
                self.yaml = yaml.safe_load(_f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(
                "Overriding model.yaml nc=%s with nc=%s",
                self.yaml['nc'], nc)
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(
                "Overriding model.yaml anchors with anchors=%s",
                anchors)
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(
            deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.Tensor(
                [s / x.shape[-2]
                 for x
                 in self.forward(torch.zeros(1, ch, s, s))])  # forward
            # must be in pixel-space (not grid-space)
            m = check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        # self.info()
        # LOGGER.info('')

    def forward(self, x, visualize=False):
        """
        single-scale inference, train
        """
        return self._forward_once(x, visualize)

    def _forward_once(self, x, visualize=False):
        """
        _forward_once
        """
        y = []
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = (y[m.f]  # from earlier layers
                      if isinstance(m.f, int)
                      else [x if j == -1 else y[j] for j in m.f])
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _descale_pred(self, p, flips, scale, img_size):
        """
        _descale_pred
        """
        # de-scale predictions following augmented inference
        # (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] /\
                scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        """
        _clip_autmented
        """
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):
        '''
        initialize biases into Detect(), cf is class frequency
        https://arxiv.org/abs/1708.02002 section 3.3
        '''
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            # obj (8 objects per 640 image)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += (math.log(0.6 / (m.nc - 0.999999))
                               if cf is None
                               else torch.log(cf / cf.sum()))  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def fuse(self):
        '''
        fuse model Conv2d() + BatchNorm2d() layers
        '''
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def _apply(self, fn):
        '''
        Apply to(), cpu(), cuda(), half() to model tensors
        that are not parameters or registered buffers
        '''
        super()._apply(fn)
        md = self.model[-1]  # Detect()
        if isinstance(md, Detect):
            md.stride = fn(md.stride)
            md.grid = list(map(fn, md.grid))
            if isinstance(md.anchor_grid, list):
                md.anchor_grid = list(map(fn, md.anchor_grid))
        return self


def parse_model(d, ch):
    '''
    model_dict, input_channels(3)
    '''
    anchors, nc, gd, gw = \
        d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = ((len(anchors[0]) // 2)
          if isinstance(anchors, list)
          else anchors)  # number of anchors
    # number of outputs = anchors * (classes + 5)
    no = na * (nc + 5)

    # layers, savelist, ch out
    layers, save, c2 = [], [], ch[-1]
    # from, number, module, args
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in (Conv, Bottleneck, DWConv,
                  C3, nn.ConvTranspose2d, SPPF):
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [C3]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]

        ms = (nn.Sequential(*(m(*args) for _ in range(n)))
               if n > 1 else m(*args))  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in ms.parameters())  # number params
        # attach index, 'from' index, type, number params
        ms.i, ms.f, ms.type, ms.np = i, f, t, np
        save.extend(x % i for x in ([f] if isinstance(
            f, int) else f) if x != -1)  # append to savelist
        layers.append(ms)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)
