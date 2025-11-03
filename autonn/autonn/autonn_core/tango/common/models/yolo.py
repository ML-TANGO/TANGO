import argparse
import logging
import sys
import contextlib
from copy import deepcopy
from pathlib import Path
import os
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn
from tango.common.models.common import *
from tango.common.models.experimental import *
from tango.common.models.search_block import *
from tango.utils.autoanchor import check_anchor_order
from tango.utils.anchor_generator import make_anchors, dist2bbox
from tango.utils.general import make_divisible, colorstr #, check_file, set_logging
from tango.utils.torch_utils import (   time_synchronized,
                                        fuse_conv_and_bn,
                                        model_info,
                                        model_summary,
                                        scale_img,
                                        initialize_weights,
                                        copy_attr
                                    )
from tango.utils.loss import SigmoidBin

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


logger = logging.getLogger(__name__)


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                    wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z, )
        elif self.concat:
            out = torch.cat(z, 1)
        else:
            out = (torch.cat(z, 1), x)

        return out

    @staticmethod
    def _make_grid(nx=20, ny=20):
        # type: ignore[attr-defined]
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing="ij")
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=torch.float32,
                                           device=z.device)
        box @= convert_matrix                          
        return (box, score)


class DDetect(nn.Module):
    # YOLO Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4), max(
            (ch[0], min((self.nc * 2, 128)))
        )  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3),
                Conv(c2, c2, 3, g=4),
                nn.Conv2d(c2, 4 * self.reg_max, 1, groups=4)
            )
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        ''' Credit:  ultralytics/nn/modules/head.py, line 99-125, in def _inference(), class Detect
            [1] TensorFlow conversion takes {Torch.Tensor}.split() complicated operation 'FlexSplitV'
                solution:
                don't use .split() and seperately compute each of 'box' and 'cls' instead
            [2] TFLite INT8 quantization stability
                dist2bbox() outputs are [0 ~ imgsz],
                and TFLite INT8 quantiation needs to normalize them with imgsz
                it is quite large value to operate with, which may cause values unstable
                solution:
                pre-normalize 'anchor' and 'stride' before using them in dist2bbox()
        '''
        # box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
        # dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in {
            "tf",
            "tf-int8",
        }:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and (self.format == "tf-int8"):
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor(
                [grid_w, grid_h, grid_w, grid_h], device=box.device
            ).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(
                self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2]
            )
        else:
            dbox = (
                self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0))
                * self.strides
            )
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=True, dim=1)

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)


class DualDDetect(nn.Module):
    # YOLO Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch) // 2  # number of detection layers
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4), max((ch[0], min((self.nc * 2, 128))))  # channels
        c4, c5 = make_divisible(max((ch[self.nl] // 4, self.reg_max * 4, 16)), 4), max((ch[self.nl], min((self.nc * 2, 128))))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3, g=4), nn.Conv2d(c2, 4 * self.reg_max, 1, groups=4)) for x in ch[:self.nl])
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch[:self.nl])
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3, g=4), nn.Conv2d(c4, 4 * self.reg_max, 1, groups=4)) for x in ch[self.nl:])
        self.cv5 = nn.ModuleList(
            nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nc, 1)) for x in ch[self.nl:])
        self.dfl = DFL(self.reg_max)
        self.dfl2 = DFL(self.reg_max)

    def forward(self, x):
        shape = x[0].shape  # BCHW
        d1 = []
        d2 = []
        for i in range(self.nl):
            d1.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
            d2.append(torch.cat((self.cv4[i](x[self.nl+i]), self.cv5[i](x[self.nl+i])), 1))
        if self.training:
            return [d1, d2]
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (d1.transpose(0, 1) for d1 in make_anchors(d1, self.stride, 0.5))
            self.shape = shape

        ''' Credit:  ultralytics/nn/modules/head.py, line 99-125, in def _inference(), class Detect
            [1] TensorFlow conversion takes {Torch.Tensor}.split() complicated operation 'FlexSplitV'
            [2] TFLite INT8 quantization stability
                dist2bbox() outputs are [0 ~ imgsz], and TFLite INT8 quantiation needs to normalize them with imgsz
                it is quite large value to operate with, which may cause values unstable
                solution: pre-normalize 'anchor' and 'stride' before using them in dist2bbox()
        '''
        # box, cls = torch.cat([di.view(shape[0], self.no, -1) for di in d1], 2).split((self.reg_max * 4, self.nc), 1)
        # dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        # box2, cls2 = torch.cat([di.view(shape[0], self.no, -1) for di in d2], 2).split((self.reg_max * 4, self.nc), 1)
        # dbox2 = dist2bbox(self.dfl2(box2), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        x_cat = torch.cat([di.view(shape[0], self.no, -1) for di in d1], 2)
        x_cat2 = torch.cat([di.view(shape[0], self.no, -1) for di in d2], 2)
        if self.export and self.format in {
            "tf",
            "tf-int8",
        }:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
            box2 = x_cat2[:, : self.reg_max * 4]
            cls2 = x_cat2[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
            box2, cls2 = x_cat2.split((self.reg_max * 4, self.nc), 1)

        if self.export and (self.format == "tf-int8"):
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor(
                [grid_w, grid_h, grid_w, grid_h], device=box.device
            ).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(
                self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2]
            )
            dbox2 = self.decode_bboxes(
                self.dfl2(box2) * norm, self.anchors.unsqueeze(0) * norm[:, :2]
            )
        else:
            dbox = (
                self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0))
                * self.strides
            )
            dbox2 = (
                self.decode_bboxes(self.dfl2(box2), self.anchors.unsqueeze(0))
                * self.strides
            )
        ''' Credit: lhy0718/yolov9/models/yolo.py, line 411-417, in def forward(), class DualDDetect
            Interesting solution to avoid unnecessary if-condition during inference
            it makes dual heads LIST to single head TENSOR forcely
            like [(1, 84, 8400), (1, 84, 8400)] -> (1, 84, 16800)
            actually, 1st one in the list is auxiliary so should be ignored
            it may be a temporary workaround to adapt TANGO model to Ultralytics model
            but it costs its latency
        '''
        if not self.export:
            y = [torch.cat((dbox, cls.sigmoid()), 1), torch.cat((dbox2, cls2.sigmoid()), 1)]
        else:
            y = torch.cat(
                [
                    torch.cat((dbox, cls.sigmoid()), 1),
                    torch.cat((dbox2, cls2.sigmoid()), 1),
                ],
                2,
            )
        return y if self.export else (y, [d1, d2])
        #y = torch.cat((dbox2, cls2.sigmoid()), 1)
        #return y if self.export else (y, d2)
        #y1 = torch.cat((dbox, cls.sigmoid()), 1)
        #y2 = torch.cat((dbox2, cls2.sigmoid()), 1)
        #return [y1, y2] if self.export else [(y1, d1), (y2, d2)]
        #return [y1, y2] if self.export else [(y1, y2), (d1, d2)]

    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=True, dim=1)

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
        for a, b, s in zip(m.cv4, m.cv5, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)


class TripleDDetect(nn.Module):
    # YOLO Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch) // 3  # number of detection layers
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4), \
                                max((ch[0], min((self.nc * 2, 128))))  # channels
        c4, c5 = make_divisible(max((ch[self.nl] // 4, self.reg_max * 4, 16)), 4), \
                                max((ch[self.nl], min((self.nc * 2, 128))))  # channels
        c6, c7 = make_divisible(max((ch[self.nl * 2] // 4, self.reg_max * 4, 16)), 4), \
                                max((ch[self.nl * 2], min((self.nc * 2, 128))))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3, g=4), 
                          nn.Conv2d(c2, 4 * self.reg_max, 1, groups=4)) for x in ch[:self.nl])
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch[:self.nl])
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3, g=4), 
                          nn.Conv2d(c4, 4 * self.reg_max, 1, groups=4)) for x in ch[self.nl:self.nl*2])
        self.cv5 = nn.ModuleList(
            nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nc, 1)) for x in ch[self.nl:self.nl*2])
        self.cv6 = nn.ModuleList(
            nn.Sequential(Conv(x, c6, 3), Conv(c6, c6, 3, g=4), 
                          nn.Conv2d(c6, 4 * self.reg_max, 1, groups=4)) for x in ch[self.nl*2:self.nl*3])
        self.cv7 = nn.ModuleList(
            nn.Sequential(Conv(x, c7, 3), Conv(c7, c7, 3), nn.Conv2d(c7, self.nc, 1)) for x in ch[self.nl*2:self.nl*3])
        self.dfl = DFL(self.reg_max)
        self.dfl2 = DFL(self.reg_max)
        self.dfl3 = DFL(self.reg_max)

    def forward(self, x):
        shape = x[0].shape  # BCHW
        d1 = []
        d2 = []
        d3 = []
        for i in range(self.nl):
            d1.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
            d2.append(torch.cat((self.cv4[i](x[self.nl+i]), self.cv5[i](x[self.nl+i])), 1))
            d3.append(torch.cat((self.cv6[i](x[self.nl*2+i]), self.cv7[i](x[self.nl*2+i])), 1))
        if self.training:
            return [d1, d2, d3]
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (d1.transpose(0, 1) for d1 in make_anchors(d1, self.stride, 0.5))
            self.shape = shape

        box, cls = torch.cat([di.view(shape[0], self.no, -1) for di in d1], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        box2, cls2 = torch.cat([di.view(shape[0], self.no, -1) for di in d2], 2).split((self.reg_max * 4, self.nc), 1)
        dbox2 = dist2bbox(self.dfl2(box2), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        box3, cls3 = torch.cat([di.view(shape[0], self.no, -1) for di in d3], 2).split((self.reg_max * 4, self.nc), 1)
        dbox3 = dist2bbox(self.dfl3(box3), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        #y = [torch.cat((dbox, cls.sigmoid()), 1), torch.cat((dbox2, cls2.sigmoid()), 1), torch.cat((dbox3, cls3.sigmoid()), 1)]
        #return y if self.export else (y, [d1, d2, d3])
        y = torch.cat((dbox3, cls3.sigmoid()), 1)
        return y if self.export else (y, d3)

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
        for a, b, s in zip(m.cv4, m.cv5, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
        for a, b, s in zip(m.cv6, m.cv7, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)


class IDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # export torchscript or onnx w/o nms
    end2end = False # export onnx w/nms
    include_nms = False
    concat = False

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(IDetect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))  # conv
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference or no export
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def fuseforward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference or no export
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                # if not torch.onnx.is_in_onnx_export():
                #     y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                #     y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                # else:
                xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z, )
        elif self.concat:
            out = torch.cat(z, 1)
        else:
            out = (torch.cat(z, 1), x)

        return out
    
    def fuse(self):
        logger.info("IDetect.fuse: implicit values fused into conv's weight and bias")
        # fuse ImplicitA and Convolution
        for i in range(len(self.m)):
            c1,c2,_,_ = self.m[i].weight.shape
            c1_,c2_, _,_ = self.ia[i].implicit.shape
            self.m[i].bias += torch.matmul(self.m[i].weight.reshape(c1,c2),self.ia[i].implicit.reshape(c2_,c1_)).squeeze(1)

        # fuse ImplicitM and Convolution
        for i in range(len(self.m)):
            c1,c2, _,_ = self.im[i].implicit.shape
            self.m[i].bias *= self.im[i].implicit.reshape(c2)
            self.m[i].weight *= self.im[i].implicit.transpose(0,1)
            
    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing="ij")
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=torch.float32,
                                           device=z.device)
        box @= convert_matrix                          
        return (box, score)


class IAuxDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(IAuxDetect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch[:self.nl])  # output conv
        self.m2 = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch[self.nl:])  # output conv
        
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch[:self.nl])
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch[:self.nl])

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))  # conv
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            x[i+self.nl] = self.m2[i](x[i+self.nl])
            x[i+self.nl] = x[i+self.nl].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                    wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x[:self.nl])

    def fuseforward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].data  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z, )
        elif self.concat:
            out = torch.cat(z, 1)            
        else:
            out = (torch.cat(z, 1), x)

        return out
    
    def fuse(self):
        logger.info("IAuxDetect.fuse")
        # fuse ImplicitA and Convolution
        for i in range(len(self.m)):
            c1,c2,_,_ = self.m[i].weight.shape
            c1_,c2_, _,_ = self.ia[i].implicit.shape
            self.m[i].bias += torch.matmul(self.m[i].weight.reshape(c1,c2),self.ia[i].implicit.reshape(c2_,c1_)).squeeze(1)

        # fuse ImplicitM and Convolution
        for i in range(len(self.m)):
            c1,c2, _,_ = self.im[i].implicit.shape
            self.m[i].bias *= self.im[i].implicit.reshape(c2)
            self.m[i].weight *= self.im[i].implicit.transpose(0,1)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing="ij")
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=torch.float32,
                                           device=z.device)
        box @= convert_matrix                          
        return (box, score)


class Model(nn.Module):
    """ Create YOLO-based Detector NN-Model

    Args:
    -----------
    cfg: str
        path to model configuration yaml file
    ch: int
        number of input channels
    nc: int
        number of classes
    anchors: 2d-list or int
        anchors : np * 3(anchros/floor) * 2(width & height)

    Attributes:
    -----------
    traced: bool
        true if this model is traced by torchscript, onnx
    yaml: dict
        dict loaded from model configuration yaml file
    yaml_file: str
        path to model configuration yaml file
    model: nn.Sequential
        a sequence of nn.Modules
    save: list of int
        a save list of output channels at nn.Modules
    nodes_info: dict
        results parsing model, for viz reference
    names: list of str
        class names (labelmaps)
    stride: int
        yolo-specific grid size
    brief:
        model summary for log visualization

    forward(input, argument, profile):
        nn.Module forward() redefinition
    forward_once(input, profile)
        run this instead forward() if arguement == False
    fuse()
        fuse cv+bn or cv+implicit ops for inference
    nms(mode)
        add NMS if mode == True (and NMS does not exist)
        remove NMS if mode == False (and NMS exists)
    autoshape()
        deprecated method (we don't use it)
    info(verbose, img_size)
        print model information
    summary(img_size, verbose)
        another version of info(): it has return value
    """
    def __init__(self, cfg='basemodel.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        self.traced = False
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
            self.yaml_file = self.yaml.get('name', 'config_dict')
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f'{colorstr("Models: ")}Overriding nc={self.yaml["nc"]} in {self.yaml_file} with nc={nc}')
            self.yaml['nc'] = nc  # override yaml value
        def _normalize_anchors(value):
            if value is None:
                return None
            if isinstance(value, str) and value.lower() == 'none':
                return None
            return value

        anchors = _normalize_anchors(anchors)
        if anchors is not None:
            logger.info(f'{colorstr("Models: ")}Overriding anchors in {self.yaml_file} with anchors={anchors}')
            self.yaml['anchors'] = anchors  # override yaml value

        logger.info(f'\n{colorstr("Models: ")}Creating a model from {self.yaml_file}')
        self.model, self.save, self.nodes_info = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        logger.info(f'{colorstr("Models: ")}Building strides and anchors')
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())
        if isinstance(m, IDetect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once
        if isinstance(m, IAuxDetect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))[:4]])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_aux_biases()  # only run once
        if isinstance(m, DDetect):
            s = 256
            m.inplace = self.inplace
            with torch.no_grad():
                dummy = torch.zeros(1, ch, s, s)
                out = self.forward(dummy)               # DDetect.forward -> (y, x)
                feats = out[1] if isinstance(out, tuple) else out   # x: list of feature maps per scale
            m.stride = torch.tensor([s / f.shape[-2] for f in feats], dtype=torch.float)
            self.stride = m.stride
            m.bias_init()  # only run once

        if isinstance(m, (DualDDetect, TripleDDetect)):
            s = 256
            m.inplace = self.inplace
            with torch.no_grad():
                dummy = torch.zeros(1, ch, s, s)
                out = self.forward(dummy)               # DualDDetect.forward -> (y, [d1, d2])
                feats = out[1][0] if (isinstance(out, tuple) and isinstance(out[1], list)) else out[0]
            m.stride = torch.tensor([s / f.shape[-2] for f in feats], dtype=torch.float)
            self.stride = m.stride
            m.bias_init()  # only run once

        # Init weights, biases
        logger.info(f'{colorstr("Models: ")}Initializing weights')
        initialize_weights(self)
        self.briefs = self.summary()
        # self.info()
        # logger.info('')

    def forward(self, x, augment=False, profile=False):
        if augment:
            return self._forward_augment(x) # augmented inference, None
        return self._forward_once(x, profile)  # single-scale inference, train

    def _forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if not hasattr(self, 'traced'):
                self.traced=False

            if self.traced:
                if isinstance(m, (Detect, IDetect, IAuxDetect, DDetect, DualDDetect, TripleDDetect)):
                    break

            if profile:
                self._profile_one_layer(m, x, dt) # layer, input, time

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self.forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLO augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = nn.Parameter(b.view(-1), requires_grad=True)

    def _initialize_aux_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, mi2, s in zip(m.m, m.m2, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = nn.Parameter(b.view(-1), requires_grad=True)
            b2 = mi2.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b2.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b2.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi2.bias = nn.Parameter(b2.view(-1), requires_grad=True)

    def _profile_one_layer(self, m, x, dt):
        # m = this layer, x = input for this layer, dt = time to forward this layer
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_synchronized()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_synchronized() - t) * 100)
        if m == self.model[0]:
            logger.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        logger.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            logger.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        if not isinstance(m, (Detect, IDetect, IAuxDetect)):
            logger.warning(f"_print_biases: not supproted detect module {type(m)}")
            return self
    
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            logger.info(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, DDetect, DualDDetect, TripleDDetect)):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
            # m.grid = list(map(fn, m.grid))
        return self

    def fuse(self):  # fuse cv+bn or cv+implict ops into one cv op
        logger.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, RepConv):
                # logger.info(f" {m.type}: fuse_repvgg_block")
                m.fuse_repvgg_block() # update conv, remove bn-modules, and update forward
            if isinstance(m, RepConv_OREPA):
                # logger.info(f" {m.type}: switch_to_deploy")
                m.switch_to_deploy()  # update conv, remove bn-modules, and update forward
            if isinstance(m, RepConvN) and hasattr(m, 'fuse_convs'):
                # logger.info(f" {m.type}: fuse_convs")
                m.fuse_convs()
                m.forward = m.forward_fuse
            if isinstance(m, (DyConv, DyConvBlock, TinyDyConv)) and hasattr(m, 'bn'):
                # logger.info(f" {m.type}: fuse_conv_and_bn")
                m.conv = m.fuse_conv_and_bn()
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward
            if type(m) is Conv and hasattr(m, 'bn'):
                # logger.info(f" {m.type}: fuse_conv_and_bn")
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
            if isinstance(m, (IDetect, IAuxDetect)):
                # logger.info(f" {m.type}: fuse")
                m.fuse() # update conv
                delattr(m, 'im') # remove implicitM
                delattr(m, 'ia') # remove implicitA
                m.forward = m.fuseforward # update forward
        # self.info(verbose=True)
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def summary(self, img_size=640, verbose=False):
        return model_summary(self, img_size, verbose)


class BaseModel(nn.Module):
    # YOLO base model
    def forward(self, x, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        # logger.info('_forward_once')
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_synchronized()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_synchronized() - t) * 100)
        if m == self.model[0]:
            logger.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        logger.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            logger.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        logger.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, RepConvN) and hasattr(m, 'fuse_convs'):
                m.fuse_convs()
                m.forward = m.forward_fuse  # update forward
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward # m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, DDetect, DualDDetect, TripleDDetect)):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
            # m.grid = list(map(fn, m.grid))
        return self


class DetectionModel(BaseModel):
    # YOLO detection model
    def __init__(self, cfg='yolo.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        # self.model, self.save, self.nodes_info = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.model, self.save = parse_model_v9(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, DDetect)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            # check_anchor_order(m)
            # m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            m.bias_init()  # only run once
        elif isinstance(m, (DualDDetect, TripleDDetect)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0]
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            # check_anchor_order(m)
            # m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.tensor([8., 16., 32.])
        # Init weights, biases
        initialize_weights(self)
        self.briefs = self.summary()
        # self.info()
        # logger.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        # logger.info('_forward_augment')
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLO augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def summary(self, img_size=640, verbose=False):
        return model_summary(self, img_size, verbose)


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('%3s%28s%3s%10s  %-20s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    logger.info('='*100)
    nodes_info = {}
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        node = {}
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass
        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [nn.Conv2d, Conv, RobustConv, RobustConv2, DWConv, GhostConv, RepConv, RepConv_OREPA, DownC, 
                 SPP, SPPF, SPPCSPC, GhostSPPCSPC, MixConv2d, Focus, Stem, GhostStem, CrossConv, 
                 Bottleneck, BottleneckCSPA, BottleneckCSPB, BottleneckCSPC, 
                 RepBottleneck, RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC,  
                 Res, ResCSPA, ResCSPB, ResCSPC, 
                 RepRes, RepResCSPA, RepResCSPB, RepResCSPC, 
                 ResX, ResXCSPA, ResXCSPB, ResXCSPC, 
                 RepResX, RepResXCSPA, RepResXCSPB, RepResXCSPC, 
                 Ghost, GhostCSPA, GhostCSPB, GhostCSPC,
                 SwinTransformerBlock, STCSPA, STCSPB, STCSPC,
                 SwinTransformer2Block, ST2CSPA, ST2CSPB, ST2CSPC,
                 DyConv, AConv, DWConv, ADown, ELAN1, RepNCSPELAN4, SPPELAN]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [DownC, SPPCSPC, GhostSPPCSPC, 
                     BottleneckCSPA, BottleneckCSPB, BottleneckCSPC, 
                     RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC, 
                     ResCSPA, ResCSPB, ResCSPC, 
                     RepResCSPA, RepResCSPB, RepResCSPC, 
                     ResXCSPA, ResXCSPB, ResXCSPC, 
                     RepResXCSPA, RepResXCSPB, RepResXCSPC,
                     GhostCSPA, GhostCSPB, GhostCSPC,
                     STCSPA, STCSPB, STCSPC,
                     ST2CSPA, ST2CSPB, ST2CSPC]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is BBoneELAN:
            c1, c2 = ch[f], int(args[0]*(args[-1]+1))
            args = [c1, *args]
        elif m is HeadELAN:
            c1, c2 = ch[f], int((args[0]*2) + (args[0]/2 * (args[-1]-1)))
            args = [c1, *args]
        elif m is TinyELAN: # TinyELAN
            c1, c2 = ch[f], args[0]
            args = [c1, c2, *args[1:]]
        elif m is TinyDyConv:
            c1, c2 = args[0], int(args[0]//2)
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Chuncat:
            c2 = sum([ch[x] for x in f])
        elif m is Shortcut:
            c2 = ch[f[0]]
        elif m is Foldcut:
            c2 = ch[f] // 2
        elif m in [Detect, IDetect, IAuxDetect, DDetect, DualDDetect, TripleDDetect]:
            args.append([ch[x] for x in f])
            if m in [Detect, IDetect, IAuxDetect]:
                if isinstance(args[1], int):  # number of anchors
                    args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is ReOrg:
            c2 = ch[f] * 4
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        elif m is CBFuse:
            c2 = ch[f[-1]]
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '').split('.')[-1]  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%28s%3s%10.0f  %-20s%-30s' % (i, f, n, np, t, args))  # print
        node['from'] = f
        node['repeat'] = n
        node['params'] = np
        node['module'] = t
        node['arguments'] = str(args)
        nodes_info[f'{i}'.zfill(2)] = node
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    logger.info('='*100)
    return nn.Sequential(*layers), sorted(save), nodes_info


def parse_model_v9(d, ch):  # model_dict, input_channels(3)
    # Parse a YOLO model.yaml dictionary
    logger.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        RepConvN.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        logger.info(f"{colorstr('activation:')} {act}")  # print
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out

    # ==========================================================================
    # Shape tracking initialization
    # ==========================================================================
    imgsz = d.get('imgsz', 640)
    input_shape = (ch[0], imgsz, imgsz) # (C, H, W)
    shapes = [] # output shapes

    def _shape_at(idx):
        if idx == -1:
            return shapes[-1] if shapes else input_shape
        return shapes[idx]

    # ==========================================================================
    # Layer parsing loop
    # ==========================================================================
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain

        # ----------------------------------------------------------------------
        # Input shape
        # ----------------------------------------------------------------------
        in_shapes = [_shape_at(x) for x in (f if isinstance(f, list) else [f])] # [(C,H,W), ...]

        # ----------------------------------------------------------------------
        # Channel propagation (determine c2 and ajust agrs)
        # ----------------------------------------------------------------------
        if m in {
            Conv, AConv, 
            Bottleneck, SPP, SPPF, DWConv, nn.ConvTranspose2d, SPPCSPC, ADown,
            ELAN1, RepNCSPELAN4, SPPELAN}:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in {SPPCSPC}:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
            c2 = ch[f]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Shortcut:
            c2 = ch[f[0]]
        elif m is ReOrg:
            c2 = ch[f] * 4
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        # TODO: channel, gw, gd
        elif m in {Detect, DDetect, DualDDetect, TripleDDetect}:
            args.append([ch[x] for x in f])
            # if isinstance(args[1], int):  # number of anchors
            #     args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        elif m is nn.Flatten:
            # Normalize args to PyTorch signature: (start_dim=1, end_dim=-1)
            # Many YAMLs mistakenly put tuples/dicts here; ignore them.
            if len(args) == 0:
                args = []  # use defaults
            elif len(args) == 1:
                logger.info('1'*100)
                # If someone put a tuple or dict, ignore and use defaults
                if isinstance(args[0], (tuple, list, dict)) or args[0] is None:
                    logger.warning("Ignoring invalid Flatten arg; using defaults (start_dim=1, end_dim=-1)")
                    args = []
                else:
                    # keep single int as start_dim
                    args = [int(args[0])]
            else:
                logger.info('2'*100)
                # keep first two as ints (start_dim, end_dim), drop the rest
                args = [int(args[0]), int(args[1])]

            c2 = ch[f]  # channels unchanged
        elif m is nn.Linear:
            if len(in_shapes) != 1:
                logger.warning("Linear expects a single input tensor")
            C, H, W = in_shapes[0]
            in_features = C * H * W
            out_features = int(args[0])
            args = [in_features, out_features, *args[1:]]
            c2 = out_features
        else:
            c2 = ch[f]

        # ----------------------------------------------------------------------
        # Module creation
        # ----------------------------------------------------------------------
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)

        # ----------------------------------------------------------------------
        # Shape propagation
        # ----------------------------------------------------------------------
        multi_input = len(in_shapes) > 1
        if multi_input:
            cur_shape = propagate_shape(m, args, in_shapes)
        else:
            cur_shape = in_shapes[0]
            n_eff = n if isinstance(n, int) else 1
            for _ in range(n_eff):
                cur_shape = propagate_shape(m, args, [cur_shape])
        shapes.append(cur_shape)

        # ----------------------------------------------------------------------
        # Channel update
        # ----------------------------------------------------------------------
        if i == 0:
            ch = []
        ch.append(c2)

    return nn.Sequential(*layers), sorted(save)


def propagate_shape(m, args, in_shapes):
    """
    Shape inference for a single application of module class `m` with constructor args `args`.
    Works before module instantiation. Returns (C_out, H_out, W_out) when determinable,
    otherwise returns a conservative pass-through of the first input.
    Assumptions:
      - For Conv-like layers, args are normalized to [inC, outC, k, s, p, d, ...]
      - For Linear, args are normalized to [in_features, out_features, ...]
      - For Flatten, no args needed.
    """

    # ---- helpers ----
    def _conv_out_len(l_in, k, s, p, d=1):
        return ((l_in + 2*p - d*(k-1) - 1) // s + 1)

    def _as_hw_tuple(x, default):
        # normalize int/tuple to (h,w)
        if isinstance(x, (tuple, list)):
            return int(x[0]), int(x[1])
        return int(x), int(x)

    def _same_hw(hws):
        h0, w0 = hws[0]
        for (h, w) in hws[1:]:
            if h != h0 or w != w0:
                raise ValueError(f"Multi-input H,W mismatch: {hws}")
        return h0, w0
    
    def _ceil_div(a, b): 
        return (a + b - 1) // b

    single = (len(in_shapes) == 1)
    first = in_shapes[0]

    # ---- multi-input ops ----
    if m is Concat:
        # H,W must match; C sums
        H, W = _same_hw([(h, w) for (_, h, w) in in_shapes])
        C = sum(C for (C, _, _) in in_shapes)
        return (C, H, W)

    if m is Shortcut:
        # elementwise add: all (C,H,W) identical
        C0, H0, W0 = in_shapes[0]
        for (C, H, W) in in_shapes[1:]:
            if (C, H, W) != (C0, H0, W0):
                raise ValueError(f"Shortcut shape mismatch: {in_shapes}")
        return (C0, H0, W0)

    # ---- single-input ops below ----
    if not single:
        # Unknown multi-input type → conservative: pass-through first
        return first

    C_in, H_in, W_in = first

    # ReOrg
    if m is ReOrg:
        return (C_in * 4, H_in // 2, W_in // 2)

    # Contract / Expand
    if m is Contract:
        s = args[0]
        return (C_in * (s**2), H_in // s, W_in // s)

    if m is Expand:
        s = args[0]
        return (C_in // (s**2), H_in * s, W_in * s)

    # Flatten
    if m is torch.nn.Flatten:
        return (C_in * H_in * W_in, 1, 1)

    # Linear (args already normalized)
    if m is torch.nn.Linear:
        out_features = args[1] if len(args) > 1 else args[0]
        return (int(out_features), 1, 1)

    # BatchNorm2d
    if m is torch.nn.BatchNorm2d:
        return (C_in, H_in, W_in)

    # MaxPool2d / AvgPool2d  (args: k, s=None, p=0, ...)
    if m in (torch.nn.MaxPool2d, torch.nn.AvgPool2d):
        k = args[0] if len(args) > 0 else 2
        s = args[1] if len(args) > 1 and args[1] is not None else k
        p = args[2] if len(args) > 2 else 0
        kh, kw = _as_hw_tuple(k, 2)
        sh, sw = _as_hw_tuple(s, kh)
        ph, pw = _as_hw_tuple(p, 0)
        H_out = _conv_out_len(H_in, kh, sh, ph, 1)
        W_out = _conv_out_len(W_in, kw, sw, pw, 1)
        return (C_in, H_out, W_out)

    # AdaptiveAvgPool2d (args[0] can be int or (H_out, W_out))
    if m is torch.nn.AdaptiveAvgPool2d:
        out_sz = args[0] if len(args) > 0 else 1
        if isinstance(out_sz, int):
            return (C_in, int(out_sz), int(out_sz))
        return (C_in, int(out_sz[0]), int(out_sz[1]))

    # Upsample (constructor may pass size= or scale_factor=. Here we read from args if present.)
    if m is torch.nn.Upsample:
        # best-effort: accept either kw-style dict or positional list
        # common: kwargs dict as last arg or only arg
        size = None
        scale_factor = None
        # try kwargs dict at the end
        if args and isinstance(args[-1], dict):
            size = args[-1].get('size')
            scale_factor = args[-1].get('scale_factor')
        # or positional (rare in yaml)
        if size is None and len(args) >= 1 and isinstance(args[0], (tuple, list, int)):
            # heuristics: treat first positional as size if provided
            size = args[0] if not isinstance(args[0], (float,)) else None
        if scale_factor is None and len(args) >= 1 and isinstance(args[0], (float, tuple, list)):
            scale_factor = args[0] if isinstance(args[0], (float, tuple, list)) else None

        if size is not None:
            if isinstance(size, (tuple, list)):
                return (C_in, int(size[0]), int(size[1]))
            return (C_in, int(size), int(size))
        if scale_factor is None:
            scale_factor = 2
        if isinstance(scale_factor, (tuple, list)):
            sh, sw = float(scale_factor[0]), float(scale_factor[1])
        else:
            sh = sw = float(scale_factor)
        return (C_in, int(round(H_in * sh)), int(round(W_in * sw)))

    # ConvTranspose2d (args: inC, outC, k, s, p, out_pad, groups, bias, d)
    if m is torch.nn.ConvTranspose2d:
        k = args[2] if len(args) > 2 else 2
        s = args[3] if len(args) > 3 else 2
        p = args[4] if len(args) > 4 else 0
        op = args[5] if len(args) > 5 else 0
        d = args[8] if len(args) > 8 else 1
        kh, kw = _as_hw_tuple(k, 2)
        sh, sw = _as_hw_tuple(s, 2)
        ph, pw = _as_hw_tuple(p, 0)
        oph, opw = _as_hw_tuple(op, 0)
        # (in - 1)*s - 2p + d*(k-1) + out_pad + 1
        H_out = (H_in - 1) * sh - 2 * ph + d * (kh - 1) + oph + 1
        W_out = (W_in - 1) * sw - 2 * pw + d * (kw - 1) + opw + 1
        # C_out은 outC(= args[1])이지만, 여기서는 외부에서 채널 관리 가능
        C_out = int(args[1]) if len(args) > 1 else C_in
        return (C_out, H_out, W_out)

    # Conv-like / composite blocks (treat as single conv step):
    # Expect args like [inC, outC, k, s, p, d, ...] after your channel-prop normalization.
    if m in (nn.Conv2d, Conv, DWConv):
        C_out = int(args[1]) if len(args) > 1 else C_in
        k = args[2] if len(args) > 2 else 1
        s = args[3] if len(args) > 3 else 1
        p = args[4] if len(args) > 4 else (k // 2 if isinstance(k, int) else 0)
        d = args[5] if len(args) > 5 else 1
        kh, kw = _as_hw_tuple(k, 1)
        sh, sw = _as_hw_tuple(s, 1)
        ph, pw = _as_hw_tuple(p, kh // 2)
        dh, dw = _as_hw_tuple(d, 1)
        H_out = _conv_out_len(H_in, kh, sh, ph, dh)
        W_out = _conv_out_len(W_in, kw, sw, pw, dw)
        return (C_out, H_out, W_out)

    # Spatially preserved blocks
    if m in (RepNCSPELAN4, ELAN1, SPPELAN, Bottleneck, SPP, SPPF, SPPCSPC):
        C_out = int(args[1]) if len(args) > 1 else C_in
        if m in (RepNCSPELAN4, ELAN1) and len(args) > 2 and (int(args[2]) % 2):
            logger.warning(f"{m.__name__}: c3 must be divisible by 2 for chunk(2,1)")
        return (C_out, H_in, W_in)
    
    # AConv / ADown: avg_pool2d(k=2,s=1) → H-1,W-1 → stride-2
    if m in (AConv, ADown):
        C_out = int(args[1]) if len(args) > 1 else C_in
        H_mid = H_in - 1  # avg_pool2d(k=2, s=1)
        W_mid = W_in - 1
        H_out = _ceil_div(H_mid, 2)
        W_out = _ceil_div(W_mid, 2)
        return (C_out, H_out, W_out)
    
    # CBLinear: conv → split; track as sum of splits
    if m is CBLinear:
        c2s = args[1] if len(args) > 1 else []
        if not isinstance(c2s, (list, tuple)) or len(c2s) == 0:
            logger.warning("CBLinear: c2s must be a non-empty list/tuple.")
            return (C_in, H_in, W_in)
        try:
            out_ch = int(sum(int(x) for x in c2s))
        except Exception:
            logger.warning(f"CBLinear: invalid c2s={c2s!r}, falling back to C_in")
            out_ch = C_in
        return (out_ch, H_in, W_in)

    if m is CBFuse:
        return in_shapes[-1]

    # Detect-family heads: keep passthrough for backbone feature tracking
    if m in (Detect, DDetect, DualDDetect, TripleDDetect):
        return (C_in, H_in, W_in)

    # Fallback: unknown layer → pass-through (safe default)
    return (C_in, H_in, W_in)
