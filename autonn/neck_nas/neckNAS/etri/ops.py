from yolov5_utils.general import (LOGGER, check_version, non_max_suppression)
from typing import List
import torch
import torch.nn as nn
# import nni.retiarii.nn.pytorch as nn

import math


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else (x // 2 for x in k)
    return p


def build_activation(act_func, inplace=True):
    if act_func == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_func == 'relu6':
        return nn.ReLU6(inplace=inplace)
    elif act_func == 'leakyRelu':
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    elif act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'sigmod':
        return nn.Sigmoid()
    elif act_func == 'silu':
        return nn.SiLU(inplace=inplace)
    elif act_func == 'mish':
        return nn.Mish(inplace=inplace)
    elif act_func is None:
        return nn.Identity()
    else:
        raise ValueError('do not support: %s' % act_func)


class Conv(nn.Module):
    default_act = nn.SiLU()  # clever trick for reducing footprint

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g,
                              bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True \
            else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Focus(nn.Module):
    # focus width and height information into channel space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        return self.conv(torch.cat([x[..., ::2, ::2],
                                   x[..., 1::2, ::2],
                                   x[..., ::2, 1::2],
                                   x[..., 1::2, 1::2]], 1))


class CBR(Conv):
    def __init__(self, c1, c2, k=1, s=1, act=nn.ReLU()):
        super().__init__(c1, c2, k, s, act=act)


class CBR6(Conv):
    def __init__(self, c1, c2, k=1, s=1, act=nn.ReLU6()):
        super().__init__(c1, c2, k, s, act=act)


class DBR6(Conv):
    # Depth-wise convolution in MobileNet v.2
    def __init__(self, c1, c2, k=1, s=1, g=1, act=nn.ReLU6()):
        super().__init__(c1, c2, k=k, s=s,
                         g=g if g else math.gcd(c1, c2), act=act)


class CBL(Conv):
    def __init__(self, c1, c2, k=1, s=1, act=nn.LeakyReLU(negative_slope=0.1)):
        super().__init__(c1, c2, k, s, act=act)


class CBS(Conv):
    def __init__(self, c1, c2, k=1, s=1, act=True):
        super().__init__(c1, c2, k, s, act=act)


class CBM(Conv):
    def __init__(self, c1, c2, k=1, s=1, act=nn.Mish()):
        super().__init__(c1, c2, k, s, act=act)


class Bottleneck(nn.Module):
    # Bottleneck used in YOLO v.3 ~ v.5
    def __init__(self, c1, c2, shorcut=True, e=0.5):
        super().__init__()
        ch_middle = int(c2 * e)
        self.conv1 = CBS(c1, ch_middle, 1, 1)
        self.conv2 = CBS(ch_middle, c2, 3, 1)
        # self.conv1 = Conv(c1, ch_middle, 1, 1, act='silu')
        # self.conv2 = Conv(ch_middle, c2, 3, 1, act='silu')
        self.add = shorcut and c1 == c2

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) \
            if self.add else self.conv2(self.conv1(x))


# https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
class MBInvertedResidual(nn.Module):
    def __init__(self, c1, c2, s=1, e=6):
        super().__init__()

        self.stride = s
        if s not in [1, 2]:
            raise ValueError(
                f"inverted residual block"
                f" can not have stride of {self.stride}")

        ch_middle = int(round(c1 * e))
        self.use_shortcut = self.stride == 1 and c1 == c2

        # self.pw_conv = CBR6(c1, ch_middle, k=1, s=1)
        # self.dw_conv = DBR6(ch_middle, ch_middle, k=3, s=s)
        # self.pw_conv_linear =  Conv(ch_middle, c2, k=1, s=1, act=None)

        layers: List[nn.Module] = []
        if e != 1:
            layers.append(CBR6(c1, ch_middle, k=1, s=1))  # point-wise
        layers.append(DBR6(ch_middle, ch_middle, k=3, s=s, g=ch_middle))
        layers.append(Conv(ch_middle, c2, k=1, s=1, act=None))

        self.inverted_residual = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.inverted_residual(x)
        else:
            return self.inverted_residual(x)


class MB(nn.Module):
    # Bottleneck block used in MobileNet v.2
    # https://arxiv.org/pdf/1812.00332.pdf
    def __init__(self, ch_in, t, c, n, s):
        super().__init__()
        ch_out = c

        layers: List[nn.Module] = []
        for i in range(n):
            stride = s if i == 0 else 1
            expand_ratio = t
            block = MBInvertedResidual(ch_in, ch_out, stride, expand_ratio)
            layers.append(block)
            ch_in = ch_out

        self.mobile_bottleneck = nn.Sequential(*layers)

    def forward(self, x):
        return self.mobile_bottleneck(x)


class CSP(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super().__init__()
        ch_middle = int(round(c2 * e))
        self.conv1 = CBS(c1, ch_middle, 1, 1)
        self.conv2 = CBS(c1, ch_middle, 1, 1)
        self.conv3 = CBS(2 * ch_middle, c2, 1)
        # self.conv1 = Conv(c1, ch_middle, 1, 1, act='silu')
        # self.conv2 = Conv(c1, ch_middle, 1, 1, act='silu')
        # self.conv3 = Conv(2 * ch_middle, c2, 1, act='silu')
        self.net = nn.Sequential(
            *(Bottleneck(ch_middle,
                         ch_middle,
                         shortcut,
                         e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.conv3(
            torch.cat((self.net(self.conv1(x)), self.conv2(x)), dim=1))


class SPP(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    # equivalent to original SPP(k=(5, 9, 13))
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = CBS(c1, c_, 1, 1)
        self.cv2 = CBS(c_ * 4, c2, 1, 1)
        # self.cv1 = Conv(c1, c_, 1, 1, act='silu')
        # self.cv2 = Conv(c_ * 4, c2, 1, 1, act='silu')
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class ConcatForNas(nn.Module):
    # Sig = torch.nn.Sigmoid()

    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        self.forward_type = self.forward_search

    def forward(self, input_list):
        LOGGER.debug('\n concat: input list')
        for i, input in enumerate(input_list):
            LOGGER.debug(f'[{i}] {input.shape}')
        return self.forward_type(input_list)

    def forward_search(self, input_list):
        out_list = []
        # a_sig = torch.sigmoid(self.arch_weight)
        # a_sig = self.Sig(self.arch_weight)
        if self.training:
            # out_list.append(self.ops[0](input_list[0]))
            # sample = torch.bernoulli(a_sig)
            # self.active_path = sample.tolist()
            # for a, i, o, s in zip(a_sig, input_list[1:], self.ops[1:],
            #                       sample):
            # for a, i, o in zip(a_sig, input_list[1:], self.ops[1:]):
            for i, o in zip(input_list, self.ops):
                # a_detach = a.detach()
                # if s > 0.5:     # c = 1
                #     b = 1. - a_detach
                #     c = a + b
                # else:           # c = 0
                #     b = -1. * a_detach
                #     c = a + b
                # out_list.append(c*o(i))
                # print(f'{i.shape} -> {a} * hidden -> {o(i).shape}')
                out_list.append(o(i))
            # print('-------------------------------\n')
            LOGGER.debug(' concat: manipulated input list')
            for i, output in enumerate(out_list):
                LOGGER.debug(f'[{i}] {output.shape}')
            return torch.cat(out_list, self.d)
        else:
            # out_list.append(self.ops[0](input_list[0]))
            # for a, i, o in zip(a_sig, input_list[1:], self.ops[1:]):
            for i, o in zip(input_list, self.ops):
                # (tenace comment) is it inteneded? it might be something wrong
                # why does it mulplicate ops by sigmoid of arch_weight?
                # we need output list of o(i)
                #   where i is the path that bernoulli gate is on(1)
                #   at train phase
                # if a > 0.5:     # c = 1
                #     c = 1
                # else:           # c = 0
                #     c = 0
                # out_list.append(c*o(i))
                out_list.append(o(i))
            return torch.cat(out_list, self.d)

    def forward_retrain(self, input_list):
        out_list = []
        for i, o in zip(input_list, self.ops):
            # TODO:
            # we need the policy picking up the best architecture
            # for examaple, given the archeture parameters, choose the best one
            # or choose ones that are over the threshold...
            out_list.append(o(i))
        return torch.cat(out_list, self.d)

    def arch_param_define(self,
                          prev_sz, prev_ch,
                          f_sz, f_ch,
                          refer_sz=None, refer_ch=None,  # device,
                          path_freezing=False,
                          neck_channel=[80, 160, 320],
                          ch_res_rule=0, fidx=None):
        refer_ch = sorted(refer_ch)
        neck_channel = sorted(neck_channel)
        # Search Stage OR Retrain Stage
        if path_freezing:
            self.forward_type = self.forward_retrain
        else:
            # print('Architecture Parameter(Neck Path Weight) Initialized')
            # self.arch_weight = torch.nn.Parameter(torch.zeros(len(f_ch)-1),
            #                                       requires_grad=True)
            pass
            # self.arch_weight.to(device)  # (tenace comment) bug!

        self.ops = nn.ModuleList([])
        if ch_res_rule == 0:  # from 'Backbone'
            print(f'==reference size and channels==')
            for fl, (s, c) in enumerate(zip(refer_sz, refer_ch)):
                print(f'P{fl}= s{s} c{c}', end='  ')
            print(f'\n(expect P{refer_sz.index(prev_sz)}'
                  f' s{prev_sz} c{prev_ch})')
            for ii, (sz, ch) in enumerate(zip(f_sz, f_ch)):
                print(f'--input #{ii}: P{refer_sz.index(sz)} {sz} {ch}')
                if fidx[ii] == -1:
                    self.ops.append(torch.nn.Identity())
                    continue
                if refer_sz.index(sz) > refer_sz.index(prev_sz):
                    temp_ops = []
                    in_channels = ch
                    for n in range(refer_sz.index(sz)
                                   - refer_sz.index(prev_sz)):
                        print("upsampling...")
                        # TODO: Channel- should be performed
                        # up to its backbone, like CBS, CBL, CBM, CBR, CBR6...
                        temp_ops.append(
                            CBS(in_channels,  # CBR6(in_channels,
                                # neck_channel[refer_ch.index(ch)-(n+1)],
                                refer_ch[refer_sz.index(sz)-(n+1)],
                                k=1, s=1))
                        temp_ops.append(torch.nn.Upsample(None, 2, 'nearest'))
                        in_channels = refer_ch[refer_sz.index(sz)-(n+1)]

                    self.ops.append(torch.nn.Sequential(*temp_ops))
                elif refer_sz.index(sz) < refer_sz.index(prev_sz):
                    temp_ops = []
                    in_channels = ch
                    for n in range(refer_sz.index(prev_sz)
                                   - refer_sz.index(sz)):
                        # TODO: Downsampling & Channel+ should be performed
                        # up to its backbone, like CBS, CBL, CBM, CBR, CBR6...
                        temp_ops.append(
                            CBS(in_channels,  # CBR6(in_channels,
                                #  neck_channel[refer_ch.index(ch)-(n+1)],
                                refer_ch[refer_sz.index(sz)+(n+1)],
                                k=3, s=2))
                        in_channels = refer_ch[refer_sz.index(sz)+(n+1)]
                        print(f'downsampling.... channel = {in_channels}')
                    self.ops.append(torch.nn.Sequential(*temp_ops))
                else:  # refer_ch.index(ch) == neck_channel.index(prev_ch)
                    if ch == prev_ch:
                        self.ops.append(torch.nn.Identity())
                    else:  # (tenace comment: it is not going to happen...)
                        self.ops.append(
                            CBS(ch,  # CBR6(ch,
                                # neck_channel[refer_ch.index(ch)], k=3, s=1))
                                refer_ch[refer_sz.index(sz)], k=3, s=1))
        elif ch_res_rule == 1:  # from 'PreLayer'
            print(f'==reference size and channels==')
            refer_ch.insert(0, refer_ch[0])
            for fl, (s, c) in enumerate(zip(refer_sz, refer_ch)):
                print(f'P{fl}= s{s} c{c}', end='  ')
            print(f'\n(expect P{refer_sz.index(prev_sz)}'
                  f' s{prev_sz} c{prev_ch})')
            for ii, (sz, ch) in enumerate(zip(f_sz, f_ch)):
                print(f'--input #{ii}: P{refer_sz.index(sz)} {sz} {ch}')
                if fidx[ii] == -1:
                    self.ops.append(torch.nn.Identity())
                    continue

                if refer_sz.index(sz) > refer_sz.index(prev_sz):
                    temp_ops = []
                    in_channels = ch
                    for n in range(refer_sz.index(sz)
                                   - refer_sz.index(prev_sz)):
                        # (tenace comment) Upsampling must be
                        #                  with channel reducing operation
                        #                  Attach an additional CBR6
                        temp_ops.append(
                            CBS(in_channels,
                                #  neck_channel[refer_ch.index(ch)-(n+1)],
                                refer_ch[refer_sz.index(sz)-(n+1)],
                                k=1, s=1))
                        temp_ops.append(torch.nn.Upsample(None, 2, 'nearest'))
                        in_channels = neck_channel[refer_ch.index(ch)-(n+1)]
                    self.ops.append(torch.nn.Sequential(*temp_ops))

                elif refer_sz.index(sz) < refer_sz.index(prev_sz):
                    temp_ops = []
                    in_channels = ch
                    for n in range(refer_sz.index(prev_sz)
                                   - refer_sz.index(sz)):
                        # (tenace comment) Downsampling must be
                        #                  with channel increasing operation
                        # temp_ops.append(CBR6(ch, ch, k=3, s=2))
                        temp_ops.append(
                            CBS(in_channels,
                                #  neck_channel[refer_ch.index(ch)+(n+1)],
                                refer_ch[refer_sz.index(sz)+(n+1)],
                                k=3, s=2))
                    self.ops.append(torch.nn.Sequential(*temp_ops))

                else:
                    self.ops.append(torch.nn.Identity())
        else:
            raise NotImplementedError('Check ch_res_rule')

    def get_arch_weight(self):
        return self.arch_weight

    def get_active_path(self):
        return self.active_path


class ConcatSPP(nn.Module):
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_hidden = c1 // 2
        self.cv1 = Conv(c1, c_hidden, 1, 1)
        self.cv2 = Conv(c_hidden * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1,
                               padding=x // 2) for x in k])

    def forward(self, input_list):
        out_list = []
        out_list.append(self.cv1(self.ops[0](input_list[0])))
        for m, i, o in zip(self.m, input_list, self.ops):
            out_list.append(m(self.cv1(o(i))))
        return self.cv2(torch.cat(out_list, 1))

    def arch_param_define(self, prev_ch, f_ch, device, path_freezing=False,
                          refer_ch=None, neck_channel=[80, 160, 320],
                          ch_res_rule=0, fidx=None):
        refer_ch = sorted(refer_ch)
        neck_channel = sorted(neck_channel)
        self.ops = nn.ModuleList([])
        if ch_res_rule == 0:  # from 'Backbone'
            print(f'==reference channels==')
            for fl, c in enumerate(refer_ch):
                print(f'P{fl}={c}', end='  ')
            print(f'\n(expect P{neck_channel.index(prev_ch)} {prev_ch})')
            for ii, ch in enumerate(f_ch):
                print(f'--input #{ii}: P{refer_ch.index(ch)} {ch} {fidx[ii]}')
                if fidx[ii] == -1:
                    print("identity....")
                    self.ops.append(torch.nn.Identity())
                    continue
                if refer_ch.index(ch) > neck_channel.index(prev_ch):
                    temp_ops = []
                    in_channels = ch
                    for n in range(refer_ch.index(ch)
                                   - neck_channel.index(prev_ch)):
                        # TODO: Channel- should be performed
                        # up to its backbone, like CBS, CBL, CBM, CBR, CBR6...
                        temp_ops.append(
                            CBS(in_channels,  # CBR6(in_channels,
                                neck_channel[refer_ch.index(ch)-(n+1)],
                                k=1, s=1))
                        temp_ops.append(torch.nn.Upsample(None, 2, 'nearest'))
                        in_channels = neck_channel[refer_ch.index(ch)-(n+1)]

                    self.ops.append(torch.nn.Sequential(*temp_ops))
                elif refer_ch.index(ch) < neck_channel.index(prev_ch):
                    temp_ops = []
                    in_channels = ch
                    for n in range(neck_channel.index(prev_ch)
                                   - refer_ch.index(ch)):
                        # TODO: Downsampling & Channel+ should be performed
                        # up to its backbone, like CBS, CBL, CBM, CBR, CBR6...
                        temp_ops.append(
                            CBS(in_channels,  # CBR6(in_channels,
                                neck_channel[refer_ch.index(ch)+(n+1)],
                                k=3, s=2))
                        in_channels = neck_channel[refer_ch.index(ch)+(n+1)]
                    self.ops.append(torch.nn.Sequential(*temp_ops))
                else:  # refer_ch.index(ch) == neck_channel.index(prev_ch)
                    if ch == prev_ch:
                        self.ops.append(torch.nn.Identity())
                    else:  # (tenace comment: it is not going to happen...)
                        self.ops.append(
                            CBS(ch,  # CBR6(ch,
                                neck_channel[refer_ch.index(ch)], k=3, s=1))
        elif ch_res_rule == 1:  # from 'PreLayer'
            print(f'==reference channels==')
            for fl, c in enumerate(refer_ch):
                print(f'P{fl}={c}', end='  ')
            print(f'\n(expect P{neck_channel.index(prev_ch)} {prev_ch})')
            for ii, ch in enumerate(f_ch):
                print(f'--input #{ii}: P{refer_ch.index(ch)} {ch}')
                if fidx[ii] == -1:
                    self.ops.append(torch.nn.Identity())
                    continue

                if refer_ch.index(ch) > neck_channel.index(prev_ch):
                    temp_ops = []
                    in_channels = ch
                    for n in range(refer_ch.index(ch)
                                   - neck_channel.index(prev_ch)):
                        # (tenace comment) Upsampling must be
                        #                  with channel reducing operation
                        #                  Attach an additional CBR6
                        temp_ops.append(
                            CBS(in_channels,
                                neck_channel[refer_ch.index(ch)-(n+1)],
                                k=1, s=1))
                        temp_ops.append(torch.nn.Upsample(None, 2, 'nearest'))
                        in_channels = neck_channel[refer_ch.index(ch)-(n+1)]
                    self.ops.append(torch.nn.Sequential(*temp_ops))

                elif refer_ch.index(ch) < neck_channel.index(prev_ch):
                    temp_ops = []
                    in_channels = ch
                    for n in range(neck_channel.index(prev_ch)
                                   - refer_ch.index(ch)):
                        # (tenace comment) Downsampling must be
                        #                  with channel increasing operation
                        # temp_ops.append(CBR6(ch, ch, k=3, s=2))
                        temp_ops.append(
                            CBS(in_channels,
                                neck_channel[refer_ch.index(ch)+(n+1)],
                                k=3, s=2))
                    self.ops.append(torch.nn.Sequential(*temp_ops))

                else:
                    self.ops.append(torch.nn.Identity())
        else:
            raise NotImplementedError('Check ch_res_rule')


class Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    # detection layer
    def __init__(self, nc=80, anchors=(), ch_in=(), inplace=True):
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        # init anchor grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]
        # shape(nl,na,2)
        self.register_buffer(
            'anchors',
            torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.last_conv = nn.ModuleList(
            nn.Conv2d(x, self.no * self.na, 1) for x in ch_in)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.last_conv[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs,
                             self.na,
                             self.no,
                             ny,
                             nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = \
                        self._make_grid(nx, ny, i)

                # if isinstance(self, Segment):  # (boxes + masks)
                # To be checked later
                if False:
                    xy, wh, conf, mask = x[i].split((
                        2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    # xy
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]
                    # wh
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) \
            if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0,
                   torch_1_10=check_version(torch.__version__, '1.10.0')):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), \
            torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij') \
            if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        # add grid offset, i.e. y = 2.0 * x - 0.5
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5
        anchor_grid = (self.anchors[i]
                       * self.stride[i]).view((1,
                                               self.na,
                                               1,
                                               1,
                                               2)).expand(shape)
        return grid, anchor_grid


class NMS(nn.Module):
    """ Non-Maximum Suppression (NMS) module """
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf,
                                   iou_thres=self.iou, classes=self.classes)
