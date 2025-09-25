# import argparse
import math
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
# import nni.retiarii.nn.pytorch as nn
from nni.retiarii.nn.pytorch import LayerChoice

from .models.ops_syolov4 \
    import (OPS, MBottleneckCSP2, ActConv, MBottleneck, Conv,
            Bottleneck, BottleneckCSP, BottleneckCSP2, SPP,
            SPPCSP, Concat)

from .syolo_utils.general import check_anchor_order, make_divisible
from .syolo_utils.torch_utils \
    import (time_synchronized, fuse_conv_and_bn, model_info, scale_img)


class Detect(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.stride = None  # strides computed during build
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        # shape(nl,1,na,1,1,2)
        self.register_buffer('anchor_grid',
                             a.clone().view(self.nl, 1, -1, 1, 1, 2))
        # output conv
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.export = False  # onnx export

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no,
                             ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                # xy
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 +
                               self.grid[i].to(x[i].device)) * self.stride[i]
                # wh
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class SearchYolov4(nn.Module):
    def __init__(self, cfg='yolov4-p5.yaml', names=None, hyp=None, weights='',
                 ch=3, nc=None):
        super(SearchYolov4, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                # model dict
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)

        # Define model
        if nc and nc != self.yaml['nc']:
            print('Overriding %s nc=%g with nc=%g' %
                  (cfg, self.yaml['nc'], nc))
            self.yaml['nc'] = nc    # override yaml value
        # model, savelist, ch_out
        self.model, self.save = self._parse_model(deepcopy(self.yaml), ch=[ch])
        self.weights = weights

        # Model parameters
        self.nc, self.names = nc, names  # attach number of classes to model
        self.hyp = hyp  # attach hyperparameters to model
        self.gr = 1.0   # giou loss ratio (obj_loss = 1.0 or giou)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            # forward
            m.stride = \
                torch.tensor([s / x.shape[-2] for x
                             in self.forward(torch.zeros(1, ch, s, s))])
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        self.info()
        print('')

    def _parse_model(self, d, ch):  # model_dict, input_channels(3)
        print('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params',
                                                'module', 'arguments'))
        anchors, nc, gd, gw = \
            d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
        # number of anchors
        na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
        no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

        layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
        # from, number, module, args
        for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
            m = eval(m) if isinstance(m, str) else m  # eval strings
            for j, a in enumerate(args):
                try:
                    # eval strings
                    args[j] = eval(a) if isinstance(a, str) else a
                except Exception:
                    pass

            n = max(round(n * gd), 1) if n > 1 else n  # depth gain
            if m in [nn.Conv2d, Conv, ActConv, Bottleneck, MBottleneck, SPP,
                     BottleneckCSP, BottleneckCSP2, MBottleneckCSP2, SPPCSP]:
                c1, c2 = ch[f], args[0]

                # Normal
                # if i > 0 and args[0] != no:  # channel expansion factor
                #     ex = 1.75  # exponential (default 2.0)
                #     e = math.log(c2 / ch[1]) / math.log(2)
                #     c2 = int(ch[1] * ex ** e)
                # if m != Focus:

                c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

                # Experimental
                # if i > 0 and args[0] != no:  # channel expansion factor
                #     ex = 1 + gw  # exponential (default 2.0)
                #     ch1 = 32  # ch[1]
                #     e = math.log(c2 / ch1) / math.log(2)  # level 1-n
                #     c2 = int(ch1 * ex ** e)
                # if m != Focus:
                #     c2 = make_divisible(c2, 8) if c2 != no else c2

                args = [c1, c2, *args[1:]]
                if m in [BottleneckCSP, BottleneckCSP2,
                         MBottleneckCSP2, SPPCSP]:
                    args.insert(2, n)
                    n = 1
            elif m is nn.BatchNorm2d:
                args = [ch[f]]
            elif m is Concat:
                c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
            elif m is Detect:
                args.append([ch[x + 1] for x in f])
                if isinstance(args[1], int):  # number of anchors
                    args[1] = [list(range(args[1] * 2))] * len(f)
            else:
                c2 = ch[f]

            if m is MBottleneckCSP2:
                op_candidates = [OPS['3x3_relu_BNCSP2'](*args),
                                 OPS['3x3_leaky_BNCSP2'](*args),
                                 OPS['3x3_mish_BNCSP2'](*args),
                                 OPS['5x5_relu_BNCSP2'](*args),
                                 OPS['5x5_leaky_BNCSP2'](*args),
                                 OPS['5x5_mish_BNCSP2'](*args)]
                layer_op = LayerChoice(op_candidates, label="m_{}".format(i))
                m_ = m(layer_op, op_candidates)
            else:
                m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 \
                    else m(*args)  # module
            t = str(m)[8:-2].replace('__main__.', '')  # module type
            np = sum([x.numel() for x in m_.parameters()])  # number params
            # attach index, 'from' index, type, number params
            m_.i, m_.f, m_.type, m_.np = i, f, t, np
            # print
            print('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))
            save.extend(x % i for x in ([f] if isinstance(f, int) else f)
                        if x != -1)  # append to savelist
            layers.append(m_)
            ch.append(c2)
        return nn.Sequential(*layers), sorted(save)

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si)
                yi = self.forward_once(xi)[0]  # forward
                # save
                # cv2.imwrite('img%g.jpg' % s, 255 *
                #             xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            # single-scale inference, train
            return self.forward_once(x, profile)

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                # from earlier layers
                x = y[m.f] if isinstance(m.f, int) \
                    else [x if j == -1 else y[j] for j in m.f]

            if profile:
                try:
                    import thop
                    o = thop.profile(m, inputs=(x,), verbose=False)[0] \
                        / 1E9 * 2  # FLOPS
                except Exception:
                    o = 0
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    # initialize biases into Detect(), cf is class frequency
    def _initialize_biases(self, cf=None):
        # cf = torch.bincount(
        #     torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(),
        #     minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            # obj (8 objects per 640 image)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None \
                else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) %
                  (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             # shortcut weights
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ', end='')
        for m in self.model.modules():
            if type(m) is Conv:
                # pytorch 1.6.0 compatability
                m._non_persistent_buffers_set = set()
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                m.bn = None  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        # self.info()
        return self

    def info(self):  # print model information
        model_info(self)

    def init_model(self):
        # Model init
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                # nn.init.kaiming_normal_(m.weight, mode='fan_out',
                #                         nonlinearity='relu')
                pass
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
                m.inplace = True
