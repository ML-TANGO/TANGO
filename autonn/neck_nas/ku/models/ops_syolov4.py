from nni.retiarii.serializer import basic_unit
import torch
import nni.retiarii.nn.pytorch as nn
import torch.nn.functional as F
from nni.retiarii.nn.pytorch import LayerChoice

import math
# import torch.nn as nn
from models.common import Conv, autopad
from mish_cuda import MishCuda as Mish
import collections


OPS = {
    # '3x3_BNCSP2': lambda c1, c2, n: MBottleneckCSP2Layer(c1, c2, n, 3, activation='none'),
    '3x3_relu_BNCSP2': lambda c1, c2, n: MBottleneckCSP2Layer(c1, c2, n, 3, activation='relu'),
    '3x3_leaky_BNCSP2': lambda c1, c2, n: MBottleneckCSP2Layer(c1, c2, n, 3, activation='leaky'),
    '3x3_mish_BNCSP2': lambda c1, c2, n: MBottleneckCSP2Layer(c1, c2, n, 3, activation='mish'),
    # '5x5_BNCSP2': lambda c1, c2, n: MBottleneckCSP2Layer(c1, c2, n, 5, activation='none'),
    '5x5_relu_BNCSP2': lambda c1, c2, n: MBottleneckCSP2Layer(c1, c2, n, 5, activation='relu'),
    '5x5_leaky_BNCSP2': lambda c1, c2, n: MBottleneckCSP2Layer(c1, c2, n, 5, activation='leaky'),
    '5x5_mish_BNCSP2': lambda c1, c2, n: MBottleneckCSP2Layer(c1, c2, n, 5, activation='mish')
}

class MBottleneckCSP2(nn.Module):
    def __init__(self, mutable_bottleneckcsp2_layer, op_candidates_list):
        super(MBottleneckCSP2, self).__init__()

        self.mutable_bottleneckcsp2_layer = mutable_bottleneckcsp2_layer
        self.op_candidates_list = op_candidates_list

    def forward(self, x):
        out = self.mutable_bottleneckcsp2_layer(x)
        return out

class ActConv(nn.Module):
    # convolution with optional activation function
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, activation='mish'):   # ch_in, ch_out, kernel, stride, padding, groups, activation function
        super(ActConv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)

        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'leaky':
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation == 'mish':
            self.act = Mish()
        elif activation == 'none':
            self.act = nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    def fuseforward(self, x):
        return self.act(self.conv(x))

class MBottleneck(nn.Module):
    # Mutable activation bottleneck
    def __init__(self, c1, c2, kernel_size=3, shortcut=True, g=1, e=0.5, activation='mish'):  # ch_in, ch_out, shortcut, groups, expansion, activation function
        super(MBottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels

        self.cv1 = ActConv(c1, c_, 1, 1, activation=activation)
        self.cv2 = ActConv(c_, c2, k=kernel_size, s=1, g=g, activation=activation)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class MBottleneckCSP2Layer(nn.Module):
    # Mutable activation BottleneckCSP2
    def __init__(self, c1, c2, n=1, kernel_size=3, shortcut=False, g=1, e=0.5, activation='mish'):  # ch_in, ch_out, number, shortcut, groups, expansion, activation function
        super(MBottleneckCSP2Layer, self).__init__()
        c_ = int(c2)  # hidden channels

        self.cv1 = ActConv(c1, c_, 1, 1, activation=activation)
        self.cv2 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv3 = ActConv(2 * c_, c2, 1, 1, activation=activation)
        self.bn = nn.BatchNorm2d(2 * c_)

        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'leaky':
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation == 'mish':
            self.act = Mish()
        elif activation == 'none':
            self.act = nn.Identity()

        self.m = nn.Sequential(*[MBottleneck(c_, c_, kernel_size=kernel_size, shortcut=shortcut, g=g, e=1.0, activation=activation) for _ in range(n)])

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(self.act(self.bn(torch.cat((y1, y2), dim=1))))

    @staticmethod
    def is_zero_layer():
        return False


# class SPPCSP(nn.Module):
#     # CSP SPP https://github.com/WongKinYiu/CrossStagePartialNetworks
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
#         super(SPPCSP, self).__init__()
#         c_ = int(2 * c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
#         self.cv3 = Conv(c_, c_, 3, 1)
#         self.cv4 = Conv(c_, c_, 1, 1)
#         self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
#         self.cv5 = Conv(4 * c_, c_, 1, 1)
#         self.cv6 = Conv(c_, c_, 3, 1)
#         self.bn = nn.BatchNorm2d(2 * c_) 
#         self.act = Mish()
#         self.cv7 = Conv(2 * c_, c2, 1, 1)

#     def forward(self, x):
#         x1 = self.cv4(self.cv3(self.cv1(x)))
#         y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
#         y2 = self.cv2(x)
#         return self.cv7(self.act(self.bn(torch.cat((y1, y2), dim=1))))
