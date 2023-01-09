""" MIT License
Copyright (c) 2022 Hyunwoo Cho
"""
import logging
import yaml
from pathlib import Path
from copy import deepcopy

import torch
# import nni.retiarii.nn.pytorch as nn
import torch.nn as nn
from autonn.ops import *
from autonn.yolov5_utils.general import LOGGER

# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)


class Backbone(nn.Module):
    """ Backbone network class to creates nn.Module from yaml file.

    This will examine backbone architecture for find and save the connetions
    from this backbone to any kind of neck. Since CNNs are like a shape of
    pyramid as tensors are moving forward along the network, we can find out
    exit points on each stage of the pyramid, which are connecting to
    corresponding entry points of neck network.


    Parameters
    ----------
    cfg : str
        Path to yaml file containing backbone network architecture.
    np : int
        Number of starges in a pyramid for multi-scaled features

    Attributes
    ----------
    np : int
        number of pyramid-stages (default: 3)
    cfg_file : str
        Name of yaml file without directory (ex. basemodel.yaml)
    cfg : dict
        Dictonary loaded from yaml file
    layers : nn.ModuleList (deprecated, do not use it to build model)
        Backbone network
    channels : list of int
        Channel length of tensors connecting to neck
    connect : list of int
        Index of layers connecting to neck (start:0 )
    size : list of int
        Denominator of size (ex. 32 means 1/32 size)
    """
    def __init__(self, cfg='yaml/basemodel.yaml', np=3):
        super().__init__()
        self.np = np  # number of pyramid-floors conneting to neck

        # read yaml file
        self.cfg_file = Path(cfg).name
        with open(cfg, encoding='ascii', errors='ignore') as f:
            self.cfg = yaml.safe_load(f)

        # parse yaml file
        self.layers, self.channels, self.connect, self.size = \
            parse_backbone(deepcopy(self.cfg), np)

    def forward(self, x):
        return self._forward_once(x)

    def _forward_once(self, x):
        y = []  # outputs
        for m in self.layers:
            if m.f != -1:  # if it is not from previous layer
                x = y[m.f] if isinstance(m.f, int) \
                    else [x if j == -1 else y[j] for j in m.f]
            x = m(x)  # run
            y.append(x)
        return x


def parse_backbone(d, p):  # d=model_dictionary, p=number_pyramid-floors
    LOGGER.debug('\n%3s%18s%10s  %-20s%-30s'
                % ('', 'from', 'params', 'module', 'arguments'))
    layers = []
    output = []  # layers connecting between backbone and neck
    ch = [3]  # default : RGB(3-ch) images
    sz = [1]  # if image size is 640 then input size for a layer is 640/sz
    for i, (f, n, m, args) in enumerate(d['backbone']):
        # from, number(repetition), module(layer), arguments
        m = eval(m) if isinstance(m, str) else m   # nn module name
        for j, a in enumerate(args):
            args[j] = eval(a) if isinstance(a, str) else a  # arguments (list)
        if m in [Focus, Conv, CBR, CBR6, CBL, CBS, CBM, Bottleneck, CSP]:
            ch_in, ch_out = ch[f], args[0]
            sz_in = sz_out = sz[f]
            args = [ch_in, ch_out, *args[1:]]
            # [ch_in, ch_out] if CSP else [ch_in, ch_out, kernel, stride]
            if m is CSP:
                args.insert(2, n)  # [ch_in, ch_out, repetition, shortcut]
            elif m is Bottleneck:
                pass
            else:
                if m is Focus:
                    sz_out = sz_in * 2
                elif i > 0 and args[3] == 2:
                    output.append(i-1)
                    sz_out = sz_in * 2
        elif m is MB:
            ch_in, ch_out = ch[f], args[1]
            sz_in = sz_out = sz[f] 
            args.insert(0, ch_in)
            # [ch_in, ch_multiple(t), ch_out(c), repetition(n), stride(s)]
            if i > 0 and args[4] == 2:
                output.append(i-1)
                sz_out = sz_in * 2

        if i == len(d['backbone'])-1:
            output.append(i)

        m_ = m(*args)  # nn module
        t = str(m)[8:-2].replace('__main__.', '')
        split_t = t.split(".")
        t = split_t[-1]
        params = sum([x.numel() for x in m_.parameters()])
        m_.i, m_.f, m_.type, m_.params = i, f, t, params
        # print(f'{m_.type} added: {len(list(m_.modules()))}')
        layers.append(m_)
        LOGGER.debug('%3s%18s%10.0f  %-20s%-30s'
                    % (i, f, params, t, args))  # print
        # ch[x] : input channel for layer-#x
        if i == 0:
            ch = []
            sz = []
        ch.append(ch_out)
        sz.append(sz_out)

    # create meta data for backbone-to-neck connection
    connects = {}
    for index, layer in enumerate(layers):
        if index in output[-p:]:
            layer.cp = True
            connects[index] = ch[index]
        else:
            layer.cp = False
    connects = sorted(connects.items())
    # connects = sorted(connects.items(), \
    #     key = lambda item: item[0], reverse = True)
    # warning : 'connects' is a list composed of tuples now, not a dictionary

    """ for debugging """
    # print(f'---------------------backbone-------------------------')
    # for index, layer in enumerate(layers):
    #     to = '+1,neck' if layer.cp else '+1'
    #     print(
    #         f'[{layer.i}] from:{layer.f},'
    #         + f'to:{to}\t {layer.type}\t {layer.params}'
    #     )
    #     if index in output:  # this output is connecting to 'neck'
    #         print(f'------------------------------------------------------')
    return nn.ModuleList([*layers]), ch, connects, sz


""" for unit test """
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser('backbone_parser')
    parser.add_argument(
        '--cfg', type=str,
        default='basemodel.yaml', help='basemodel.yaml'
    )
    parser.add_argument(
        '--np', type=int,
        default=3, help='number of pyramid-floors'
    )
    args = parser.parse_args()
    backbone = Backbone(args.cfg, args.np).cuda()

    x = torch.rand(1, 3, 604, 604).cuda()
    out = backbone(x)  # warm-up
