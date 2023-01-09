import torch
# import nni.retiarii.nn.pytorch as nn
import torch.nn as nn
from autonn.ops import *

import yaml
from pathlib import Path
from copy import deepcopy


class Neck(nn.Module):
    def __init__(self,  backbone, neck_cfg='yaml/neck.yaml', np=3):
        super().__init__()

        # Temp: read yaml file
        self.cfg_file = Path(neck_cfg).name
        with open(neck_cfg, encoding='utf-8', errors='ignore') as f:
            self.cfg = yaml.safe_load(f)
        self.layers, self.channels, self.connect = \
            parse_neck(deepcopy(self.cfg), backbone, np)

        # TO-DO : search neck architecture, not from yaml file
        # self.layers. self.channels. self.connect = \
        #     search_neck(xxx, backbone, np)

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


def parse_neck(nd, bm, p):
    """ nd=neck_model_dictionary,
        bm=backbone_module,
        p=number_pyramid-floors """
    # layers from pre-built backbone
    layers = bm.layers
    si = len(layers)  # starting index of 'neck'

    connect_fr = []  # backbone layers
    for t in bm.connect:
        connect_fr.append(t[0])
    # channels from pre-built backbone
    ch = bm.channels
    # print(f'backbone output channels : {ch}')

    neck_layers = []
    neck_ch = []
    # from, number(repetition), module(layer), arguments
    for i, (f, n, m, args) in enumerate(nd['neck']):
        m = eval(m) if isinstance(m, str) else m  # nn module name
        for j, a in enumerate(args):
            try:
                # arguments (list)
                args[j] = eval(a) if isinstance(a, str) else a
            except BaseException:
                pass
        if m in [CBR, CBR6, CBL, CBS, CBM, Bottleneck, CSP, SPP]:
            ch_in, ch_out = ch[f], args[0]
            # [ch_in, ch_out] if CSP else [ch_in, ch_out, kernel, stride]
            args = [ch_in, ch_out, *args[1:]]
            if m is CSP:
                # [ch_in, ch_out, repetition, shortcut]
                args.insert(2, n)
        elif m is MB:
            # [ch_in, ch_multiple(t), ch_out(c), repetition(n), stride(s)]
            args.insert(0, ch[f])
        elif m is Concat:
            for idx, x in enumerate(f):
                if isinstance(x, str) and x == "Backbone":
                    f[idx] = connect_fr.pop(-2)
                elif x != -1:
                    f[idx] += si
            ch_out = sum([ch[x if x < 0 else x] for x in f])

        m_ = m(*args)  # nn module
        t = str(m)[8:-2].replace('__main__.', '')
        split_t = t.split(".")
        t = split_t[-1]
        params = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type, m_.params = si+i, f, t, params
        neck_layers.append(m_)
        # ch[x] : input channel for layer-#x
        ch.append(ch_out)
        neck_ch.append(ch_out)

    # connection between neck to head
    output = deepcopy(nd['neck-to-head'])
    for k in range(len(output)):
        output[k] += si

    """ for debugging """
    # print(f'---------------------backbone-------------------------')
    # for index, layer in enumerate(layers):
    #     print('[%2d] from:%8s %9s %10d'
    #           % (layer.i, layer.f, layer.type, layer.params))
    #     if index in output:  # these are connecting to 'head'
    #         print(f'------------------------------------------------------')
    #     elif index in connect_fr:
    #         print(f'---------------------neck-----------------------------')

    # print(f'---------------------backbone-------------------------')
    # for index, layer in enumerate(layers):
    #     print('[%2d] from:%8s %9s %10d'
    #           % (layer.i, layer.f, layer.type, layer.params))
    # print(f'-----------------------neck---------------------------')
    # for index, layer in enumerate(neck_layers):
    #     print('[%2d] from:%8s %9s %10d'
    #           % (layer.i, layer.f, layer.type, layer.params))
    #     if index in nd['neck-to-head']:  # these are connecting to 'head'
    #         print(f'------------------------------------------------------')

    return nn.ModuleList([*neck_layers]), neck_ch, output


""" for unit test """
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser('neck_parser')
    parser.add_argument(
        '--backbone_cfg', type=str,
        default='basemodel.yaml', help='backbone.yaml'
    )
    parser.add_argument(
        '--neck_cfg', type=str,
        default='neck.yaml', help='neck.yaml'
    )
    args = parser.parse_args()

    from backbone import Backbone
    backbone = Backbone(args.backbone_cfg, np=3).cuda()
    backbone_and_neck = Neck(backbone, args.neck_cfg, np=3).cuda()

    x = torch.rand(1, 3, 608, 608).cuda()
    out = backbone_and_neck(x)  # warm-up
