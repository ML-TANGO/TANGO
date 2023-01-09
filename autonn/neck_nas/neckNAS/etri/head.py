import torch
# import nni.retiarii.nn.pytorch as nn
import torch.nn as nn
from autonn.ops import *

import yaml
from pathlib import Path
from copy import deepcopy


class Head(nn.Module):
    def __init__(self, backbone, neck=None, dataset='yaml/dataset.yaml', np=3):
        super().__init__()

        if neck is None:
            bpn = backbone
            self.layers = bpn.layers
            channels = bpn.channels
            connect = bpn.connect
        else:
            if isinstance(neck, str):
                with open(neck, encoding='utf-8', errors='ignore') as f:
                    neck = yaml.safe_load(f)

            self.layers = backbone.layers
            self.layers.append(neck.layers)
            channels = backbone.channels
            channels.append(neck.channels)
            connect = neck.connect

        ch_in = []
        for i in connect:
            ch_in.append(channels[i])

        with open(dataset, encoding='utf-8') as f:
            self.yaml = yaml.load(f, Loader=yaml.SafeLoader)
        self.nc = self.yaml['num_classes']
        self.anchors = self.yaml['anchors']
        self.classes = self.yaml['names']
        self.np = np if np else len(self.anchors)  # 3 (P3, P4, P5)

        m_ = Detect(nc=self.nc, anchors=self.anchors, ch_in=ch_in)
        t = "Detect"
        params = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type, m_.params = len(self.layers), connect, t, params

        self.layers.append(m_)

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


""" for unit test """
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser('neck_parser')
    parser.add_argument('--backbone_cfg', type=str,
                        default='yaml/mobilenetv2_backbone.yaml',
                        help='backbone.yaml')
    parser.add_argument('--neck_cfg', type=str,
                        default='yaml/neck_for_mobilenetv2.yaml',
                        help='neck.yaml')
    parser.add_argument('--dataset', type=str,
                        default='yaml/dataset.yaml',
                        help='datasets.yaml')
    args = parser.parse_args()

    from backbone import Backbone
    backbone = Backbone(args.backbone_cfg, np=3).cuda()

    from neck import Neck
    backbone_plus_neck = Neck(backbone, args.neck_cfg, np=3).cuda()

    model = Head(backbone_plus_neck, args.dataset, np=3).cuda()

    """ for debugging """
    print(f'---------------------network-------------------------')
    for index, layer in enumerate(model.layers):
        print('[%2d] from:%12s %9s %10d'
              % (layer.i, layer.f, layer.type, layer.params))
    print(f'-----------------------------------------------------')

    x = torch.rand(1, 3, 608, 608).cuda()
    out = model(x)  # warm-up
