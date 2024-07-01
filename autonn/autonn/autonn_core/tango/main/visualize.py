import os, sys
import yaml, json
from collections import OrderedDict
import logging
from pathlib import Path
from copy import deepcopy

import torch.nn as nn

from . import Node, Edge
from django.core import serializers
from autonn_core.serializers import NodeSerializer
from autonn_core.serializers import EdgeSerializer


logger = logging.getLogger(__name__)

class Viz:
    def __init__(self, uid, pid):
        self.userid = uid
        self.projid = pid
        self.initialize_viz()

    def initialize_viz(self):
        # clear all nodes & edges before constructing a new architecture
        self.nodes = Node.objects.all()
        self.edges = Edge.objects.all()
        self.nodes.delete()
        self.edges.delete()

    def parse_model(self, basemodel_yaml):
        # load basemode.yaml
        if os.path.isfile(basemodel_yaml):
            with open(basemodel_yaml) as f:
                basemodel = yaml.safe_load(f)
        else:
            logger.warn(f"not found {basemodel_yaml}")
            return

        self.base_yaml = basemodel_yaml

        # parse basemodel
        nc = basemodel.get('nc', 80)  # default number of classes = 80 (coco)
        ch = [basemodel.get('ch', 3)] # default channel = 3 (RGB)
        layers, lines, c2, edgeid = [], [], ch[-1], 0
        for i, (f, n, m, args) in enumerate(basemodel['backbone'] + basemodel['head']):  # from, number, module, args
            logger.debug(f"💧 Read yaml layer-{i} : ({f}, {n}, {m}, {args})")

            # node parsing -------------------------------------------------
            # m = eval(m) if isinstance(m, str) else m  # eval strings
            node = OrderedDict()
            for j, a in enumerate(args):
                try:
                    args[j] = eval(a) if isinstance(a, str) else a  # eval strings
                except Exception as e:
                    if a == 'nc':
                        args[j] = nc
                    elif a == 'anchors':
                        args[j] = basemodel.get('anchors', ())
                    elif isinstance(a, nn.Module):
                        # for example, nn.LeakeyReLU(0.1)
                        args[j] = a
                    elif a in ('nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'):
                        # list of upsampling mode
                        args[j] = a
                    else:
                        logger.warn(f"unsupported arguements: {a}...ignored.")

            if   m == 'nn.Conv2d':
                c1 = ch[f]
                c2 = args[0]
                k = args[1]
                s = args[2]
                p = args[3]
                b = args[4]
                params = (
                    f"'in_channels': {c1} \n "
                    f"'out_channels': {c2} \n "
                    f"'kernel_size': ({k}, {k}) \n "
                    f"'stride': ({s}, {s}) \n "
                    f"'padding': ({p}, {p}) \n "
                    f"'bias': {b}"
                )
            elif m == 'nn.BatchNorm2d':
                c1 = ch[f]
                c2 = c1
                if len(args) > 0:
                    c2 = args[0]
                if c1 != c2:
                    logger.warn(f"Error! BatchNorm2d has to be the same features in {c1} & out {c2} of it")
                    c2 = c1
                params = (
                    f"'num_features': {c2}"
                )
            elif m == 'nn.MaxPool2d':
                c1 = ch[f]
                c2 = c1
                k = args[0]
                s = args[1]
                p = args[2]
                d = 1
                r = False
                c = False
                if len(args) > 3:
                    d = args[3]
                    if len(args) > 4:
                        r = args[4]
                        if len(args) > 5:
                            c = args[5]
                params = (
                    f"'kernel_size': ({k}, {k}) \n "
                    f"'stride': ({s}, {s}) \n "
                    f"'padding': ({p}, {p}) \n "
                    f"'dilation': {d} \n "
                    f"'return_indices': {r} \n "
                    f"'ceil_mode': {c}"
                )
            elif m == 'nn.AvgPool2d':
                c1 = ch[f]
                c2 = c1
                k = args[0]
                s = args[1]
                p = args[2]
                params = (
                    f"'kernel_size': ({k}, {k}) \n "
                    f"'stride': ({s}, {s}) \n "
                    f"'padding': ({p}, {p})"
                )
            elif m == 'nn.AdaptiveAvgPool2d':
                c1 = ch[f]
                c2 = c1
                o = args[0]
                params = (
                    f"'output_size': ({o}, {o})"
                )
            elif m == 'nn.ConstantPad2d':
                c1 = ch[f]
                c2 = c1
                p = args[0]
                v = args[1]
                params = (
                    f"'padding': {p} \n "
                    f"'value': {v}"
                )
            elif m == 'nn.ZeroPad2d':
                c1 = ch[f]
                c2 = c1
                p = args[0]
                params = (
                    f"'padding': {p}"
                )
            elif m in ('nn.ReLU', 'nn.ReLU6'):
                c1 = ch[f]
                c2 = c1
                inp = True
                if len(args) > 0:
                    inp = args[0]
                params = (
                    f"'inplace': {inp}"
                )
            elif m in ('nn.Sigmoid', 'nn.Tanh'):
                c1 = ch[f]
                c2 = c1
                params = ()
            elif m == 'nn.LeakyReLU':
                c1 = ch[f]
                c2 = c1
                neg = args[0]
                inp = True
                if len(args) > 1:
                    inp = args[1]
                params = (
                    f"'negative_slope': {neg} \n "
                    f"'inplace': {inp}"
                )
            elif m == 'nn.Softmax':
                c1 = ch[f]
                c2 = c1
                d = args[0]
                params = (
                    f"'dim': {d}"
                )
            elif m == 'nn.Linear':
                c1 = ch[f]
                c2 = args[0]
                b = True
                if len(args) > 1:
                    b = args[1]
                params = (
                    f"'in_features': {c1} \n "
                    f"'out_features': {c2} \n "
                    f"'bias': {b}"
                )
            elif m == 'nn.Dropout':
                c1 = ch[f]
                c2 = c1
                p = args[0]
                inp = True
                if len(args) > 1:
                    inp = args[1]
                params = (
                    f"'p': {p} \n "
                    f"'inplace': {inp}"
                )
            elif m == 'nn.MESLoss':
                c1 = ch[f]
                c2 = c1
                avg = args[0]
                r1 = args[1]
                r2 = args[2]
                params = (
                    f"'size_average': {avg} \n "
                    f"'reduce': {r1} \n "
                    f"'reduction': {r2}"
                )
            elif m == 'nn.BCELoss':
                c1 = ch[f]
                c2 = c1
                w = args[0]
                avg = args[1]
                r1 = args[2]
                r2 = args[3]
                params = (
                    f"'weight': {w} \n "
                    f"'size_average': {avg} \n"
                    f"'reduce': {r1} \n "
                    f"'reduction': {r2}"
                )
            elif m == 'nn.CrossEntropyLoss':
                c1 = ch[f]
                c2 = c1
                w = args[0]
                avg = args[1]
                ign_idx = args[2]
                r1 = args[3]
                r2 = args[4]
                lsmooth = args[5]
                params = (
                    f"'weight': {w} \n "
                    f"'size_average': {avg} \n "
                    f"ignore_index': {ign_idx} \n "
                    f"reduce': {r1} \n "
                    f"'reduction': {r2} \n"
                    f"'label_smoothing': {lsmooth}"
                )
            elif m == 'Flatten':
                # reshape input into a 1-dim tensor
                # hard to say how many output channel is
                c1 = ch[f]
                s_dim = args[0]
                e_dim = args[1]
                params = (
                    f"'start_dim': {s_dim} \n "
                    f"'end_dim': {e_dim}"
                )
            elif m == 'nn.Upsample':
                c1 = ch[f]
                c2 = c1
                size = args[0]
                scale = args[1]
                mode = 'nearest'
                align, recompute = False, False
                if len(args)>2:
                    mode = args[2]
                if len(args)>3:
                    align = args[3]
                if len(args)>4:
                    recompute = args[4]
                params = (
                    f"'size': {size} \n "
                    f"'scale_factor': {scale} \n "
                    f"'mode': {mode} \n "
                    f"'align_corners': {align} \n "
                    f"'recompute_scale_factor': {recompute}"
                )
            elif m in ('BasicBlock', 'Bottleneck'):
                expansion = 1
                if m == 'Bottleneck':
                    expansion = 4
                inplanes = ch[f]
                planes = args[0]
                c1 = inplanes
                c2 = planes * expansion
                s = args[1]
                downsample = args[2]
                g = args[3]
                basewidth = args[4]
                d = args[5]
                norm_layer = args[6]
                params = (
                    f"'inplanes': {inplanes} \n "
                    f"'planes': {planes} \n "
                    f"'stride': ({s}, {s}) \n "
                    f"'downsample': {downsample} \n "
                    f"'groups': {g} \n "
                    f"'base_width': {basewidth} \n "
                    f"'dilation': {d} \n "
                    f"'norm_layer': {norm_layer}"
                )
            elif m == 'Conv':
                c1 = ch[f]
                c2 = args[0]
                k = args[1]
                s = args[2]
                p = args[3]
                g = args[4]
                a = args[5]
                params = (
                    f"'in_channels': {c1} \n "
                    f"'out_channels': {c2} \n "
                    f"'kernel_size': {k} \n "
                    f"'stride': {s} \n "
                    f"'pad': {p} \n "
                    f"'groups': {g} \n "
                    f"'act': {a}"
                )
            elif m == 'Concat':
                d = args[0]
                if not isinstance(f, list):
                    c1 = ch[f]
                    c2 = c1
                else:
                    c1 = [ch[x] for x in f]
                    if d == 1: # (N, C, H, W); channel-wise concatentation
                        c2 = sum(c1)
                    else:
                        logger.warn("warning! only channel-wise concat is supported.")
                        c2 = max(c1) # TODO: should be treated more elegantly..
                params = (
                    f"'dim': {d}"
                )
            elif m == 'Shortcut':
                d = args[0]
                if isinstance(f, int):
                    c1 = ch[f]
                    c2 = c1
                else:
                    c1 = ch[f[0]]
                    for x in f:
                        if ch[x] != c1:
                            logger.warn("warning! all input must have the same dimension")
                    c2 = c1
                params = (
                    f"'dim': {d}"
                )
            elif m == 'DownC':
                c1 = ch[f]
                c2 = args[0]
                n = 1
                if len(args) > 1:
                    n = args[1]
                    if len(args) > 2:
                        k = args[2]
                params = (
                    f"'in_channels': {c1} \n "
                    f"'out_channels': {c2} \n "
                    f"'n': {n} \n "
                    f"'kernel_size': {k}"
                )
            elif m == 'SPPCSPC':
                c1 = ch[f]
                c2 = args[0]
                n = args[1]
                shortcut = args[2]
                g = args[3]
                e = args[4]
                k = args[5]
                params = (
                    f"'in_channels': {c1} \n "
                    f"'out_channels': {c2} \n "
                    f"'n': {n} \n "
                    f"'shortcut': {shortcut} \n "
                    f"'groups': {g} \n"
                    f"'expansion': {e} \n"
                    f"'kernels': {k}"
                )
            elif m == 'ReOrg':
                c1 = ch[f]
                c2 = 4 * c1
                params = ()
            elif m == 'MP':
                c1 = ch[f]
                c2 = c1
                k = 2
                if len(args) > 0:
                    k = args[0]
                params = (
                    f"'k': {k}"
                )
            elif m == 'SP':
                c1 = ch[f]
                c2 = c1
                k = 3
                s = 1
                if len(args) > 1:
                    k = args[0]
                    s = args[1]
                elif len(args) == 1:
                    k = args[0]
                params = (
                    f"'kernel_size': {k} \n "
                    f"'stride': {s}"
                )
            elif m == 'IDetect':
                c2 = None
                nc = args[0]
                if isinstance(f, list):
                    nl = len(f) # number of detection layers
                    c1 = [ch[x] for x in f]
                else:
                    logger.warn("warning! detection module needs two or more inputs")
                    nl = 1
                    c1 = [ch[f]]
                anchors = [] # viz2code needs to store this
                if len(args)>1:
                    if isinstance(args[1], list):
                        # anchors = len(args[1])
                        if len(args[1]) != nl:
                            print(f"warning! the number of detection layer is {nl},"
                                  f" but anchors is for {len(args[1])} layers.")
                        anchors = args[1]
                    else:
                        anchors = [list(range(args[1]*2))] * nl
                ch_ = []
                if len(args)>2:
                    ch_ = args[2]
                params = (
                    f"'nc': {nc} \n "
                    f"'anchors': {anchors} \n "
                    f"'ch': {ch_}"
                )
            else:
                logger.warn(f"unsupported module... {m}")
                c1 = ch[f]
                c2 = c1
                params = ()

            # append node --------------------------------------------------
            node['order'] = i + 1 # start from 1
            if 'nn.' in m:
                m = m.replace('nn.', '')
            node['layer'] = m
            node['parameters'] = params
            layers.append(node)
            logger.info(f"🚩 Create a node #{node['order']} : {node['layer']} - args \n {node['parameters']}")

            # manipulate input channel -------------------------------------
            if i == 0:
                ch = []
            ch.append(c2)

            # edge parsing -------------------------------------------------
            if i == 0:
                continue

            prior = f if isinstance(f, list) else [f]
            for p in prior:
                edge = OrderedDict()
                if p < 0:
                    p += (i+1)
                edgeid = edgeid + 1
                edge['id'] = edgeid
                edge['prior'] = p
                edge['next'] = i + 1
                lines.append(edge)
                logger.info(f"  ※ Create an edge #{edge['id']} : {edge['prior']}->{edge['next']}")

        self.base_dict = deepcopy(basemodel)
        self.layers = layers
        self.lines = lines

    def update(self):
        json_data = OrderedDict()
        json_data['node'] = self.layers
        json_data['edge'] = self.lines

        for i in json_data.get('node'):
            serializer = NodeSerializer(data=i)
            if serializer.is_valid():
                serializer.save()
        for j in json_data.get('edge'):
            serializer = EdgeSerializer(data=j)
            if serializer.is_valid():
                serializer.save()




