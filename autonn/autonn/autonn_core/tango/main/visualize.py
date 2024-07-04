import os, sys
import yaml, json
from collections import OrderedDict
import logging
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn

from . import Info, Node, Edge, Pth
from django.core import serializers
from autonn_core.serializers import NodeSerializer
from autonn_core.serializers import EdgeSerializer

from tango.viz.graph import CGraph, CEdge, CNode, CShow2
from tango.viz.binder import CPyBinder

logger = logging.getLogger(__name__)

TORCH_NN_MODULES =  {   'Conv2d', 'ConvTranspose2d',
                        'MaxPool2d', 'AvgPool2d', 'AdaptiveMaxPool2d', 'AdaptiveAvgPool2d',
                        'ZeroPad2d',
                        'ReLU', 'ReLU6', 'GELU', 'Sigmoid', 'SiLU', 'Mish', 'Tanh', 'Softmax',
                        'BatchNorm2d', 'LayerNorm',
                        'TransformerEncoder', 'TransformerDecoder',
                        'Identify', 'Linear', 'Dropout',
                        'Embedding',
                        'MSELoss', 'CrossEntropyLoss', 'BCELoss',
                        'Upsample'
                    }
HEAD_MODULES = { 'Classify', 'Detect', 'IDetect', 'IAuxDetect', 'IKeypoint',
                 'IBin', 'Segment', 'Pose'}


class BasemodelViewer:
    def __init__(self, uid, pid):
        self.userid = uid
        self.projid = pid
        self.initialize()

    def initialize(self):
        # clear all nodes & edges before constructing a new architecture
        nodes = Node.objects.all()
        edges = Edge.objects.all()
        nodes.delete()
        edges.delete()

        self.base_yaml = ''
        self.base_dict = {}
        self.layers = []
        self.lines = []

    def parse_yaml(self, basemodel_yaml):
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
                            logger.warn(f"warning! the number of detection layer is {nl},"
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

        self.initialize()

        for i in json_data.get('node'):
            serializerN = NodeSerializer(data=i)
            if serializerN.is_valid():
                serializerN.save()
        for j in json_data.get('edge'):
            serializerE = EdgeSerializer(data=j)
            if serializerE.is_valid():
                serializerE.save()

        try:
            info = Info.objects.get(userid=self.userid, project_id=self.projid)
            info.progress = 'viz_update'
            info.save()
        except Info.DoesNotExist:
            logger.warn(f'not found {userid}/{project_id} information')

def export_pth(file_path):
    '''
    Make a graph with nodes and edges and export pytorch model from the graph
    '''
    nodes = Node.objects.all()
    edges = Edge.objects.all()

    graph = CGraph()
    self_binder = CPyBinder()
    for node in nodes:
        # pylint: disable-msg=bad-option-value, consider-using-f-string
        params_string = "{parameters}". \
            format(**node.__dict__).replace("\n", '>')
        # print(f"{params_string}")

        # tenace -------------------------------------------------------------->
        # workaround to avoid eval() error when a value is string or nn.Module
        params_dict = {}
        params_ = params_string.split('>')
        for p in params_:
            try:
                eval_params_ = eval("{"+p+"}")
            except:
                # print(p)
                p_key, p_value = p.split(': ') # [0] key [1] value
                if 'LeakyReLU' in p_value:
                    p_value = f"nn.{p_value}"
                    eval_params_ = eval("{"+p_key+": "+p_value+"}")
                elif isinstance(p_value, str):
                    # print(f"---{p_key}---{p_value}---")
                    p_key = p_key.strip()
                    p_value = p_value.strip()
                    # print(f"---{p_key}---{p_value}---")
                    p_key = p_key.replace("'","")
                    p_value = p_value.replace("'", "")
                    # print(f"---{p_key}---{p_value}---")
                    eval_params_[p_key.strip("'")] = p_value
                else:
                    # print("forced to convert string-to-dictionary")
                    p_key.strip()
                    p_value.strip()
                    p_key = p_key.replace("'","")
                    p_value = p_value.replace("'", "")
                    eval_params_[p_key] = p_value
            finally:
                params_dict.update(eval_params_)
        # print(f"{params_dict}")
        # tenace <--------------------------------------------------------------

        # pylint: disable-msg=bad-option-value, eval-used
        graph.addnode(CNode("{order}".format(**node.__dict__),
                            type_="{layer}".format(**node.__dict__),
                            params=params_dict)) # params=eval("{" + params_string + "}")))
    for edge in edges:
        # pylint: disable-msg=bad-option-value, consider-using-f-string
        graph.addedge(CEdge("{prior}".format(**edge.__dict__),
                            "{next}".format(**edge.__dict__)))
    net = CPyBinder.exportmodel(self_binder, graph)
    logger.info(net)
    torch.save(net, file_path)
    return net

def export_yml(name, yaml_path):
    '''
    Make a graph with nodes and edges and export pytorch model from the graph
    '''
    nodes = Node.objects.all()
    edges = Edge.objects.all()

    node_layer_list = []
    node_order_list = []
    node_parameters_list = []

    edge_id_list = []
    edge_prior_list = []
    edge_next_list = []

    for node in nodes:
        node_order_list.append(node.order)
        node_layer_list.append(node.layer)
        node_parameters_list.append(node.parameters)

    for edge in edges:
        edge_id_list.append(edge.id)
        edge_prior_list.append(edge.prior)
        edge_next_list.append(edge.next)

    # json ---------------------------------------------------------------------
    #json_data = serializers.serialize('json', nodes)
    json_data = OrderedDict()
    json_data['node'] = []
    json_data['edge'] = []

    for c in range(len(node_order_list)):
        json_data['node'].append({
            "order": node_order_list[c],
            "layer": node_layer_list[c],
            "parameters": node_parameters_list[c]
        })

    for a in range(len(edge_id_list)):
        json_data['edge'].append({
            "id": edge_id_list[a],
            "prior": edge_prior_list[a],
            "next": edge_next_list[a]
        })

    logger.info(json.dumps(json_data, ensure_ascii=False, indent="\t"))

    # yolo-style yaml_data generation ------------------------------------------
    yaml_data = {}
    yaml_data['name'] = name
    yaml_data['hyp'] = 'p5'
    yaml_data['imgsz'] = 640
    yaml_data['nc'] = 80
    yaml_data['depth_multiple'] = 1.0
    yaml_data['width_multiple'] = 1.0
    yaml_data['anchors'] = 3
    yaml_data['backbone'] = []
    yaml_data['head'] = []

    # for yaml_index, node_index in enumerate(nodes_order):
    #     json_index = node_order_list.index(node_index)
    for yaml_index, node_index in enumerate(node_order_list):
        json_index = node_index - 1

        # YOLO-style yaml module description
        # [from, number, module, args]
        number_ = 1                             # repetition
        module_ = node_layer_list[json_index]   # pytorch nn module

        # module & its arguements
        # str_params = "{"+node_parameters_list[json_index]+"}"
        # str_params = str_params.replace('\n', ',')
        # params_ = eval(str_params)
        str_params = node_parameters_list[json_index]

        params_ = {}
        str_param = str_params.split('\n')
        for p in str_param:
            try:
                eval_params_ = eval("{"+p+"}")
            except:
                # print(p)
                p_key, p_value = p.split(': ') # [0] key [1] value
                if 'LeakyReLU' in p_value:
                    p_value = f"nn.{p_value}"
                    eval_params_ = eval("{"+p_key+": "+p_value+"}")
                elif isinstance(p_value, str):
                    # print(f"---{p_key}---{p_value}---")
                    p_key = p_key.strip()
                    p_value = p_value.strip()
                    # print(f"---{p_key}---{p_value}---")
                    p_key = p_key.replace("'","")
                    p_value = p_value.replace("'", "")
                    # print(f"---{p_key}---{p_value}---")
                    eval_params_[p_key.strip("'")] = p_value
                else:
                    print("forced to convert string-to-dictionary")
                    p_key.strip()
                    p_value.strip()
                    p_key = p_key.replace("'","")
                    p_value = p_value.replace("'", "")
                    eval_params_[p_key] = p_value
            finally:
                params_.update(eval_params_)

        args_ = []
        if module_ == 'Conv2d':
            ch_ = params_['out_channels']
            k_ = params_['kernel_size'][0]
            s_ = params_['stride'][0]
            p_ = params_['padding'][0]
            b_ = params_['bias']
            args_ = [ch_, k_, s_, p_, b_]
        elif module_ == 'BatchNorm2d':
            ch_ = params_['num_features']
            args_ = [ch_]
        elif module_ in ('MaxPool2d', 'AvgPool2d'):
            k_ = params_['kernel_size'][0]
            s_ = params_['stride'][0]
            p_ = params_['padding'][0]
            args_ = [k_, s_, p_]
        elif module_ == 'AdaptiveAvgPool2d':
            o_ = params_['output_size']
            if o_[0] == o_[1]:
                args_ = [o_[0]]
            else:
                args_ = o_
        elif module_ == 'MP':
            k_ = params_['k']
            if k_ == 2:
                args_ = []
            else:
                args_ = [k_]
        elif module_ == 'SP':
            k_ = params_['kernel_size']
            s_ = params_['stride']
            args_ = [k_, s_]
        elif module_ == 'ConstantPad2d':
            p_ = params_['padding']
            v_ = params_['value']
            args_ = [p_, v_]
        elif module_ == 'ZeroPad2d':
            p_ = params_['padding']
            args_ = [p_]
        elif module_ in ('ReLU', 'ReLU6', 'Sigmoid', 'LeakyReLU', 'Tanh'):
            args_ = []
        elif module_ == 'Softmax':
            d_ = params_['dim']
            args_ = [d_]
        elif module_ == 'Linear':
            o_ = params_['out_features']
            b_ = params_['bias']
            args_ = [o_, b_]
        elif module_ == 'Dropout':
            p_ = param_['p']
            args_ = [p_]
        elif module_ in ('BCELoss', 'CrossEntropyLoss', 'MESLoss'):
            r_ = params_['reduction']
            args_ = [r_]
        elif module_ == 'Flatten':
            # TODO: needs to wrap this with export-friendly classes
            d_st = params_['start_dim']
            d_ed = params_['end_dim']
            args_ = [d_st, d_ed]
        elif module_ == 'ReOrg':
            args_ = []
        elif module_ == 'Upsample':
            s_ = params_['size']
            f_ = params_['scale_factor']
            m_ = params_['mode']
            args_ = [s_, f_, m_]
        elif module_ in ('Bottleneck', 'BasicBlock'):
            # torchvision.models.resnet
            ch_ = params_['planes']
            s_ = params_['stride'][0]
            g_ = params_['groups']
            d_ = params_['dilation']
            n_ = params_['norm_layer']
            downsample_ = params_['downsample']
            w_ = params_['base_width']
            norm_ = params_['norm_layer']
            args_ = [ch_, s_, downsample_, g_, w_, d_, norm_]
        elif module_ == 'Conv':
            ch_ = params_['out_channels']
            k_ = params_['kernel_size']
            s_ = params_['stride']
            p_ = params_['pad']
            g_ = params_['groups']
            a_ = params_['act']
            args_ = [ch_, k_, s_, p_, g_, a_]
        elif module_ == 'Concat':
            d_ = params_['dim']
            args_ = [d_]
        elif module_ == 'Shortcut':
            d_ = params_['dim']
            args_ = [d_]
        elif module_ == 'DownC':
            ch_ = params_['out_channels']
            n_ = params_['n']
            k_ = params_['kernel_size'][0]
            args_ = [ch_, n_, k_]
        elif module_ == 'SPPCSPC':
            ch_ = params_['out_channels']
            n_ = params_['n']
            sh_ = params_['shortcut']
            g_ = params_['groups']
            e_ = params_['expansion']
            k_ = params_['kernels']
            args_ = [ch_, n_, sh_, g_, e_, k_]
        elif module_ == 'IDetect':
            nc_ = params_['nc']
            anchors_ = params_['anchors']
            ch_ = params_['ch']
            args_ = [nc_, anchors_, ch_]
        else:
            print(f"{module_} is not supported yet")
            continue
        # yaml_index: 0, 1, 2, ...
        # node_index: 1, 2, 3, ...
        # json_index: 1, 2, 3, ...
        logger.info(f"layer #{yaml_index} (node_index #{node_index}; json_index #{json_index}) : {module_}")

        # from
        f_ = []
        for a in range(len(edge_id_list)):
            if edge_next_list[a] == node_index:
                f_.append(edge_prior_list[a])
        logger.info(f"f_={f_}")
        if not f_:
            from_ = -1 # this has to be the first layer
            assert yaml_index == 0, f'it must be the first layer but index is {yaml_index}'
        elif len(f_) == 1:
            # x = nodes_order.index(f_[0])
            from_ = f_[0] - 1 - yaml_index # node_index = yaml_index + 1
            if from_ < -5:
                # too far to calcaulate prior node, write explicit node number instead.
                from_ = f_[0] - 1
        else:
            # 2 or more inputs
            f_multiple = []
            for f_element in f_:
                # x = nodes_order.index(f_element)
                # if x == nodes_order[yaml_index-1]:
                #     x = -1
                x = f_element - 1 - yaml_index
                if x < -5:
                    x = f_element - 1
                f_multiple.append(x)
            if all(num < 0 for num in f_multiple):
                f_multiple.sort(reverse=True)
            else:
                f_multiple.sort(reverse=False)
            from_ = f_multiple
        logger.info(f"from : {from_}")

        if module_ in TORCH_NN_MODULES:
            module_ = f"nn.{module_}"
        layer_ = [from_, number_, module_, args_]
        if module_ in HEAD_MODULES:
            yaml_data['head'].append(layer_)
        else:
            yaml_data['backbone'].append(layer_)

    # print yaml data
    # print('-'*100)
    # print(yaml_data)
    # print('-'*100)
    # print(yaml.dump(yaml_data, sort_keys=False, default_flow_style=False))
    logger.debug('-'*100)
    for k, v in yaml_data.items():
        if isinstance(v, list):
            if len(v):
                logger.debug(f"{k}: ")
                for v_element in v:
                    logger.debug(f"\t- {v_element}")
            else:
                logger.debug(f"{k}: {v}")
        else:
            logger.debug(f"{k}: {v}")
    logger.debug('-'*100)

    with open(yaml_path, 'w') as f:
                yaml.dump(yaml_data, f)

    return yaml_data
