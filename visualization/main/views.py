"""
high level support for doing this and that.
"""
import os
import random
import string
from time import sleep
from datetime import datetime
import torch
import torch.nn as nn
import requests
from django.core import serializers
import json, yaml
from collections import OrderedDict
from shutil import copyfile

from rest_framework.decorators import api_view
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import render
from django.http import HttpResponse

from .serializers import NodeSerializer
from .serializers import PthSerializer
from .serializers import EdgeSerializer
from .serializers import ArchitectureSerializer
from .serializers import StartSerializer
from .serializers import StatusSerializer
from .serializers import RunningSerializer
from .serializers import StopSerializer
from .serializers import SortSerializer

from .models import Node
from .models import Edge
from .models import Pth
from .models import Architecture
from .models import Start
from .models import Running
from .models import Sort
from .models import Status

from .graph import CGraph, CEdge, CNode, CShow2
from .binder import CPyBinder
import json

# Create your views here.

# https://pytorch.org/docs/stable/nn.html
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

@api_view(['GET', 'POST', 'DELETE', 'UPDATE'])
# pylint: disable = invalid-name, inconsistent-return-statements
def mainList(request):
    '''
        main List
    '''
    # pylint: disable = no-else-return, no-member
    if request.method == 'GET':
        return render(request, 'index.html')
    return render(request, 'index.html')


@api_view(['GET', 'POST'])
def nodelist():
    '''
    node list
    '''
    print('node')
    if request.method == 'GET':
        print('get')
    if request.method == "POST":
        print('post')

    nodes = Node.objects.all()
    serializer = NodeSerializer(nodes, many=True)
    return Response(serializer.data)


@api_view(['GET'])
def edgelist():
    '''
    edge list
    '''
    print('edge')
    if request.method == 'GET':
        print('get')

    edges = Edge.objects.all()
    serializer = EdgeSerializer(edges, many=True)
    return Response(serializer.data)


@api_view(['GET', 'POST', 'DELETE', 'UPDATE'])
def pthlist(request):
    '''
    pth list
    '''
    print("pth")

    if request.method == 'GET':
        print("get")
        pth = Pth.objects.all()
        serializer = PthSerializer(pth, many=True)
        return Response(serializer.data)

    if request.method == 'POST':
        print("post")
        #CShow2()
        host_ip = str(request.get_host())[:-5]
        #print(host_ip)

        edges = Edge.objects.all()
        nodes = Node.objects.all()

        name = random_char(8)
        # name = 'resnet50'

        if nodes and edges:
            created_model = make_branches(nodes, edges)
            file_path = (os.getcwd() + '/model_' +
                         name + '.pt').replace("\\", '/')

            torch.save(created_model, file_path)
            serializer = PthSerializer(data={'model_output': file_path})


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


            print(json.dumps(json_data, ensure_ascii=False, indent="\t"))

            # yolo-style yaml_data generation
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

            # sort the node and re-order node_order_list -------------------------------------------------------------->
            # TODO: buggy codes here... needs to solve this later on
            # for n in edge_next_list:
            #     if n not in edge_prior_list:
            #         # this is the last node
            #         last_node = n
            # for p in edge_prior_list:
            #     if p not in edge_next_list:
            #         # this is the first node
            #         first_node = p

            # print(f"first node = #{first_node}, last node = #{last_node}")
            # print('-'*50)
            # print(f"node[0] = #{first_node}")
            # print('-'*50)

            # total_node = len(node_order_list)
            # nodes_order = [0 for i in range(total_node)]
            # nodes_order[0] = first_node
            # # nodes_order[-1] = last_node

            # reserved_nodes = []
            # for i in range(total_node):
            #     print(f" prior node = {nodes_order[i]}")
            #     next_nodes = []
            #     print(" look up the edge prior list...")
            #     for index, p in enumerate(edge_prior_list):
            #         if p == nodes_order[i]:
            #             # print(f"  edge-#{index}: prior node={p}")
            #             next_nodes.append(edge_next_list[index])
            #     print(f" next nodes = {next_nodes}")

            #     if not next_nodes:
            #         # 3 possible cases when next nodes are empty
            #         # - the last node
            #         # - the end node of a branch
            #         # - error(not the last node, but no reserved nodes)
            #         if i+1 == total_node:
            #             print(" this is the last node")
            #         else:
            #             if not reserved_nodes:
            #                 print(f" error: this is not the last node,"
            #                       f" but does not have any connection from output port")
            #             else:
            #                 # not the last node but just a branch end,
            #                 # search other branch from reserved nodes
            #                 next_node = reserved_nodes.pop(reserved_nodes.index(min(reserved_nodes)))
            #     else:
            #         if len(next_nodes) > 1:
            #             # 2 or more next nodes : this node must be a branch point
            #             next_node = next_nodes.pop(next_nodes.index(min(next_nodes)))
            #             reserved_nodes = reserved_nodes + next_nodes
            #         else:
            #             # only 1 next node : simple connection
            #             next_node = next_nodes[0]
            #     if next_node in nodes_order:
            #         print(f" warning: this node (#{next_node}) is already allocated in nodes_order, ignored..")
            #     else:
            #         print('-'*50)
            #         print(f"node[{i+1}] = #{next_node}")
            #         print('-'*50)
            #         nodes_order[i+1] = next_node
            # if len(reserved_nodes) > 0:
            #     print(f" error: reserved nodes left...{reserved_nodes}")

            # print(nodes_order)
            # sort the node and re-order node_order_list <--------------------------------------------------------------

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
                print(f"layer #{yaml_index} (node_index #{node_index}; json_index #{json_index}) : {module_}")

                # from
                f_ = []
                for a in range(len(edge_id_list)):
                    if edge_next_list[a] == node_index:
                        f_.append(edge_prior_list[a])
                print(f"f_={f_}")
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
                print(f"from : {from_}")

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
            print('-'*100)
            for k, v in yaml_data.items():
                if isinstance(v, list):
                    if len(v):
                        print(f"{k}: ")
                        for v_element in v:
                            print(f"\t- {v_element}")
                    else:
                        print(f"{k}: {v}")
                else:
                    print(f"{k}: {v}")
            print('-'*100)

            if serializer.is_valid():
                print("valid")
                serializer.save()
                # pylint: disable = invalid-name, missing-timeout, unused-variable

                # get user_id & project_id from status_request
                get_status = Status.objects.all()
                user_id = get_status[len(get_status)-1].user_id
                project_id = get_status[len(get_status)-1].project_id

                # save json file
                json_path = ('/shared/common/'+str(user_id)+'/'+str(project_id)+'/basemodel.json').replace("\\", '/')
                with open(json_path, 'w', encoding="utf-8") as make_file:
                    json.dump(json_data, make_file, ensure_ascii=False, indent='\t')

                # save yaml file
                yaml_path = ('/shared/common/'+str(user_id)+'/'+str(project_id)+'/basemodel.yaml')
                with open(yaml_path, 'w') as f:
                    yaml.dump(yaml_data, f)

                print(f"save generated models to {json_path}, {yaml_path}")

                # status_report
                url = 'http://projectmanager:8085/status_report'
                #url = 'http://' + host_ip + ':8091/status_report'
                headers = {
                    'Content-Type': 'text/plain'
                }
                payload = {
                    'container_id':"visualization",
                    'user_id': user_id,
                    'project_id': project_id,
                    'status': "success"
                }
                r = requests.get(url, headers=headers, params=payload)

                return Response(serializer.data,
                                status=status.HTTP_201_CREATED)

            if not serializer.is_valid():
                print("invalid")
                print(serializer.errors)
                #serializer.save()

                # get user_id & project_id from status_request
                get_status = Status.objects.all()
                user_id = get_status[len(get_status)-1].user_id
                project_id = get_status[len(get_status)-1].project_id

                url = 'http://projectmanager:8085/status_report'  ##
                #url = 'http://' + host_ip + ':8091/status_report'
                headers = {
                    'Content-Type': 'text/plain'
                }
                payload = {
                    'container_id':"visualization",
                    'user_id': user_id,
                    'project_id': project_id,
                    'status': "failed"
                }
                r = requests.get(url, headers=headers, params=payload)

                return Response("invalid pth",
                                status=status.HTTP_400_BAD_REQUEST)
        return Response("invalid node or edge",
                        status=status.HTTP_400_BAD_REQUEST)

    if request.method == 'DELETE':
        print('delete')

    if request.method == 'UPDATE':
        print('update')
    return None


@api_view(['GET', 'POST', 'DELETE', 'UPDATE'])
def sortlist(request):
    '''
    sort list
    '''
    print("sort")
    if request.method == 'GET':
        sort = Sort.objects.all()
        serializer = SortSerializer(sort, many=True)
        return Response(serializer.data)
    if request.method == 'POST':
        print("post")
        #CShow2()
        host_ip = str(request.get_host())[:-5]
        print(host_ip)
        edges = Edge.objects.all()
        nodes = Node.objects.all()
        if nodes and edges:
            sorted_ids = post_sorted_id(nodes, edges)
            sorted_ids_str = ''
            for id in sorted_ids:
                sorted_ids_str = sorted_ids_str+id+','
            serializer = SortSerializer(data={'id': 1, 'sorted_ids': sorted_ids_str[:-1]})
            print()
            if serializer.is_valid():
                print("valid")
                serializer.save()
                # pylint: disable = invalid-name, missing-timeout, unused-variable
                return Response(serializer.data,
                                status=status.HTTP_201_CREATED)

            if not serializer.is_valid():
                print("invalid")
                print(serializer.errors)
                serializer.save()
                return Response("invalid sort",
                                status=status.HTTP_400_BAD_REQUEST)
        return Response("invalid node or edge",
                        status=status.HTTP_400_BAD_REQUEST)
    return None


@api_view(['GET', 'POST', 'DELETE', 'UPDATE'])
def sortlist_detail(request, pk):
    '''
    sort detail list
    '''
    try:
        pev_sorted_ids = Sort.objects.get(pk=pk)
    except Sort.DoesNotExist:
        print('sort detail 안 됨 ~~~')
        return Response(status=status.HTTP_404_NOT_FOUND)
    print("sort")
    if request.method == 'GET':
        serializer = SortSerializer(pev_sorted_ids)
        return Response(serializer.data)
    if request.method == 'DELETE':
        pev_sorted_ids.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


@api_view(['GET', 'POST', 'DELETE', 'UPDATE'])
# pylint: disable = invalid-name, inconsistent-return-statements
def startList(request):
    '''
        start List
    '''
    # pylint: disable = no-else-return, no-member
    if request.method == 'GET':
        print(f"________GET  /Start viz2code____________")
        try:
            user_id = request.GET['user_id']
            project_id = request.GET['project_id']
            serializer = StartSerializer(data={'msg': 'started',
                                               'user_id': user_id,
                                               'project_id': project_id})
            print(f"⚡ Get user id {user_id} and project id {project_id} from rest api")

            if serializer.is_valid():
                serializer.save()
                try:
                    print(f"🧹 Clear all nodes & edges")
                    nodes = Node.objects.all()
                    nodes.delete()

                    edges = Edge.objects.all()
                    edges.delete()

                    yaml_path = '/shared/common/'+str(user_id)+'/'+str(project_id)+'/basemodel.yaml'
                    json_path = '/shared/common/'+str(user_id)+'/'+str(project_id)+'/basemodel.json'

                    if os.path.isfile(yaml_path):
                        # YOLO-style YAML file import
                        # copyfile(yaml_path, my_yaml_path)
                        with open(yaml_path) as f:
                            basemodel_yaml = yaml.load(f, Loader=yaml.SafeLoader)

                        print(f"🚛 Load yaml file from {yaml_path}")

                        # 'node' & 'edge' parsing
                        # import torch, torchvision.models.resnet
                        nc = basemodel_yaml.get('nc', 80)  # default number of classes = 80 (coco)
                        ch = [basemodel_yaml.get('ch', 3)] # default channel = 3 (RGB)
                        layers, lines, c2, edgeid = [], [], ch[-1], 0
                        for i, (f, n, m, args) in enumerate(basemodel_yaml['backbone'] + basemodel_yaml['head']):  # from, number, module, args
                            print(f"💧 Read yaml layer-{i} : ({f}, {n}, {m}, {args})")
                            # m = eval(m) if isinstance(m, str) else m  # eval strings
                            node = OrderedDict()
                            for j, a in enumerate(args):
                                try:
                                    args[j] = eval(a) if isinstance(a, str) else a  # eval strings
                                except Exception as e:
                                    if a == 'nc':
                                        args[j] = nc
                                    elif a == 'anchors':
                                        args[j] = basemodel_yaml.get('anchors', ())
                                    elif isinstance(a, nn.Module):
                                        # for example, nn.LeakeyReLU(0.1)
                                        args[j] = a
                                    elif a in ('nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'):
                                        # list of upsampling mode
                                        args[j] = a
                                    else:
                                        print(f"unsupported arguements: {a}...ignored.")

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
                                    print(f"Error! BatchNorm2d has to be the same features in {c1} & out {c2} of it")
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
                                # hard to say how long output channel is
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
                                        print("warning! only channel-wise concat is supported.")
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
                                            print("warning! all input must have the same dimension")
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
                                    print("warning! detection module needs two or more inputs")
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
                                print(f"unsupported module... {m}")
                                c1 = ch[f]
                                c2 = c1
                                params = ()

                            node['order'] = i + 1 # start from 1
                            if 'nn.' in m:
                                m = m.replace('nn.', '')
                            node['layer'] = m
                            node['parameters'] = params
                            layers.append(node)

                            print(f"🚩 Create a node #{node['order']} : {node['layer']} - args \n {node['parameters']}")

                            if i == 0:
                                ch = []
                            ch.append(c2)

                            # 'edge' parsing
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

                                print(f"  ※ Create an edge #{edge['id']} : {edge['prior']}->{edge['next']}")

                        # json formatting for 'node' & 'edge'
                        json_data = OrderedDict()
                        json_data['node'] = layers
                        json_data['edge'] = lines
                        print(f"🚛 Generate json data")
                        print(json.dumps(json_data, ensure_ascii=False, indent="\t"))

                        # save
                        for i in json_data.get('node'):
                            serializer = NodeSerializer(data=i)
                            if serializer.is_valid():
                                serializer.save()
                        for j in json_data.get('edge'):
                            serializer = EdgeSerializer(data=j)
                            if serializer.is_valid():
                                serializer.save()
                        print(f"🏳‍🌈 Save Nodes and Edges")

                    elif os.path.isfile(json_path):
                        # json import
                        # copyfile(json_path, my_json_path)
                        # with open("./frontend/src/resnet50.json", "r", encoding="utf-8-sig") as f:
                        print(f"🚛 Load json file from {json_path}")
                        with open(json_path, "r", encoding="utf-8-sig") as f:
                            data = json.load(f)
                            for i in data.get('node'):
                                serializer = NodeSerializer(data=i)
                                if serializer.is_valid():
                                    serializer.save()
                            for j in data.get('edge'):
                                serializer = EdgeSerializer(data=j)
                                if serializer.is_valid():
                                    serializer.save()
                        print(f"🏳‍🌈 Save Nodes and Edges")
                    else:
                        print(f"not found basemode.yaml neither basemodel.json")
                        return Response("error", status=404, content_type="text/plain")
                except Exception as ex:
                    print(ex)
                return Response("started", status=200, content_type="text/plain")
            else:
                return Response("error", status=200, content_type="text/plain")
        # pylint: disable = broad-except
        except Exception as e:
            # return HttpResponse(e, HttpResponse)
            # return HttpResponse('error', content_type="text/plain")
            return Response("error", status=200, content_type="text/plain")
    elif request.method == 'POST':
        print(f"________POST /Start viz2code____________")
        # with open("./frontend/src/resnet50.json", "r") as f:
        #     data = json.load(f)
        #     if (len(data["node"]) > 1) :
        #         return True
        #     else:
        #         return False
    else:
        print('no request')


@api_view(['GET', 'POST', 'DELETE', 'UPDATE'])
# pylint: disable = invalid-name, inconsistent-return-statements
def stopList(request):
    '''
        stop List
    '''
    # pylint: disable = no-else-return, no-member
    if request.method == 'GET':
        try:
            user_id = request.GET['user_id']
            project_id = request.GET['project_id']
            serializer = StopSerializer(data={'msg': 'finished',
                                              'user_id': user_id,
                                              'project_id': project_id})
            if serializer.is_valid():
                serializer.save()
            #return HttpResponse(serializer.data['msg'],content_type="text/plain")
            return Response(serializer.data['msg'], status=200, content_type="text/plain")
            # return Response(serializer.data)
        # pylint: disable = broad-except
        except Exception as e:
            # return HttpResponse(e, HttpResponse)
            #return HttpResponse('error', content_type="text/plain")
            return Response("error", status=200, content_type="text/plain")
    else:
        print('no request')


@api_view(['GET', 'POST', 'DELETE', 'UPDATE'])
# pylint: disable = invalid-name, inconsistent-return-statements
def statusList(request):
    '''
        status List
    '''
    # pylint: disable = no-else-return, no-member
    print('status')
    if request.method == 'GET':
        print('get')
        try:
            user_id = request.GET['user_id']
            project_id = request.GET['project_id']

            serializer = StatusSerializer(data={'msg': 'status',
                                                'user_id': user_id,
                                                'project_id': project_id
                                                })


            # timestamp 값 가져오기
            host_ip = str(request.get_host())[:-5]
            # pylint: disable=missing-timeout
            print("host_ip:", host_ip)
            get_time = requests.get('http://' +
                                    host_ip +
                                    ':8091/api/running/', verify=False)

            # print("get_time:", get_time)
            #time = get_time.text[-16:-3]

            # 현재 시점의 timestamp와 비교하기
            saved_time = datetime.fromtimestamp(int(1696232398148)/1000)

            now = datetime.now()
            print("now: ", now)
            # print(type(now))
            diff = now - saved_time
            diff_sec = diff.seconds
            print(f"diff: {diff_sec}s")
            #print(type(diff_sec))

            if diff_sec > 5:  # 1분 이상이면
                #started를 running으로 변경
                if serializer.is_valid():
                    serializer.save()
                    print('status request api GET (running)')
                    #return HttpResponse('running', content_type="text/plain")
                    return Response("running", status=200, content_type="text/plain")
                else:
                    print(serializer.errors)
                    print('status request api GET (ready)')
                    #return HttpResponse('ready', content_type="text/plain")
                    return Response("ready", status=200, content_type="text/plain")


            else:
                print('status request api GET (started)')
                #return HttpResponse('started', content_type="text/plain")
                return Response("started", status=200, content_type="text/plain")

        # pylint: disable=broad-except
        except Exception as e:
            print('status request api GET (failed). error: ', e)
            #return HttpResponse('failed', content_type="text/plain")
            return Response("failed", status=200, content_type="text/plain")
    else:
        print('no request')


@api_view(['GET', 'POST', 'DELETE', 'UPDATE'])
# pylint: disable = invalid-name, inconsistent-return-statements
def runningList(request):
    '''
        running List
    '''
    print('running')
    # pylint: disable = no-else-return, no-member
    if request.method == 'GET':
        print('get')
        running = Running.objects.all()
        serializer = RunningSerializer(running, many=True)
        return Response(serializer.data)
    elif request.method == 'POST':
        print('post')
        running = Running.objects.all()
        serializer = RunningSerializer(data=running)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    else:
        print('no request')
    # running = Running.objects.all()
    # print(running.values()[len(running.values()) - 1])


@api_view(['GET', 'POST', 'DELETE', 'UPDATE'])
def architecturelist():
    '''
    architecture list
    '''
    architecture = Architecture.objects.all()
    serializer = ArchitectureSerializer(architecture, many=True)
    return Response(serializer.data)


class NodeView(viewsets.ModelViewSet):
    # pylint: disable=too-many-ancestors
    '''
    Node View
    '''
    serializer_class = NodeSerializer
    queryset = Node.objects.all()

    def print_serializer(self):
        '''
        print serializer class
        '''
        print("Node serializer")

    def print_objects(self):
        '''
        print objects
        '''
        print("Node objects")


class EdgeView(viewsets.ModelViewSet):
    # pylint: disable=too-many-ancestors
    '''
    Edge View
    '''
    serializer_class = EdgeSerializer
    queryset = Edge.objects.all()

    def print_serializer(self):
        '''
        print serializer class
        '''
        print("Edge serializer")

    def print_objects(self):
        '''
        print objects
        '''
        print("Edge objects")


class PthView(viewsets.ModelViewSet):
    # pylint: disable=too-many-ancestors
    '''
    Pth View
    '''
    serializer_class = PthSerializer
    queryset = Pth.objects.all()

    def print_serializer(self):
        '''
        print serializer class
        '''
        print("Pth serializer")

    def print_objects(self):
        '''
        print objects
        '''
        print("Pth objects")


class SortView(viewsets.ModelViewSet):
    # pylint: disable=too-many-ancestors
    '''
    Pth View
    '''
    serializer_class = SortSerializer
    queryset = Sort.objects.all()

    def print_serializer(self):
        '''
        print serializer class
        '''
        print("Sort serializer")

    def print_objects(self):
        '''
        print objects
        '''
        print("Sort objects")


class ArchitectureView(viewsets.ModelViewSet):
    # pylint: disable=too-many-ancestors
    '''
    Architecture View
    '''
    serializer_class = ArchitectureSerializer
    queryset = Architecture.objects.all()

    def print_serializer(self):
        '''
        print serializer class
        '''
        print("Architecture serializer")

    def print_objects(self):
        '''
        print objects
        '''
        print("Architecture queryset")


# pylint: disable = too-many-ancestors
class StartView(viewsets.ModelViewSet):
    '''
        Start View
    '''
    # pylint: disable = no-member
    serializer_class = StartSerializer
    queryset = Start.objects.all()

    def print_serializer(self):
        '''
        print serializer class
        '''
        print("Start serializer")

    def print_objects(self):
        '''
        print objects
        '''
        print("Start queryset")


# pylint: disable = too-many-ancestors


class RunningView(viewsets.ModelViewSet):
    '''
        Running View
    '''
    # pylint: disable = no-member
    serializer_class = RunningSerializer
    queryset = Running.objects.all()

    def print_serializer(self):
        '''
        print serializer class
        '''
        print("Running serializer")

    def print_objects(self):
        '''
        print objects
        '''
        print("Running queryset")


# pylint: disable = too-many-ancestors


# class StopView(viewsets.ModelViewSet):
#     '''
#         Stop View
#     '''
#     # pylint: disable = no-member
#     serializer_class = StopSerializer
#     queryset = Stop.objects.all()

#     def print_serializer(self):
#         '''
#         print serializer class
#         '''
#         print("Stop serializer")

#     def print_objects(self):
#         '''
#         print objects
#         '''
#         print("Stop queryset")


def make_branches(get_node, get_edge):
    '''
    make a graph with nodes and edges and export pytorch model from the graph
    '''
    graph = CGraph()
    self_binder = CPyBinder()
    for node in get_node:
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
                    print("forced to convert string-to-dictionary")
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
    for edge in get_edge:
        # pylint: disable-msg=bad-option-value, consider-using-f-string
        graph.addedge(CEdge("{prior}".format(**edge.__dict__),
                            "{next}".format(**edge.__dict__)))
    net = CPyBinder.exportmodel(self_binder, graph)
    print(net)
    return net


def post_sorted_id(get_node, get_edge):
    '''
    test branches
    '''
    graph = CGraph()
    self_binder = CPyBinder()
    for node in get_node:
        # pylint: disable-msg=bad-option-value, consider-using-f-string
        params_string = "{parameters}". \
            format(**node.__dict__).replace("\n", ',')
        # pylint: disable-msg=bad-option-value, eval-used
        graph.addnode(CNode("{order}".format(**node.__dict__),
                            type_="{layer}".format(**node.__dict__),
                            params=eval("{" + params_string + "}")))
    for edge in get_edge:
        # pylint: disable-msg=bad-option-value, consider-using-f-string
        graph.addedge(CEdge("{prior}".format(**edge.__dict__),
                            "{next}".format(**edge.__dict__)))
    sorted_ids = CPyBinder.sort_id(self_binder, graph)
    return sorted_ids


def random_char(number):
    '''
    return random strings
    '''
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for x in range(number))
