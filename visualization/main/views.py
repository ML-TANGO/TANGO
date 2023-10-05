"""
high level support for doing this and that.
"""
import os
import random
import string
from time import sleep
from datetime import datetime
import torch
import requests
from django.core import serializers
import json
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
    nodes = Node.objects.all()
    serializer = NodeSerializer(nodes, many=True)
    return Response(serializer.data)


@api_view(['GET'])
def edgelist():
    '''
    edge list
    '''
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

        #name = random_char(8)
        name = 'resnet50'

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



            if serializer.is_valid():
                print("valid")
                serializer.save()
                # pylint: disable = invalid-name, missing-timeout, unused-variable

                # get user_id & project_id from status_request
                get_status = Status.objects.all()
                user_id = get_status[len(get_status)-1].user_id
                project_id = get_status[len(get_status)-1].project_id

                json_path = ('/shared/common/'+str(user_id)+'/'+str(project_id)+'/basemodel.json').replace("\\", '/')
                with open(json_path, 'w', encoding="utf-8") as make_file:
                    json.dump(json_data, make_file, ensure_ascii=False, indent='\t')

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


    return None


@api_view(['GET', 'POST', 'DELETE', 'UPDATE'])
def sortlist(request):
    '''
    pth list
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
    pth list
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
        try:
            user_id = request.GET['user_id']
            project_id = request.GET['project_id']
            serializer = StartSerializer(data={'msg': 'started',
                                               'user_id': user_id,
                                               'project_id': project_id})
            if serializer.is_valid():
                serializer.save()
                try:
                    nodes = Node.objects.all()
                    nodes.delete()

                    edges = Edge.objects.all()
                    edges.delete()

                    copyfile('/shared/common/'+str(user_id)+'/'+str(project_id)+'/basemodel.json', '/visualization/frontend/src/resnet50.json')
                    #copyfile('./frontend/src/resnet50.json', './frontend/src/VGG16.json')
                    with open("./frontend/src/resnet50.json", "r", encoding="utf-8-sig") as f:
                        data = json.load(f)
                        for i in data.get('node'):
                            serializer = NodeSerializer(data=i)
                            if serializer.is_valid():
                                serializer.save()
                        for j in data.get('edge'):
                            serializer = EdgeSerializer(data=j)
                            if serializer.is_valid():
                                serializer.save()

                    print('start api GET')

                except Exception as ex:
                    print(ex)
                return Response("started", status=200, content_type="text/plain")
            else:
                return Response("error", status=200, content_type="text/plain")
        # pylint: disable = broad-except
        except Exception as e:
            # return HttpResponse(e, HttpResponse)
            #return HttpResponse('error', content_type="text/plain")
            return Response("error", status=200, content_type="text/plain")


    elif request.method == 'POST':
        with open("./frontend/src/resnet50.json", "r") as f:
            data = json.load(f)
            if (len(data["node"]) > 1) :
                return True
            else:
                return False


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
    if request.method == 'GET':
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

            #print("get_time:", get_time)
            #time = get_time.text[-16:-3]

            # 현재 시점의 timestamp와 비교하기
            saved_time = datetime.fromtimestamp(int(1696232398148)/1000)

            now = datetime.now()
            print("now: ", now)
            print(type(now))
            diff = now - saved_time
            diff_sec = diff.seconds

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
    # pylint: disable = no-else-return, no-member
    if request.method == 'GET':
        running = Running.objects.all()
        serializer = RunningSerializer(running, many=True)
        return Response(serializer.data)
    elif request.method == 'POST':
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