from tkinter.messagebox import NO
from django.http import JsonResponse
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.parsers import JSONParser
from .serializers import NodeSerializer
from .serializers import PthSerializer
from .serializers import EdgeSerializer
from .serializers import ArchitectureSerializer

from .models import Node
from .models import Edge
from .models import Pth
from .models import Architecture

from rest_framework import status
from django.core.files import File
from .graph import *
from .binder import *
import torch
import random
import string
import os
# Create your views here.

@api_view(['GET', 'POST'])
def nodeList(request):
    nodes = Node.objects.all()
    serializer = NodeSerializer(nodes, many=True)
    return Response(serializer.data)


@api_view(['GET'])
def edgeList(request):
    edges = Edge.objects.all()
    serializer = EdgeSerializer(edges, many=True)
    return Response(serializer.data)


@api_view(['GET', 'POST', 'DELETE', 'UPDATE'])
def pthList(request):
    if request.method == 'GET':
        pth = Pth.objects.all()
        serializer = PthSerializer(pth, many=True)
        return Response(serializer.data)
    elif request.method == 'POST':
        edges = Edge.objects.all()
        nodes = Node.objects.all()
        if nodes and edges:
            created_model = test_branches(nodes, edges)
            file_path = (os.getcwd()+ '/model_'+random_char(8)+'.pth').replace("\\", '/')

            torch.save(created_model, file_path)
            serializer = PthSerializer(data={'model_output': file_path})
            if serializer.is_valid():
                print("valid")
                serializer.save()
                return Response(serializer.data, status=status.HTTP_201_CREATED)
            else:
                print("invalid")
                print(serializer.errors)
                serializer.save()
                return Response("invalid pth", status=status.HTTP_400_BAD_REQUEST)
        return Response("invalid node or edge", status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET', 'POST', 'DELETE', 'UPDATE'])
def architectureList(request):
    architecture = Architecture.objects.all()
    serializer = ArchitectureSerializer(architecture, many=True)
    return Response(serializer.data)


class NodeView(viewsets.ModelViewSet):
    serializer_class = NodeSerializer
    queryset = Node.objects.all()


class EdgeView(viewsets.ModelViewSet):
    serializer_class = EdgeSerializer
    queryset = Edge.objects.all()


class PthView(viewsets.ModelViewSet):
    serializer_class = PthSerializer
    queryset = Pth.objects.all()

class ArchitectureView(viewsets.ModelViewSet):
    serializer_class = ArchitectureSerializer
    queryset = Architecture.objects.all()

def test_branches(get_node, get_edge):
    graph = c_Graph()
    for node in get_node:
        string = "{parameters}".format(**node.__dict__).replace("\n", ',')
        graph.addNode(c_Node("{order}".format(**node.__dict__), type="{layer}".format(**node.__dict__), params=eval("{"+string+"}")))
    for edge in get_edge:
        graph.addEdge(c_Edge("{prior}".format(**edge.__dict__), "{next}".format(**edge.__dict__)))
    net = c_PyBinder.exportModel(graph)
    return net


def random_char(y):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for x in range(y))