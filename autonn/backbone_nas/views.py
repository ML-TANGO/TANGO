'''
views.py
'''

import os

import torch
from django.shortcuts import render
# from rest_framework.response import Response
from rest_framework.decorators import api_view

from .net_generator.run_backbone_nas import run_nas

# from rest_framework import viewsets
# from .serializers import URSSerializer
from . import models


def index(request):
    '''index'''
    return render(request, 'backboneNAS/index.html')


@api_view(['POST'])
def URSList(request):
    '''URSList'''
    if request.method == 'POST':
        # Fetching the form data
        uploadedFile = request.FILES["data_yaml"]

        # Saving the information in the database
        urs = models.URS(
            data_yaml=uploadedFile
        )
        urs.save()
        # urss = models.URS.objects.all()
        return render(request, "backboneNAS/index.html")
        # user_reqs = models.URS.objects.all()
        # serializer = URSSerializer(user_reqs, many=True)
        # return Response(serializer.data)


@api_view(['GET'])
def create_net(request):
    '''create_net'''
    if request.method == 'GET':
        # URS id 관리 추가 예정
        # models.URS.objects.filter
        user_reqs = models.URS.objects.all()
        data_path = user_reqs[0].data_yaml.url
        created_model = run_nas(data_path)

        created_model_name = 'best_det_backbone.pth'
        created_model_path = (
            os.path.dirname(os.path.abspath(__file__)) +
            "/media/temp_files/model/" + created_model_name)
        torch.save(created_model.state_dict(), created_model_path)

        return render(request, "backboneNAS/index.html",
                      context={"path": created_model_path})

        # file_path = (os.getcwd()+ '/model_'+ \
        #  random_char(8)+'.pth').replace("\\", '/')
        # torch.save(created_model, file_path)
        # serializer = PthSerializer(data={'model_output': file_path})

# class URSView(viewsets.ModelViewSet):
#    serializer_class = URSSerializer
#    queryset = models.URS.objects.all()
