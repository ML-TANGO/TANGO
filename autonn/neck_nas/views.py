'''
views.py
'''

import os
import json
import torch
import requests

from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view

from .ku.main import run_nas

# from rest_framework import viewsets
# from .serializers import URSSerializer
from . import models


# def index(request):
#     '''index'''
#     return render(request, 'neckNAS/index.html')


# @api_view(['POST'])
# def InfoList(request):
#     '''Information List for Neck NAS'''
#     if request.method == 'POST':

#         # Fetching the form data
#         uploadedFile = request.FILES["data_yaml"]
#         usrId = request.data['userid']
#         prjId = request.data['project_id']
#         target = request.data['target']
#         task = request.data['task']
#         sts = request.data['status']

#         # Saving the information in the database
#         updatedInfo = models.NasContainerInfo(
#             userid=usrId,
#             project_id=prjId,
#             target_device=target,
#             data_yaml=uploadedFile,
#             task=task,
#             status=sts
#         )
#         updatedInfo.save()

#         return render(request, "neckNAS/index.html")

@api_view(['GET'])
def start(request):
    print("_________GET /neck/start_____________")
    params = request.query_params
    userid = params['userid']
    project_id = params['project_id']

    # check user id & project id
    # info_list = models.NasContainerInfo.objects.all()
    # saved_userid = [info.userid for info in info_list]
    # saved_projid = [info.project_id for info in info_list]

    # if saved_userid and saved_projid:
    #     if saved_userid != userid or saved_projid != project_id:
    #         Response(status=200, content_type="error")

    # models.NasContainerInfo.userid = userid
    # models.NasContainerInfo.project_id = project_id

    if request.method == 'GET':
        run_nas()
    return Response(status=200, content_type="text/plain")


@api_view(['GET'])
def stop(request):
    print("_________GET /neck/stop_____________")

    if request.method == 'GET':
        stop_nas()
    return Response(status=200, content_type="text/plain", param="finished")

@api_view(['GET'])
def status_request(request):
    print("_________GET /neck/status-request_____________")

    if request.method == 'GET':
        status = query_nas()
    return Response(status=200, content_type=status)

def status_report():
    try:
        url = 'http://0.0.0.0:8085/status_report'
        headers = {
            'Content-Type' : 'text/plain'
        }
        payload = {
            'container_id' : "neck_nas",
            'userid' : userid,
            'project_id' : projid
        }
        response = requests.get(url, params=payload)
        print(response.text)

    except ValueError as e:
        print(e)
