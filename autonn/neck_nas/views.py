'''
views.py
'''

import os
import json
import torch
import requests
# import threading
import multiprocessing

from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.response import Response
from rest_framework.decorators import api_view
from pathlib import Path
from .ku.main import run_nas

# from rest_framework import viewsets
# from .serializers import InfoSerializer
from . import models

# THREADS = []
PROCESSES = []

def index(request):
    '''index'''
    return render(request, 'neckNAS/index.html')


@api_view(['GET', 'POST'])
def InfoList(request):
    '''Information List for Neck NAS'''
    if request.method == 'POST':

        # Fetching the form data
        uploadedFile = request.FILES["data_yaml"]
        usrId = request.data['userid']
        prjId = request.data['project_id']
        target = request.data['target']
        task = request.data['task']
        sts = request.data['status']
        prcId = request.data['process_id']

        # Saving the information in the database
        updatedInfo = models.Info(
            userid=usrId,
            project_id=prjId,
            target_device=target,
            data_yaml=uploadedFile,
            task=task,
            status=sts,
            process_id=prcId
        )
        updatedInfo.save()

        return render(request, "neckNAS/index.html")


@api_view(['GET'])
def start(request):
    print("_________GET /neck/start_____________")
    params = request.query_params
    userid = params['userid']
    project_id = params['project_id']

    # check user id & project id
    try:
        nasinfo = models.Info.objects.get(userid=userid,
                                          project_id=project_id)
        # if nasinfo.status != "ready":
        #     print(f"existed user & project : {nasinfo.status}... ignore this start signal")
        #     return Response("error", status=200, content_type="text/plain")
    except models.Info.DoesNotExist:
        print("new user or project")
        # for i in models.Info.objects.all():
        #     if i.status != "ready":
        #         print("not allow runnnig one more nas at the same time..."
        #               " ignore this start signal")
        #         return Response("error", status=200, content_type="text/plain")
        nasinfo = models.Info(userid=userid,
                              project_id=project_id)

    if request.method == 'GET':
        data_yaml, target_yaml = get_user_requirements(userid, project_id)
        print(data_yaml, target_yaml)
        # run_nas(str(data_yaml), str(target_yaml))
        # th = threading.Thread(target = run_nas, args=(ev))
        # THREADS.append(th)
        # THREADS[-1].start()

        pr = multiprocessing.Process(target = process_nas, args=(userid, project_id))
        PROCESSES.append(pr)
        print(f'{len(PROCESSES)}-th process is starting')
        PROCESSES[-1].start()

        nasinfo.target_device=str(target_yaml)
        nasinfo.data_yaml=str(data_yaml)
        nasinfo.status="started"
        # nasinfo.thread_id = len(THREADS)-1
        nasinfo.process_id = len(PROCESSES)-1
        nasinfo.save()
        return Response("started", status=200, content_type="text/plain")


@api_view(['GET'])
def stop(request):
    print("_________GET /neck/stop_____________")
    params = request.query_params
    userid = params['userid']
    project_id = params['project_id']
    try:
        nasinfo = models.Info.objects.get(userid=userid,
                                          project_id=project_id)
    except models.Info.DoesNotExist:
        print("no such user or project...")
        return Response('failed', status=200, content_type='text/plain')

    PROCESSES[nasinfo.process_id].terminate()
    PROCESSES.pop(nasinfo.process_id)
    # nasinfo.delete()
    nasinfo.status = "stopped"
    nasinfo.save()

    return Response("stopped", status=200, content_type="text/plain")


@api_view(['GET'])
def status_request(request):
    print("_________GET /neck/status-request_____________")
    params = request.query_params
    userid = params['userid']
    project_id = params['project_id']
    try:
        nasinfo = models.Info.objects.get(userid=userid,
                                          project_id=project_id)
    except models.Info.DoesNotExist:
        print("no such user or project...")
        return Response('failed', status=200, content_type='text/plain')
    # if THREADS[nasinfo.thread_id].is_alive():
    if PROCESSES[nasinfo.process_id].is_alive():
        print("found thread running nas")
        nasinfo.status = "running"
        nasinfo.save()
        return Response('running', status=200, content_type='text/plain')
    else:
        print("tracked nas you want, but not running anymore")
        nasinfo.status = "stopped"
        nasinfo.save()
        return Response('stopped', status=200, content_type='text/plain')


def get_user_requirements(userid, projid):
    common_root = Path('/shared/common/')
    proj_path = common_root / userid / projid
    target_yaml_path = proj_path / 'project_info.yaml' # 'target.yaml'
    dataset_yaml_path = proj_path / 'datasets.yaml'
    return dataset_yaml_path, target_yaml_path


def status_report(userid, project_id, status="success"):
    try:
        url = 'http://0.0.0.0:8085/status_report'
        headers = {
            'Content-Type' : 'text/plain'
        }
        payload = {
            'container_id' : "neck_nas",
            'userid' : userid,
            'project_id' : project_id,
            'status' : status
        }
        response = requests.get(url, headers=headers, params=payload)
        print(response.text)

        nasinfo = models.Info.objects.get(userid=userid,
                                      project_id=project_id)
        nasinfo.status = "ready"
        nasinfo.save()
        PROCESSES.pop(-1)
        print(f'report func: {threading.current_thread()}')
    except ValueError as e:
        print(e)


def process_nas(userid, project_id):
    try:
        run_nas()
        status_report(userid, project_id, status="success")
        print("process_nas ends")
    except e:
        print(e)



