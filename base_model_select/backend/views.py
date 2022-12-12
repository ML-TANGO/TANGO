from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import HttpResponse
from pathlib import Path
import requests

import torch
import torch.multiprocessing as mp

import sys
sys.path.append('../yolov5')

from yolov5.predict import docker_run
import django
django.setup()
from . import models

PROCESSES = []

@api_view(['GET'])
def start_api(request):
    print("_________GET /start_____________")
    params = request.query_params
    userid = params['user_id']
    project_id = params['project_id']
    print(userid, project_id) 
    
    try:
        bmsinfo = models.Info.objects.get(userid=userid,
                                        project_id=project_id)
    except models.Info.DoesNotExist:
        bmsinfo = models.Info(userid=userid, project_id=project_id)  
        print("new user or project")
        
        # if bmsinfo.status != "ready":
        #     print(f"existed user & project : {bmsinfo.status}... ignore this start signal")
        #     return Response("error", status=200, content_type="text/plain")
    
        # for i in models.Info.objects.all():
        #     if i.status != "ready":
        #         print("not allow runnnig one more bms at the same time..."
        #               " ignore this start signal")
        #         return Response("error", status=200, content_type="text/plain")
    

    if request.method == 'GET':
        
        data_yaml, target_yaml = get_user_requirements(userid, project_id)
        print(data_yaml, target_yaml)
                
        pr = mp.Process(target = queue_bms, args=(userid, project_id))
        mp.set_start_method('spawn')
                
        PROCESSES.append(pr)
        print(f'{len(PROCESSES)}-th process is starting')
        PROCESSES[-1].start()
        
        print("does it come here\n")
        bmsinfo.target_device=str(target_yaml)
        bmsinfo.data_yaml=str(data_yaml)
        bmsinfo.status="started"
        
        bmsinfo.process_id = len(PROCESSES)-1
        bmsinfo.save()
        return Response("started", status=200, content_type="text/plain")
        
        
            
@api_view(['GET'])
def stop_api(request):
    print("_________GET /stop_____________")
    params = request.query_params
    userid = params['user_id']
    project_id = params['project_id']
    try:
        bmsinfo = models.Info.objects.get(userid=userid, project_id=project_id)
    except models.Info.DoesNotExist:
        print("no such user or project...")
        return Response('failed', status=200, content_type='text/plain')

    PROCESSES[bmsinfo.process_id].terminate()
    PROCESSES.pop(bmsinfo.process_id)
    # bmsinfo.delete()
    bmsinfo.status = "stopped"
    bmsinfo.save()
    return Response("stopped", status=200, content_type="text/plain")

@api_view(['GET'])
def status_request(request):
    print("_________GET /status_request_____________")
    params = request.query_params
    userid = params['user_id']
    project_id = params['project_id']
    print(userid, project_id) 
    try:
        bmsinfo = models.Info.objects.get(userid=userid, project_id=project_id)
        # if THREADS[bmsinfo.thread_id].is_alive():
        print("process iD is", bmsinfo.process_id)
        if PROCESSES[bmsinfo.process_id].is_alive():
            print("found thread running nas")
            bmsinfo.status = "running"
            bmsinfo.save()
            return Response("running", status=200, content_type='text/plain')
        else:
            print("tracked bms you want, but not running anymore")
            bmsinfo.status = "stopped"
            bmsinfo.save()
            return Response("stopped", status=200, content_type='text/plain')
        
    except models.Info.DoesNotExist:
        # print("no such user or project...")
        # return Response('failed', status=200, content_type='text/plain')
        print("new user or project")
        bmsinfo = models.Info(userid=userid, project_id=project_id)
        bmsinfo.status = "ready"
        bmsinfo.save()
        return Response("ready", status=200, content_type='text/plain')

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
            'container_id' : "bms",
            'user_id' : userid,
            'project_id' : project_id,
            'status' : status
        }
        response = requests.get(url, headers=headers, params=payload)
        print(response.text)

        bmsinfo = models.Info.objects.get(userid=userid,
                                      project_id=project_id)
        bmsinfo.status = "ready"
        bmsinfo.save()
        PROCESSES.pop(-1)
        
    except ValueError as e:
        print(e)

def queue_bms(userid, project_id):
    try:
        docker_run(userid, project_id)        
        status_report(userid, project_id, status="success")
        print("process_bms ends")
    except ValueError as e:
        print(e)
