'''
views.py
'''

import os
import json
import shutil
import torch
import requests
import multiprocessing
import json
import yaml

from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.response import Response
from rest_framework.decorators import api_view
from pathlib import Path

from .net_generator.run_nas import run_nas

from . import models

PROCESSES = []
COMMON_ROOT = Path('/shared/common/')

def index(request):
    '''index'''
    return render(request, 'bnas/index.html')


@api_view(['GET', 'POST'])
def InfoList(request):
    '''Information List for Backbone NAS'''
    if request.method == 'POST':

        # Fetching the form data
        uploadedFile = request.FILES["data_yaml"]
        usrId = request.data['user_id']
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
        # updatedInfo = models.Info(
        #     userid="jgp",
        #     project_id="test1")
        updatedInfo.save()

        return render(request, "bnas/index.html")

@api_view(['GET'])
def start(request):
    params = request.query_params
    userid = params['user_id']
    project_id = params['project_id']
    
    # check user id & project id
    try:
        nasinfo = models.Info.objects.get(userid=userid,
                                          project_id=project_id)
    except models.Info.DoesNotExist:
        nasinfo = models.Info(userid=userid,
                              project_id=project_id)
    
    if request.method == 'GET':
        data_yaml, target_yaml = get_user_requirements(userid, project_id)
        # print(data_yaml, target_yaml)

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
    print("_________GET /stop_____________")
    params = request.query_params
    userid = params['user_id']
    project_id = params['project_id']
    try:
        nasinfo = models.Info.objects.get(userid=userid,
                                          project_id=project_id)
    except models.Info.DoesNotExist:
        return Response('failed', status=200, content_type='text/plain')

    PROCESSES[nasinfo.process_id].terminate()
    PROCESSES.pop(nasinfo.process_id)
    nasinfo.status = "stopped"
    nasinfo.save()

    return Response("stopped", status=200, content_type="text/plain")

@api_view(['GET'])
def status_request(request):
    print("_________GET /status_request_____________")
    params = request.query_params
    userid = params['user_id']
    project_id = params['project_id']
    try:
        nasinfo = models.Info.objects.get(userid=userid,
                                          project_id=project_id)
        if PROCESSES[nasinfo.process_id].is_alive():
            print("found thread running nas")
            nasinfo.status = "running"
            nasinfo.save()
            return Response("running", status=200, content_type='text/plain')
        else:
            nasinfo.status = "stopped"
            nasinfo.save()
            return Response("stopped", status=200, content_type='text/plain')
    except models.Info.DoesNotExist:
        nasinfo = models.Info(userid=userid,
                      project_id=project_id)
        nasinfo.status = "ready"
        nasinfo.save()
        return Response("ready", status=200, content_type='text/plain')

def get_user_requirements(userid, projid):
    common_root = Path('/shared/common/')
    proj_path = common_root / userid / projid
    target_yaml_path = proj_path / 'project_info.yaml' # 'target.yaml'
    dataset_yaml_path = proj_path / 'dataset.yaml'
    return dataset_yaml_path, target_yaml_path

def process_nas(userid, project_id):
    proj_path = COMMON_ROOT / userid / project_id
    dataset_yaml_path = proj_path / 'dataset.yaml'
    final_pt_path = proj_path / 'best.pt'
    final_info_path = proj_path / 'neural_net_info.yaml'
    final_py_path = proj_path / 'model.py'
    pt = run_nas(dataset_yaml_path)
    torch.save(pt.state_dict(), final_pt_path)
    shutil.copy("bnas/media/temp_files/model/model.py", final_py_path)
    create_bb_info(
                final_info_path,
                final_pt_path,
                final_py_path,
                pt.arch)

    status_report(userid, project_id, status="success")
    
    # folder = os.getcwd()
    # print('folder: %s' % folder)
    # for filename in os.listdir(folder):
    #     print(filename)

def status_report(userid, project_id, status="success"):
    try:
        url = 'http://0.0.0.0:8085/status_report'
        headers = {
            'Content-Type' : 'text/plain'
        }
        payload = {
            'container_id' : "backbone_nas",
            'user_id' : userid,
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
    except ValueError as e:
        print(e)

# for Unit Test
@api_view(['GET'])
def create_net(request):
    '''create_net'''
    userid = 'user_id'
    project_id = 'project_id'

    proj_path = COMMON_ROOT / userid / project_id
    dataset_yaml_path = proj_path / 'dataset.yaml'
    final_pt_path = proj_path / 'best.pt'
    final_info_path = proj_path / 'neural_net_info.yaml'
    final_py_path = proj_path / 'model.py'

    params = request.query_params
    if params['GPUs'] == '1':
        return render(request, "bnas/index.html",
                    context={"path": "Multi-GPUs is not yet supported."})
    if params['NASset'] == 'auto':
        batch_size = -1
        max_latency = 25
        pop_size = 4
        niter = 5
    else:
        batch_size = int(params["bs"])
        max_latency = int(params["mlatency"])
        pop_size = int(params["popsize"])
        niter = int(params["niter"])

    pt = run_nas(
        dataset_yaml_path,
        batch_size=batch_size,
        max_latency=max_latency,
        pop_size = pop_size,
        niter=niter)
    pt.supernet = pt.supernet.get_active_subnet()
    
    torch.save(pt.state_dict(), final_pt_path)
    shutil.copy("bnas/media/temp_files/model/model.py", final_py_path)
    create_bb_info(
                final_info_path,
                final_pt_path,
                final_py_path,
                pt.arch)

    return render(request, "bnas/index.html",
                    context={"path": str(pt.arch)})

    # return render(request, "bnas/index.html",
    #                 context={"path": final_pt_path})

def create_bb_info(
                final_info_path,
                final_pt_path,
                final_py_path,
                arch):
    nn_info = {
        "target device": "Android S10",
        "Application": "Object Detection",
        'class_file': str(final_py_path),
        'class_name': "BestModel()",
        "architecture": json.dumps(arch),
        'weight_file': str(final_pt_path),
        "input_shape": str([1, 3, 224, 224]),
    }

    with open(final_info_path, 'w') as file:
        yaml.dump(nn_info, file, default_flow_style=False)