import os
import json
import torch
import requests
import shutil
import multiprocessing
import yaml

from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.response import Response
from rest_framework.decorators import api_view
from pathlib import Path

from .yolov7_utils.train import run_yolo
from . import models


PROCESSES = []

def index(request):
    '''index'''
    return render(request, 'yoloe_core/index.html')


@api_view(['GET', 'POST'])
def InfoList(request):
    '''Information List for Neck NAS'''
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
        updatedInfo.save()

        return render(request, "yoloe_core/index.html")


@api_view(['GET'])
def start(request):
    print("_________GET /start_____________")
    params = request.query_params
    userid = params['user_id']
    project_id = params['project_id']

    # check user id & project id
    try:
        nasinfo = models.Info.objects.get(userid=userid,
                                          project_id=project_id)
    except models.Info.DoesNotExist:
        print("new user or project")
        nasinfo = models.Info(userid=userid,
                              project_id=project_id)

    if request.method == 'GET':
        data_yaml, target_yaml = get_user_requirements(userid, project_id)
        print(data_yaml, target_yaml)

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
        print("no such user or project...")
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
            print("tracked nas you want, but not running anymore")
            nasinfo.status = "stopped"
            nasinfo.save()
            return Response("stopped", status=200, content_type='text/plain')
    except models.Info.DoesNotExist:
        print("new user or project")
        nasinfo = models.Info(userid=userid,
                      project_id=project_id)
        nasinfo.status = "ready"
        nasinfo.save()
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
            'container_id' : "yolov7",
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
        print(f'report func: {threading.current_thread()}')
    except BaseException as e:
        print(e)


def process_nas(userid, project_id):
    try:
        common_root = Path('/shared/common/')
        proj_path = common_root / userid / project_id

        final_model = run_yolo(proj_path=proj_path, train_mode='search')
        print('process_nas: train done')

        best_pt_path = proj_path / 'model.pt'
        Path(proj_path).mkdir(parents=True, exist_ok=True)
        print(str(best_pt_path))
        shutil.copyfile(final_model, str(best_pt_path))
        print(f'saved the best model: {str(best_pt_path)}')

        exp_num = exp_num_check(proj_path)
        shutil.copy(proj_path / 'project_info.yaml', proj_path / 'exp' + str(exp_num) + '_project_info.yaml')

        status_report(userid, project_id, status="success")
        print("process_nas ends")
    except ValueError as e:
        print(e)


def exp_num_check(proj_path):
    current_filelist = os.listdir(proj_path)
    for filename in current_filelist:
        exp_num_list = []
        if 'exp' in filename[:3]:
            exp_num_list.append(int(filename.split('_')[0][2:]))
    if len(exp_num_list)==0:
        return 0
    else:
        return max(exp_num_list)+1


@api_view(['GET'])
def get_ready_for_test(request):
    print("_______GET /get_ready_for_test________")
    params = request.query_params
    userid = params['user_id']
    project_id = params['project_id']

    # check user id & project id
    try:
        nasinfo = models.Info.objects.get(userid=userid,
                                          project_id=project_id)
    except models.Info.DoesNotExist:
        print("new user or project")
        nasinfo = models.Info(userid=userid,
                              project_id=project_id)

    common_root = Path('/shared/common/')
    proj_path = common_root / userid / project_id
    if request.method == 'GET':
        Path(proj_path).mkdir(parents=True, exist_ok=True)
        shutil.copytree('yoloe_core/yolov7_utils/sample_yaml/coco128',  Path('/shared/') / 'datasets' / 'coco128')
        shutil.copy('yoloe_core/yolov7_utils/sample_yaml/hyp.scratch.p5.yaml', proj_path / 'hyp.scratch.p5.yaml')
    
        with open('yoloe_core/yolov7_utils/sample_yaml/args.yaml') as f:
            args_yaml = yaml.load(f, Loader=yaml.FullLoader)
        with open('yoloe_core/yolov7_utils/sample_yaml/coco.yaml') as f:
            data_yaml = yaml.load(f, Loader=yaml.FullLoader)
    
        args_yaml['cfg'] = 'yolov7x'
        args_yaml['data'] = str(proj_path / 'coco.yaml')
        args_yaml['hyp'] = str(proj_path / 'hyp.scratch.p5.yaml')
        data_yaml['train'] = str(Path('/shared/') / 'datasets' / 'coco128' / 'images' / 'train2017')
        data_yaml['test'] = str(Path('/shared/') / 'datasets' / 'coco128' / 'images' / 'train2017')
        data_yaml['val'] = str(Path('/shared/') / 'datasets' / 'coco128' / 'images' / 'train2017')
    
        with open(proj_path / 'args.yaml', 'w') as f:
            yaml.dump(args_yaml, f, default_flow_style=False)
        with open(proj_path / 'coco.yaml', 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        return Response('ready_for_v7_test', status=200, content_type='text/plain')


def make_directory(path_list):
    path = Path('')
    for path_temp in path_list:
        path = path / path_temp
        if not os.path.isdir(path):
            os.mkdir(path)
