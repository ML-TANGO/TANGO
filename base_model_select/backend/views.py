import requests
import shutil
import os
import sys
import django
django.setup()
import multiprocessing as mp
import yaml
import random

from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import HttpResponse
from pathlib import Path

from . import models
from batch_test.batch_size_test import run_batch_test


PROCESSES = {}

task_to_model_table = {"detection": "yolov7", "classification": "resnet"}
model_to_size_table = {
    "yolov7": {
        "cloud": "-tiny",
        "k8s": "-tiny",
        "k8sjetsonnano": "-tiny",
        "pcweb": "-tiny",
        "pc": "-tiny",
        "jetsonagxorin": "-tiny",
        "jetsonagxxavier": "-tiny",
        "jetsonnano": "-tiny",
        "galaxys22": "-tiny",
        "odroidn2": "-tiny",
    },
    "resnet": {
        "cloud": "20",
        "k8s": "20",
        "k8sjetsonnano": "20",
        "pcweb": "20",
        "pc": "20",
        "jetsonagxorin": "20",
        "jetsonagxxavier": "20",
        "jetsonnano": "20",
        "galaxys22": "20",
        "odroidn2": "20",
    },
}


@api_view(['GET'])
def start(request):
    print("_________GET /start_____________")
    params = request.query_params
    userid = params['user_id']
    project_id = params['project_id']
    print(userid, project_id)

    try:
        bmsinfo = models.Info.objects.get(userid=userid, project_id=project_id)
    except models.Info.DoesNotExist:
        bmsinfo = models.Info(userid=userid, project_id=project_id)
        print("new user or project")

    try:
        proj_info_yaml = get_user_requirements(userid, project_id)

        pr = mp.Process(target=bms_process, args=(proj_info_yaml, userid, project_id), daemon=True)

        pr_id = get_process_id()
        PROCESSES[pr_id] = pr
        PROCESSES[pr_id].start()
        print(f'{len(PROCESSES)}-th process is starting')

        bmsinfo.proj_info_yaml=str(proj_info_yaml)
        bmsinfo.status="started"

        bmsinfo.process_id = pr_id
        bmsinfo.save()
        return Response("started", status=200, content_type="text/plain")
    except Exception as e_message:
        print(e_message)

        bmsinfo.status = "failed"
        bmsinfo.save()
        return Response("failed", status=200, content_type="text/plain")


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

    try:
        PROCESSES[str(bmsinfo.process_id)].terminate()
        bmsinfo.status = "stopped"
        bmsinfo.save()
        return Response("stopped", status=200, content_type="text/plain")
    except:
        return Response("failed", status=200, content_type="text/plain")


@api_view(['GET'])
def status_request(request):
    print("_________GET /status_request_____________")
    params = request.query_params
    userid = params['user_id']
    project_id = params['project_id']
    print(userid, project_id)

    try:
        bmsinfo = models.Info.objects.get(userid=userid, project_id=project_id)
        print("process iD is", bmsinfo.process_id)
    except models.Info.DoesNotExist:
        print("new user or project")
        bmsinfo = models.Info(userid=userid, project_id=project_id)
        bmsinfo.status = "ready"
        bmsinfo.save()
        return Response("ready", status=200, content_type='text/plain')

    try:
        if PROCESSES[str(bmsinfo.process_id)].is_alive():
            print("found thread running nas")
            bmsinfo.status = "running"
            bmsinfo.save()
            return Response("running", status=200, content_type='text/plain')
        else:
            print("tracked bms you want, but not running anymore")
            if bmsinfo.status in ["started", "running"]:
                bmsinfo.status = "failed"
                bmsinfo.save()
            print(f"bmsinfo.status: {bmsinfo.status}")
            return Response(bmsinfo.status, status=200, content_type='text/plain')
    except:
        if bmsinfo.status in ["started", "running"]:
            bmsinfo.status = "failed"
            bmsinfo.save()
        return Response(bmsinfo.status, status=200, content_type='text/plain')


def status_report(userid, project_id, status="success"):
    try:
        url = 'http://projectmanager:8085/status_report'
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

        bmsinfo = models.Info.objects.get(userid=userid, project_id=project_id)
        bmsinfo.status = "completed"
        bmsinfo.save()

        print(f"[ STATUS REPORT ] USER ID: {userid}, PROJECT ID: {project_id}")

    except Exception as e:
        print("An error occurs when BMS reports a project status")
        print(e)

   
def bms_process(yaml_path, userid, project_id):
    with open(yaml_path, 'r') as f:
        proj_info_dict = yaml.load(f, Loader=yaml.FullLoader)

    basemodel_yaml = create_basemodel_yaml(yaml_path, userid, project_id)

    with open(basemodel_yaml, 'r') as f:
        basemodel_dict = yaml.load(f, Loader=yaml.FullLoader)
    batch_size = run_batch_test(basemodel_yaml, 
                                proj_info_dict['task_type'],
                                basemodel_dict['imgsz'] if proj_info_dict['task_type']=='detction' else 256, 
                                f"hyperparam_yaml/yolov7/hyp.scratch.{basemodel_dict['hyp']}.yaml" if proj_info_dict['task_type']=='detection' else None, 
                                True if proj_info_dict['target_info']=='Galaxy_S22' else False,
                                )

    if batch_size == False:
        bmsinfo = models.Info.objects.get(userid=userid, project_id=project_id)
        bmsinfo.status = "failed"
        bmsinfo.save()
        print(f'[ BMS ] Memory resource is not enough for training. Please shrink Model Size')
    else:
        with open(yaml_path, 'r') as f:
            project_info_dict = yaml.load(f, Loader=yaml.FullLoader)
        project_info_dict['batchsize'] = batch_size
        with open(yaml_path, 'w') as f:
            yaml.dump(project_info_dict, f, default_flow_style=False)

        status_report(userid, project_id, status="success")


def create_basemodel_yaml(yaml_path, userid, project_id):
    with open(yaml_path, 'r') as f:
        proj_info = yaml.load(f, Loader=yaml.FullLoader)
    task = proj_info['task_type']
    target = proj_info['target_info'].replace('-', '').replace('_', '').lower()

    model = task_to_model_table[task]
    model_size = model_to_size_table[model][target]

    source_path = f'basemodel_yaml/{model}/{model}{model_size}.yaml'
    target_path = f'/shared/common/{userid}/{project_id}/basemodel.yaml'
    shutil.copy(source_path, target_path)
    
    if task == 'classification':
        source_path = f'basemodel_yaml/{model}/resnet50.json'
        target_path = f'/shared/common/{userid}/{project_id}/basemodel.json'
        shutil.copy(source_path, target_path)
        print('[ BMS ] JSON file is also created in addition to yaml file.')
    else:
        print('[ BMS ] The type of task is not classification. JSON file will not be created.')

    return target_path


def get_user_requirements(userid, projid):
    common_root = Path('/shared/common/')
    proj_path = common_root / userid / projid
    proj_info_yaml_path = proj_path / 'project_info.yaml' # 'target.yaml'

    return proj_info_yaml_path


def get_process_id():     # Assign Blank Process Number
    while True:
        pr_num = str(random.randint(10000, 99999))
        try:
            temp = PROCESSES[pr_num]
        except KeyError:
            break
    return pr_num


@api_view(['GET'])
def get_ready_for_test(request):
    print("_________GET /get_ready_for_test_____________")
    params = request.query_params
    userid = params['user_id']
    project_id = params['project_id']
    print(userid, project_id)

    try:
        bmsinfo = models.Info.objects.get(userid=userid, project_id=project_id)
    except models.Info.DoesNotExist:
        bmsinfo = models.Info(userid=userid, project_id=project_id)
        print("new user or project")

    sample_proj_yaml_cp(userid, project_id)
    create_data_yaml(userid, project_id)
    sample_data_cp()

    return Response("get ready for test", status=200, content_type="text/plain")


def sample_proj_yaml_cp(userid, project_id):
    common_path = Path('/shared/common/')
    proj_path = common_path / userid / project_id
    if not os.path.exists(proj_path):
        Path(proj_path).mkdir(parents=True, exist_ok=True)
    shutil.copy('sample_yaml/project_info.yaml', '/shared/common/'+userid+'/'+project_id+'/')


def sample_data_cp():
    if not os.path.exists('/shared/datasets/'):
        Path('/shared/datasets/').mkdir(parents=True, exist_ok=True)
    shutil.copytree('sample_data/coco128',  Path('/shared/') / 'datasets' / 'coco')


def create_data_yaml(userid, project_id):
    common_path = Path('/shared/common/')
    proj_path = common_path / userid / project_id
    if not os.path.exists('/shared/datasets/'):
        Path('/shared/datasets/').mkdir(parents=True, exist_ok=True)

    with open('sample_yaml/dataset.yaml') as f:
        data_yaml = yaml.load(f, Loader=yaml.FullLoader)

    data_yaml['train'] = str(Path('/shared/') / 'datasets' / 'coco128' / 'images' / 'train2017')
    data_yaml['test'] = str(Path('/shared/') / 'datasets' / 'coco128' / 'images' / 'train2017')
    data_yaml['val'] = str(Path('/shared/') / 'datasets' / 'coco128' / 'images' / 'train2017')

    with open(Path('/shared/datasets/coco/') / 'dataset.yaml', 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)


@api_view(['GET'])
def start_api(request):
    print("_________GET /start_____________")
    params = request.query_params
    userid = params['user_id']
    project_id = params['project_id']
    print(userid, project_id)

    try:
        bmsinfo = models.Info.objects.get(userid=userid, project_id=project_id)
    except models.Info.DoesNotExist:
        bmsinfo = models.Info(userid=userid, project_id=project_id)
        print("new user or project")

    if request.method == 'GET':

        target_yaml = get_user_requirements(userid, project_id)

        pr = mp.Process(target = queue_bms, args=(userid, project_id))
        pr_id = get_process_id()

        PROCESSES[pr_id] = pr
        print(f'{len(PROCESSES)}-th process is starting')
        PROCESSES[pr_id].start()

        print("does it come here\n")
        bmsinfo.proj_info_yaml=str(target_yaml)
        bmsinfo.status="started"

        bmsinfo.process_id = pr_id
        bmsinfo.save()
        return Response("started", status=200, content_type="text/plain")


def queue_bms(userid, project_id):
    try:
        # docker_run(userid, project_id)
        status_report(userid, project_id, status="success")
        print("process_bms ends")
    except ValueError as e:
        print(e)
