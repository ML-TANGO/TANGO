import requests
import shutil
import os 
import sys
import django
django.setup()
# import torch
import multiprocessing as mp
import yaml

# sys.path.append('../yolov5')

from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import HttpResponse
from pathlib import Path

# from yolov5.predict import docker_run
from . import models


PROCESSES = {}

task_to_model_table = {'detection': 'yolov7', 
                       'classification': 'resnet'}
model_to_size_table = {'yolov7':
                          {'cloud': '-e6e',
                            'T4': '-w6',
                            'Xavier': 'x',
                            'RKNN': '-tiny'
                           },
                       'resnet':
                          {'cloud': '101',
                           'T4': '50',
                           'Xavier': '24',
                           'RKNN': '18'
                          }
                        }

@api_view(['GET'])
def start(request):
    try:
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
    
        data_yaml, proj_info_yaml = get_user_requirements(userid, project_id)
        print(data_yaml, proj_info_yaml)
	        
        pr = mp.Process(target=task_to_model_mapping, args=(proj_info_yaml, userid, project_id), daemon=True)
        mp.set_start_method('spawn')
	         
        pr_id = get_process_id()
        PROCESSES[pr_id] = pr
        print(f'{len(PROCESSES)}-th process is starting')
        PROCESSES[pr_id].start()
    
        bmsinfo.proj_info_yaml=str(proj_info_yaml)
        bmsinfo.data_yaml=str(data_yaml)
        bmsinfo.status="started"
        
        bmsinfo.process_id = pr_id
        bmsinfo.save()
        return Response("started", status=200, content_type="text/plain")

    except Exception as e:
        print(e)


def task_to_model_mapping(yaml_path, userid, project_id):
    with open(yaml_path, 'r') as f:
        proj_info = yaml.load(f, Loader=yaml.FullLoader)
    task = proj_info['task_type']
    target = proj_info['target_info']
    model = task_to_model_table[task]
    proj_info['model_size'] = model_to_size_table[model][target]
    with open(yaml_path, 'w') as f:
        yaml.dump(proj_info, f, default_flow_style=False)
    status_report(userid, project_id, status="success")


@api_view(['GET'])
def get_ready_for_test(request):
    try:
        print("_________GET /get_ready_for_test_____________")
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
    
        sample_proj_yaml_cp(userid, project_id)
        create_data_yaml(userid, project_id)
        sample_data_cp()
    
        return Response("get ready for test", status=200, content_type="text/plain")
    except Exception as e:
        print(e)


def sample_proj_yaml_cp(userid, project_id):
    common_path = Path('/shared/common/')
    proj_path = common_path / userid / project_id
    if not os.path.exists(proj_path):
        Path(proj_path).mkdir(parents=True, exist_ok=True)
    shutil.copy('sample_yaml/project_info.yaml', '/shared/common/'+userid+'/'+project_id+'/')


def sample_data_cp():
    if not os.path.exists('/shared/datasets/'):
        Path('/shared/datasets/').mkdir(parents=True, exist_ok=True)
    shutil.copytree('sample_data/coco128',  Path('/shared/') / 'datasets' / 'coco128')


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
    
    with open(proj_path / 'dataset.yaml', 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

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
        pr_id = get_process_id()
                
        PROCESSES[pr_id] = pr
        print(f'{len(PROCESSES)}-th process is starting')
        PROCESSES[pr_id].start()
        
        print("does it come here\n")
        bmsinfo.proj_info_yaml=str(target_yaml)
        bmsinfo.data_yaml=str(data_yaml)
        bmsinfo.status="started"
        
        bmsinfo.process_id = pr_id
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
    proj_info_yaml_path = proj_path / 'project_info.yaml' # 'target.yaml'
    dataset_yaml_path = proj_path / 'datasets.yaml'

    return dataset_yaml_path, proj_info_yaml_path


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
        PROCESSES.pop(bmsinfo.process_id)
        
    except ValueError as e:
        print(e)

def queue_bms(userid, project_id):
    try:
        # docker_run(userid, project_id)        
        status_report(userid, project_id, status="success")
        print("process_bms ends")
    except ValueError as e:
        print(e)


def get_process_id():     # Assign Blank Process Number
    while True:
        pr_num = str(random.randint(10000, 99999))
        try:
            temp = PROCESSES[pr_num]
        except KeyError:
            break
    return pr_num

