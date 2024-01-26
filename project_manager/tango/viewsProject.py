"""viewsProejct module for tango
This module for vies.
Attributes:

Todo:

"""

import os
import json
import random
import math
import socket
import threading
import requests
import asyncio

from datetime import datetime
import time

import django.middleware.csrf
from django.http import HttpResponse
from rest_framework.permissions import AllowAny

from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from oauth2_provider.contrib.rest_framework import OAuth2Authentication

from .models import Project, AuthUser, Target, WorkflowOrder

from django.views.decorators.csrf import csrf_exempt, csrf_protect
from django.forms.models import model_to_dict

from .projectHandler import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(os.path.dirname(BASE_DIR))

# @permission_classes([IsAuthenticated])                  # 권한 체크 - 로그인 여부
# @authentication_classes([JSONWebTokenAuthentication])   # 토큰 확인
# @permission_classes([AllowAny])


# 컨테이너 상태 결과 응답
@api_view(['POST'])
@permission_classes([AllowAny])   # 토큰 확인
def container_start(request):

    try:
        print("----------container_start----------")
        user_id = request.data['user_id']
        project_id = request.data['project_id']
        container_id = request.data['container_id']

        project_info = Project.objects.get(id=project_id, create_user=str(user_id))

        response = asyncio.run(start_handler(container_id, user_id, project_id, project_info.target.target_info))
        to_json = json.loads(response)

        project_info.container = container_id
        project_info.container_status = 'started'
        project_info.save()

        return HttpResponse(json.dumps({'status': 200, 'message': str(container_id) + ' 시작 요청\n', 'response' : to_json['request_info']}))

    except Exception as error:
        print('container start error - ' + str(error))
        print(error)
        return HttpResponse(error)


# 컨테이너 상태 결과 응답
@api_view(['POST'])
@permission_classes([AllowAny])   # 토큰 확인
def status_request(request):

    try:
        print("----------status_request----------")
        response_log = ""

        user_id = request.data['user_id']
        project_id = request.data['project_id']

        queryset = Project.objects.get(id=project_id, create_user=str(user_id))
        container_id = queryset.container
        
        res = asyncio.run(request_handler(container_id, user_id, project_id, queryset.target.target_info))
        if res == None:
            return HttpResponse(json.dumps({'container': container_id, 'container_status': '', 'message': ''}))

        response = json.loads(res)
        log_str = str(queryset.current_log) + str(container_id) + '- status_request response : ' + str(response['response'])
        # log_str = str(queryset.current_log) 
        # log_str += response['request_info']
        response_log = log_str

        if len(response['response']) > 50:
            queryset.save()
            return HttpResponse(json.dumps({'container': container_id, 'container_status': queryset.container_status, 'message':  get_log_container_name(container_id) + ": status_request - Error\n"}))

        # status_report에서 completed 였을 때를 제외하고
        # if queryset.container_status != 'completed':
        #     queryset.container_status = response['response']

        ## 새로운 컨테이너에서 로그를 불러올때
        # 컨테이너가 실행될때는 last_logs_timestamp 이후에 실행 되니 주석 처리
        # if queryset.last_log_container != queryset.container:
        #     queryset.last_logs_timestamp = 0
        
        if container_id != "imagedeploy":
            logs = get_docker_log_handler(queryset.container, queryset.last_logs_timestamp)
        else:
            logs = get_docker_log_handler(queryset.target.target_info, queryset.last_logs_timestamp)
        
        queryset.last_logs_timestamp = time.mktime(datetime.now().timetuple()) + 1.0
        queryset.last_log_container = queryset.container


        response_log += '\n' + str(logs)
        queryset.current_log = ''

        if queryset.container_status == 'completed':
            response_log += get_log_container_name(container_id) + " 완료\n"
            response['response'] = "completed"

        if response['response'] == 'completed':
            queryset.container_status = 'completed'

        update_project_log_file(user_id, project_id, response_log)

        queryset.save()
        return HttpResponse(json.dumps({'container': container_id, 'container_status': response['response'], 'message': response_log,}))

    except Exception as error:
        print("status_request --- error")
        print(error)
        return HttpResponse(error)


# 컨테이너 상태 결과 응답
@api_view(['GET'])
@permission_classes([AllowAny])   # 토큰 확인
def status_report(request):

    try:
        print("@@@@@@@@@@@@@@@@@@@@@@@ status report @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        user_id = request.GET['user_id']
        project_id = request.GET['project_id']
        container_id = db_container_name(request.GET['container_id'])
        result = request.GET['status']

        headers = ''
        for header, value in request.META.items():
            if not header.startswith('HTTP'):
                continue
            header = '-'.join([h.capitalize() for h in header[5:].lower().split('_')])
            headers += '{}: {}\n'.format(header, value)
        queryset = Project.objects.get(id=project_id, create_user=str(user_id))
        queryset.container = container_id
        log_str =  str(queryset.current_log) 
        log_str += '---------------- Status Report ----------------'
        log_str += "\n" + get_log_container_name(container_id) + " --> Project Manager"
        log_str += "\n" + str(request)
        log_str += "\n" + "method : " + request.method
        log_str += "\n" + headers
        log_str += '---------------- Params ----------------'
        log_str += '\nuser_id : '+ str(user_id)
        log_str += '\nproject_id : '+ str(project_id)
        log_str += '\ncontainer_id : '+ str(container_id)
        log_str += '\nstatus : '+ str(result)
        log_str += '\n----------------------------------------'
        log_str += '\n\n'

        queryset.current_log = log_str

        workflow_order = WorkflowOrder.objects.filter(project_id=project_id).order_by('order')

        if queryset.project_type == 'auto':
            current_container_idx = findIndexByDicList(list(workflow_order.values()), 'workflow_name', container_id)
            if (result == 'success' or result == 'completed') and current_container_idx != None :
                if len(list(workflow_order.values())) - 1 > current_container_idx:
                    next_container = list(workflow_order.values())[current_container_idx + 1]['workflow_name']
                    if next_container:
                        queryset.container = next_container
                        log = str(queryset.current_log) + "\n" + get_log_container_name(container_id) + " 완료"
                        log += "\n" + get_log_container_name(next_container) + " 시작 요청"
                        queryset.current_log = log
                        asyncio.run(start_handler(next_container, user_id, project_id, queryset.target.target_info))
                        queryset.container_status = 'started'
        else:
            if result == 'success' or result == 'completed':
                queryset.container_status = 'completed'

        queryset.save()
        return HttpResponse(json.dumps({'status': 200}))

    except Exception as error:
        print("status_report - error")
        print(error)
        return HttpResponse(error)


# 컨테이너 상태
@api_view(['GET', 'POST'])
@authentication_classes([OAuth2Authentication])   # 토큰 확인
def status_result(request):
    """
    status_result _summary_

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """

    try:
        project_id = request.data['project_id']
        queryset = Project.objects.get(id=project_id, create_user=str(request.user))
        if last_container != queryset.container:
            last_logs_timestamp = 0

        logs = get_docker_log_handler(queryset.container, last_logs_timestamp)
        last_logs_timestamp = time.mktime(datetime.now().timetuple())
        last_container = queryset.container

        m = response_message + '\n' + str(logs)
        response_message = ''

        return HttpResponse(json.dumps({'container': queryset.container,
                                        'container_status': queryset.container_status,
                                        'message': m,}))

    except Exception as e:
        print(e)


# nn_model 다운로드(외부IDE연동)
@api_view(['GET'])
@permission_classes([AllowAny])   # 토큰 확인
def download_nn_model(request):

    try:
        user_id = request.GET['user_id']
        project_id = request.GET['project_id']
        zip_file_path = nn_model_zip(user_id, project_id)
        return HttpResponse(open(zip_file_path, 'rb').read())

    except Exception as error:
        print("download_nn_model - error")
        print(error)
        return HttpResponse(error)

# nn_model 업로드(외부IDE연동)
@api_view(['POST'])
@permission_classes([AllowAny])   # 토큰 확인
def upload_nn_model(request):

    try:
        user_id = request.data['user_id']
        project_id = request.data['project_id']
        nn_model = request.data['nn_model']
        print("upload_nn_model")
        print(user_id)
        print(project_id)
        print(nn_model)
        print(type(nn_model))
        nn_model_unzip(user_id, project_id, nn_model)

        return Response(status=200)
    except Exception as error:
        print("upload_nn_model - error")
        print(error)
        return HttpResponse(error)


# Project 리스트 요청
@api_view(['GET', 'POST'])
@authentication_classes([OAuth2Authentication])   # 토큰 확인
def project_list_get(request):
    """
    project_list_get _summary_

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """

    try:
        queryset = Project.objects.filter(create_user=str(request.user))
        data = list(queryset.values())

        print("project_list_info")
        print(data)
       
        for project in data: 
            if project['target_id'] is not None:
                project['target_info'] = model_to_dict(Target.objects.get(id=int(project['target_id'])))

        return HttpResponse(json.dumps(data))

    except Exception as e:
        print(e)


# Project 이름 수정
@api_view(['GET', 'POST'])
@authentication_classes([OAuth2Authentication])   # 토큰 확인
def project_rename(request):
    """
    project_rename _summary_

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """

    try:
        form = request.data

        print(form)

        duplicate_check = Project.objects.get(project_name=form['name'],
                                              create_user=request.user)
    except Exception as e:
        print(e)
        duplicate_check = None

    # Project 이름 변경 - 중복 Project 이름이 없는 경우
    if duplicate_check is None:
        data = Project.objects.get(id=form['id'],
                                   create_user=request.user)
        data.project_name = form['name']
        data.save()

        return Response({'result': True})
    else:
        return Response({'result': False})


# Project 설명 수정
@api_view(['GET', 'POST'])
@authentication_classes([OAuth2Authentication])   # 토큰 확인
def project_description_update(request):
    """
    project_description_update _summary_

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """

    try:
        form = request.data

        data = Project.objects.get(id=form['id'],
                                   create_user=request.user)
        data.project_description = form['description']
        data.save()

        return Response(status=200)

    except Exception as e:
        print(e)
        return Response(status=500)
    
# Project 워크플로우 진행 방식 수정
@api_view(['GET', 'POST'])
@authentication_classes([OAuth2Authentication])   # 토큰 확인
def project_type_update(request):
    """
    project_type_update _summary_

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """

    try:
        form = request.data

        data = Project.objects.get(id=form['id'],
                                   create_user=request.user)
        projectType = form['type']
        data.project_type = projectType.lower()
        data.save()

        print(data.project_type)
        print(projectType.lower())

        return Response(status=200)

    except Exception as e:
        print(e)
        return Response(status=500)


# Project 생성
@api_view(['GET', 'POST'])
@authentication_classes([OAuth2Authentication])   # 토큰 확인
def project_create(request):
    """
    project_create _summary_

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Project 생성 - 기존 Project 이름 중복 검사
    try:
        duplicate_check = Project.objects.get(project_name=request.data['project_name'],
                                              create_user=request.user)

    except Exception as e:
        print(e)
        duplicate_check = None

    # Project 생성 - 중복 Project 이름이 없는 경우
    if duplicate_check is None:
        data = Project(project_name=request.data['project_name'],
                       project_description=request.data['project_description'],
                       create_user=request.user,
                       create_date=str(datetime.now()))
        data.save()

        return Response({'result': True,
                         'id': data.id,
                         'name': data.project_name,
                         'description': data.project_description})
    else:
        return Response({'result': False})


# Project 삭제
@api_view(['GET', 'POST'])
@authentication_classes([OAuth2Authentication])   # 토큰 확인
def project_delete(request):
    """
    project_delete _summary_

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """

    queryset = Project.objects.get(id=request.data['id'],
                                   create_user=request.user)  # Project id로 검색
    queryset.delete()

    return Response(status=200)


# Project 정보 조회
@api_view(['GET', 'POST'])
@authentication_classes([OAuth2Authentication])   # 토큰 확인
def project_info(request):
    """
    project_info _summary_

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """
    try:
        result = None

        queryset = Project.objects.filter(id=request.data['id'],
                                        create_user=request.user)  # Project id로 검색
        
        workflow_order = WorkflowOrder.objects.filter(project_id=request.data['id']).order_by('order')
        workflow_dic = {"workflow": list(workflow_order.values())}

        data = list(queryset.values())

        project = dict(data[0], **workflow_dic)


        # TODO : 타겟이 0이 아닌 경우 SW 정보 전달
        if project['target_id'] is not None:
            target_info = model_to_dict(Target.objects.get(id=int(project['target_id'])))
            target_info_dic = {"target_info": target_info}

            result = dict(project,  **target_info_dic)
        else:
            # 딕셔너리 정보 합치기
            result = dict(project)

        return Response(result)
    except Exception as e:
        print('error - project_info-=============')
        print(e)


# Project 업데이트
@api_view(['GET', 'POST'])
@authentication_classes([OAuth2Authentication])   # 토큰 확인
def project_update(request):
    """
    project_update _summary_

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """

    print('project_update')
    try:

        print(request.data)
        
        target = ''

        try:
            target = int(request.data['project_target']) 
        except:
            target = ''
        
        dataset = str(request.data['project_dataset'])

        task_type = str(request.data['task_type'])
        # task_type = 'detection'
        autonn_dataset_file = str(request.data['autonn_dataset_file'])
        autonn_basemodel = str(request.data['autonn_base_model'])
        nas_type = str(request.data['nas_type'])
        deploy_weight_level = str(request.data['deploy_weight_level'])
        deploy_precision_level = str(request.data['deploy_precision_level'])
        deploy_processing_lib = str(request.data['deploy_processing_lib'])
        deploy_user_edit = str(request.data['deploy_user_edit'])
        deploy_input_method = str(request.data['deploy_input_method'])
        deploy_input_data_path = str(request.data['deploy_input_data_path'])
        deploy_output_method = str(request.data['deploy_output_method'])

        deploy_input_source = str(request.data['deploy_input_source'])

        queryset = Project.objects.get(id=request.data['project_id'],
                                       create_user=request.user)

        print('queryset')
        print(queryset)

        if target == '':
            # queryset.target = target
            queryset.dataset = dataset
            queryset.task_type = task_type
            queryset.autonn_dataset_file = autonn_dataset_file
            queryset.autonn_basemodel = autonn_basemodel
            queryset.nas_type = nas_type
            queryset.deploy_weight_level = deploy_weight_level
            queryset.deploy_precision_level = deploy_precision_level
            queryset.deploy_processing_lib = deploy_processing_lib
            queryset.deploy_user_edit = deploy_user_edit
            queryset.deploy_input_method = deploy_input_method
            queryset.deploy_input_data_path = deploy_input_data_path
            queryset.deploy_output_method = deploy_output_method
            queryset.deploy_input_source = deploy_input_source
            queryset.container = ''
            queryset.container_status = ''

            queryset.save()
            return Response(status=200)

        # 타겟 정보 수신
        data = Target.objects.get(id=int(target))
        print("target - data")
        print(data.target_info)
        # if data.target_info != 'ondevice':

        #     queryset.target = Target.objects.get(id=int(target))
        #     queryset.dataset = dataset
        #     queryset.task_type = task_type
        #     queryset.autonn_dataset_file = autonn_dataset_file
        #     queryset.autonn_basemodel = autonn_basemodel
        #     queryset.nas_type = nas_type
        #     queryset.deploy_weight_level = deploy_weight_level
        #     queryset.deploy_precision_level = deploy_precision_level
        #     queryset.deploy_processing_lib = deploy_processing_lib
        #     queryset.deploy_user_edit = deploy_user_edit
        #     queryset.deploy_input_method = deploy_input_method
        #     queryset.deploy_input_data_path = deploy_input_data_path
        #     queryset.deploy_output_method = deploy_output_method
        #     queryset.deploy_input_source = deploy_input_source
        #     queryset.container = 'init'
        #     queryset.container_status = ''

        #     queryset.save()
        # else:
        #     queryset.target = Target.objects.get(id=int(target))
        #     queryset.dataset = dataset
        #     queryset.task_type = task_type
        #     queryset.autonn_dataset_file = autonn_dataset_file
        #     queryset.autonn_basemodel = autonn_basemodel
        #     queryset.nas_type = nas_type
        #     queryset.deploy_weight_level = ''
        #     queryset.deploy_precision_level = ''
        #     queryset.deploy_processing_lib = ''
        #     queryset.deploy_user_edit = ''
        #     queryset.deploy_input_method = ''
        #     queryset.deploy_input_data_path = ''
        #     queryset.deploy_output_method = ''
        #     queryset.deploy_input_source = ''
        #     queryset.container = 'init'
        #     queryset.container_status = ''

        #     queryset.save()

        queryset.target = Target.objects.get(id=int(target))
        queryset.dataset = dataset
        queryset.task_type = task_type
        queryset.autonn_dataset_file = autonn_dataset_file
        queryset.autonn_basemodel = autonn_basemodel
        queryset.nas_type = nas_type
        queryset.deploy_weight_level = deploy_weight_level
        queryset.deploy_precision_level = deploy_precision_level
        queryset.deploy_processing_lib = deploy_processing_lib
        queryset.deploy_user_edit = deploy_user_edit
        queryset.deploy_input_method = deploy_input_method
        queryset.deploy_input_data_path = deploy_input_data_path
        queryset.deploy_output_method = deploy_output_method
        queryset.deploy_input_source = deploy_input_source
        queryset.container = 'init'
        queryset.container_status = ''

        queryset.save()

        print(queryset.deploy_user_edit)

        project_info_content = ""
        if data.target_info != 'ondevice':
            # project_info.yaml
            project_info_content += "# common\n" \
                                   "task_type : {0}\n" \
                                   "target_info : {1}\n" \
                                   "cpu : {2}\n" \
                                   "acc : {3}\n" \
                                   "memory : {4}\n" \
                                   "os : {5}\n" \
                                   "engine : {6}\n" \
                                   "nfs_ip : {7}\n" \
                                   "nfs_path : {8}\n" \
                                   "target_hostip : {9}\n" \
                                   "target_hostport : {10}\n" \
                                   "target_serviceport : {11}\n\n" \
                                   "#for autonn\n" \
                                   "dataset : {12}\n" \
                                   "#basemodel : {13}\n" \
                                   "#nas_type : {14}\n\n" \
                                   "#for deploy\n" \
                                   "lightweight_level : {15}\n" \
                                   "precision_level : {16}\n" \
                                   "#preprocessing_lib : {17}\n" \
                                   "#input_method : {18}\n" \
                                   "#input_data_location : {19}\n" \
                                   "output_method : {20}\n" \
                                    "input_source : {21}\n" \
                                   "user_editing : {22}\n".format(str(task_type),
                                                                  str(data.target_info),
                                                                  str(data.target_cpu),
                                                                  str(data.target_acc),
                                                                  str(int(int(data.target_memory) / 1024)),
                                                                  str(data.target_os),
                                                                  str(data.target_engine),
                                                                  str(data.nfs_ip),
                                                                  str(data.nfs_path),
                                                                  str(data.target_host_ip),
                                                                  str(data.target_host_port),
                                                                  str(data.target_host_service_port),
                                                                  str(dataset),
                                                                  str(autonn_basemodel),
                                                                  str(nas_type),
                                                                  str(deploy_weight_level),
                                                                  str(deploy_precision_level),
                                                                  str(deploy_processing_lib),
                                                                  str(deploy_input_method),
                                                                  str(deploy_input_data_path),
                                                                  str(deploy_output_method),
                                                                  str(deploy_input_source),
                                                                  str(deploy_user_edit))
        else:
            # project_info.yaml
            project_info_content += "# common\n" \
                                    "task_type : {0}\n" \
                                    "target_info : {1}\n" \
                                    "cpu : {2}\n" \
                                    "acc : {3}\n" \
                                    "memory : {4}\n" \
                                    "os : {5}\n" \
                                    "engine : {6}\n" \
                                    "nfs_ip : {7}\n" \
                                    "nfs_path : {8}\n" \
                                    "target_hostip : {9}\n" \
                                    "target_hostport : {10}\n" \
                                    "target_serviceport : {11}\n\n" \
                                    "#for autonn\n" \
                                    "dataset : {12}\n" \
                                    "#basemodel : {13}\n" \
                                    "#nas_type : {14}\n\n" \
                                    "#for deploy\n" \
                                    "lightweight_level : {15}\n" \
                                    "precision_level : {16}\n" \
                                    "#preprocessing_lib : {17}\n" \
                                    "#input_method : {18}\n" \
                                    "#input_data_location : {19}\n" \
                                    "output_method : {20}\n" \
                                    "input_source : {21}\n" \
                                    "user_editing : {22}\n".format(str(task_type),
                                                                     'ondevice',
                                                                     str(data.target_cpu),
                                                                     str(data.target_acc),
                                                                     str(int(int(data.target_memory) / 1024)),
                                                                     str(data.target_os),
                                                                     str(data.target_engine),
                                                                     str(data.nfs_ip),
                                                                     str(data.nfs_path),
                                                                     str(data.target_host_ip),
                                                                     str(data.target_host_port),
                                                                     str(data.target_host_service_port),
                                                                     str(dataset),
                                                                     str(autonn_basemodel),
                                                                     str(nas_type),
                                                                     str(deploy_weight_level),
                                                                     str(deploy_precision_level),
                                                                     str(deploy_processing_lib),
                                                                     str(deploy_input_method),
                                                                     str(deploy_input_data_path),
                                                                     str(deploy_output_method),
                                                                     str(deploy_input_source),
                                                                     str(deploy_user_edit))

        print('project_info_content')
        print(project_info_content)

        # project_info.yaml 파일 생성
        common_path = os.path.join(root_path, "shared/common/{0}/{1}".format(str(request.user),
                                                                             str(request.data['project_id'])))

        print('common_path')
        print(common_path)

        # 디렉토리 유뮤 확인
        if os.path.isdir(common_path) is False:
            os.makedirs(common_path)

        f = open(os.path.join(common_path, 'project_info.yaml'), 'w+')
        f.write(project_info_content)
        f.close()

        return Response(status=200)

    except Exception as e:
        print('error')
        print(e)


# target yaml 파일 생성
@api_view(['GET', 'POST'])
@authentication_classes([OAuth2Authentication])   # 토큰 확인
def target_check(request):
    """
    target_check _summary_

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """

    select_target = {
        1: 'rk3399pro',
        2: 'jetsonnano',
        3: 'x86-cuda',
        4: 'gcp',
    }

    base_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

    target_name = select_target[request.data['selectTarget']]
    target_folder_path = os.path.join(base_dir, 'data/targets/' + target_name)

    target_yaml_path = os.path.join(target_folder_path , target_name + '.yaml')

    # 타겟 폴더 경로 존재
    if os.path.isdir(target_folder_path) is True:

        f = open(target_yaml_path, 'w')
        target_yaml_structure = 'name: ' + target_name
        f.write(target_yaml_structure)
        f.close()

    # 타겟 폴더 경로 없음
    else:
        # 타겟 폴더 생성
        os.mkdir(target_folder_path)

        f = open(target_yaml_path, 'w')
        target_yaml_structure = 'name: ' + target_name
        f.write(target_yaml_structure)
        f.close()

    # 타겟 SW 정보
    target_sw_path = os.path.join(base_dir, 'data/targets/' + target_name + '/SW/sw_info.json')

    f = open(target_sw_path, 'r')
    target_sw_info = json.load(f)

    # Dictionary 정보 합치기
    result = dict({'target_yaml_path': target_yaml_path}, **target_sw_info)
    print(result)

    print('target yaml 파일 생성')
    return Response(result)


# Labelling 저작 도구 데이터 셋 유효성 검사
@api_view(['GET', 'POST'])
@authentication_classes([OAuth2Authentication])   # 토큰 확인
def dataset_check(request):
    """
    dataset_check _summary_

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """

    base_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

    # Labelling 저작 도구 데이터 셋 경로
    dataset_path = os.path.join(base_dir, 'data/datasets/' + request.data['name'])
    print(dataset_path)

    b_raw_image_check = False
    b_annotation_data_check = False
    b_train_data_check = False
    b_val_data_check = False

    # TODO 조건 1 : 서버 폴더 확인
    if os.path.isdir(dataset_path) is True:

        # TODO 조건 2 : raw data 확인
        raw_data_path = os.path.join(dataset_path, 'images')

        # raw_data 폴더 경로 - 존재
        if os.path.isdir(raw_data_path) is True:
            # raw data - 파일 개수 확인
            raw_data_count = os.listdir(raw_data_path)
            if len(raw_data_count) > 0:
                b_raw_image_check = True
        else:
            # raw_data 폴더 생성
            os.mkdir(raw_data_path)

        # TODO 조건 3 : Annotation data 확인
        annotation_data_path = os.path.join(dataset_path, 'annotations')

        # annotation data 폴더 경로 - 존재
        if os.path.isdir(annotation_data_path) is True:

            # annotation data - 파일 개수 확인
            annotation_data_count = os.listdir(annotation_data_path)
            if len(annotation_data_count) > 0:
                b_annotation_data_check = True
        else:
            # annotation data 폴더 생성
            os.mkdir(annotation_data_path)

        # TODO 조건 4 : train data 확인
        train_data_path = os.path.join(dataset_path, 'imagesets/train.txt')
        b_train_data_check = os.path.isfile(train_data_path)

        # TODO 조건 5 : validation data 확인
        validation_data_path = os.path.join(dataset_path, 'imagesets/val.txt')
        b_val_data_check = os.path.isfile(validation_data_path)

        # TODO 조건 6 : yaml 파일 확인 확인
        yaml_file_path = os.path.join(dataset_path, request.data['name'] + '.yaml')
        b_yaml_file_check = os.path.isfile(yaml_file_path)

        # raw data 또는 annotation data 없는 경우 Labelling 저작 도구 요청
        if b_raw_image_check is False or b_annotation_data_check is False:
            # b_raw_image_check, b_annotation_data_check = create_dataset_file(raw_data_path, annotation_data_path)

            dataset_response = create_dataset_file(raw_data_path, annotation_data_path)

            if dataset_response.status_code == 200:
                # Labelling 저작 도구에 전달한 param 재수신 완료
                print(dataset_response.content)

                b_raw_image_check = True
                b_annotation_data_check = True

            else:
                print('server error')
                return Response(status=401)

        # train data 또는 validation data 없는 경우
        if b_train_data_check is False or b_val_data_check is False:
            print('train & validation data 생성')

            b_train_data_check, b_val_data_check = create_train_val_data(dataset_path, raw_data_path)

        if b_yaml_file_check is False:
            print('yaml 파일 생성')
            b_yaml_file_check = create_dataset_yaml(dataset_path, yaml_file_path)

        return Response({'isPath': True, 'raw_data': b_raw_image_check,
                                         'annotation_data': b_annotation_data_check,
                                         'val_data': b_val_data_check,
                                         'train_data': b_train_data_check,
                                         'yaml_file': b_yaml_file_check,
                                         'yaml_file_path': yaml_file_path})

    else:
        return Response({'isPath': False})


# 데이터 셋 파일 생성
def create_dataset_file(r_raw_path, r_anno_path):
    """
    create_dataset_file _summary_

    Args:
        r_raw_path (_type_): _description_
        r_anno_path (_type_): _description_

    Returns:
        _type_: _description_
    """

    try:
        # 전달 param raw image 경로, annotation data 경로, task 정보 : detection
        url = "http://0.0.0.0:8086/create_dataset/"

        # OAuth 토큰 요청
        param = {
            'raw_data_path': r_raw_path,
            'annotation_data_path': r_anno_path,
            'Task': 'detection',
        }

        headers = {
            'Content-Type': 'application/json',
        }

        response = requests.request("post", url, data=json.dumps(param), headers=headers)

        print(response.status_code)

        return response

        # if response.status_code == 200:
        #     # Labelling 저작 도구에 전달한 param 재수신 완료
        #     print(response.content)
        #
        #     return True, True
        #
        # else:
        #     print('server error')

    except Exception as e:
        print(e)
        return False, False


def create_train_val_data(r_data_path, r_raw_data_path):
    """
    create_train_val_data

    Args:
        r_data_path (_type_): _description_
        r_raw_data_path (_type_): _description_

    Returns:
        _type_: _description_
    """

    try:
        print('create_train_val_data')

        # TODO - 1 : train data & annotation data 분리 [ 6 : 1 비율 ]
        # image 폴더 내에 이미지 리스트 정보
        raw_data_list = os.listdir(r_raw_data_path)
        print(raw_data_list)
        print('\n')

        # TODO - 2 : validation data 이미지 리스트 텍스트 파일 생성
        validation_list = random.choices( raw_data_list, k=( int( len(raw_data_list) / 6) ) )
        print(validation_list)
        print(len(validation_list))

        val_file = open(r_data_path + '/imagesets/val.txt', 'w')

        for v in validation_list:
            val_file.write(r_data_path + '/images/' + v + "\n")
        val_file.close()

        # TODO - 3 : train data 이미지 리스트 텍스트 파일 생성
        train_list = set(raw_data_list) - set(validation_list)
        print(train_list)
        print(len(train_list))

        train_file = open(r_data_path + '/imagesets/train.txt', 'w')

        for t in train_list:
            train_file.write(r_data_path + '/images/' + t + "\n")
        train_file.close()

        return True, True

    except Exception as e:
        print(e)
        return False, False


def create_dataset_yaml(r_data_set_path, r_yaml_path):
    """
    create_dataset_yaml _summary_

    Args:
        r_data_set_path (_type_): _description_
        r_yaml_path (_type_): _description_

    Returns:
        _type_: _description_
    """

    try:
        # yaml 파일 생성
        path = r_data_set_path
        imgs = r_data_set_path + '/images'
        annos = r_data_set_path + '/annotations'
        train = r_data_set_path + '/imagesets/train.txt'
        val = r_data_set_path + '/imagesets/val.txt'

        names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        nc = len(names)

        yaml_file_structure = 'path: ' + path + '\n' + \
                              'imgs: ' + imgs + '\n' + \
                              'annos: ' + annos + '\n' + \
                              'train: ' + train + '\n' + \
                              'val: ' + val + '\n\n' + \
                              'num_classes: ' + str(nc) + '\n' + \
                              'names: ' + str(names)

        f = open(r_yaml_path, 'w')
        f.write(yaml_file_structure)
        f.close()

        return True

    except Exception as e:
        print(e)
        return False



# 워크플로우 추가
@api_view(['POST'])
@permission_classes([AllowAny])   # 토큰 확인
def set_workflow(request):
    """
    """
    try:
        project_id = request.data["project_id"]
        workflow = request.data["workflow"]
        workflow_order = WorkflowOrder.objects.filter(project_id=project_id)
        
        if len(list(workflow_order.values())) > 0:
            workflow_order.delete()

        for index, flow in enumerate(workflow):
            target = WorkflowOrder(workflow_name=flow, order = int(index), project_id=int(project_id))
            target.save()

        save_data = WorkflowOrder.objects.filter(project_id=project_id).order_by('order')

        return HttpResponse(json.dumps({'status': 200, 'workflow': list(save_data.values())}))

    except Exception as e:
        return Response(status=500)