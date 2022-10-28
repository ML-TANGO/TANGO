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

from datetime import datetime

import django.middleware.csrf
from django.http import HttpResponse
from rest_framework.permissions import AllowAny

from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from oauth2_provider.contrib.rest_framework import OAuth2Authentication

from .models import Project, AuthUser, Target

from django.views.decorators.csrf import csrf_exempt, csrf_protect

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(os.path.dirname(BASE_DIR))

# @permission_classes([IsAuthenticated])                  # 권한 체크 - 로그인 여부
# @authentication_classes([JSONWebTokenAuthentication])   # 토큰 확인
# @permission_classes([AllowAny])


# 컨테이너 상태 결과 응답
@api_view(['GET'])
@permission_classes([AllowAny])   # 토큰 확인
def status_report(request):

    try:
        container_list = ['bms',
                          'viz2code',
                          'autonn',
                          'code_gen',
                          'cloud_deployment',
                          'ondevice_deployment']

        container_id = request.GET['container_id']

        if container_id not in container_list:
            return HttpResponse(status=400, content={'Container ID Not Find'})

        user_id = request.GET['user_id']
        project_id = request.GET['project_id']
        result = request.GET['result']

        queryset = Project.objects.get(id=project_id, create_user=str(user_id))
        queryset.container = container_id
        queryset.container_status = result

        queryset.save()

        return HttpResponse(json.dumps({'status':200}))

    except Exception as error:
        print(error)
        return HttpResponse(error)


# 컨테이너 상태
@api_view(['GET', 'POST'])
@authentication_classes([OAuth2Authentication])   # 토큰 확인
def status_result(request):
    """
    project_list_get _summary_

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """

    try:
        project_id = request.data['project_id']
        queryset = Project.objects.get(id=project_id, create_user=str(request.user))

        print(queryset.container)
        print(queryset.container_status)

        return HttpResponse(json.dumps({'container': queryset.container,
                                        'container_status': queryset.container_status}))

    except Exception as e:
        print(e)


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

    queryset = Project.objects.filter(id=request.data['id'],
                                      create_user=request.user)  # Project id로 검색
    data = list(queryset.values())

    # TODO : 타겟이 0이 아닌 경우 SW 정보 전달
    if data[0]['target'] is not None:
        print('타겟 정보 있음')
        print(data[0]['target'])

        # select_target = {
        #     1: 'rk3399pro',
        #     2: 'jetsonnano',
        #     3: 'x86-cuda',
        #     4: 'gcp',
        # }

        # 타겟 정보
        base_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
        # target_sw_path = os.path.join(base_dir, 'data/targets/' + data[0]['target'] + '/SW/sw_info.json')
        #
        # f = open(target_sw_path, 'r')
        # target_sw_info = json.load(f)

        # 데이터 셋 리스트 정보
        dataset_path = os.path.join(base_dir, 'data/datasets/')
        dataset_list = os.listdir(dataset_path)
        dataset_list_dic = {"dataset_list": dataset_list}

        # 타겟 리스트 정보
        target_path = os.path.join(base_dir, 'data/targets/')
        target_list = os.listdir(target_path)
        target_list_dic = {"target_list": target_list}

        # Dictionary 정보 합치기
        # result = dict(data[0], **target_sw_info, **dataset_list_dic, **target_list_dic)
        result = dict(data[0], **dataset_list_dic, **target_list_dic)

        return Response(result)

    else:

        # 데이터 셋 리스트 정보
        base_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
        dataset_path = os.path.join(base_dir, 'data/datasets/')
        dataset_list = os.listdir(dataset_path)
        dataset_list_dic = {"dataset_list": dataset_list}

        # 타겟 리스트 정보
        target_path = os.path.join(base_dir, 'data/targets/')
        target_list = os.listdir(target_path)
        target_list_dic = {"target_list": target_list}

        # 딕셔너리 정보 합치기
        result = dict(data[0], **dataset_list_dic, **target_list_dic)

        return Response(result)


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

        target = str(request.data['project_target'])
        dataset = str(request.data['project_dataset'])

        task_type = str(request.data['task_type'])
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

        queryset = Project.objects.get(id=request.data['project_id'],
                                       create_user=request.user)

        # 타겟 정보 수신
        data = Target.objects.get(id=int(target))
        if data.target_info != 'ondevice':

            queryset.target = target
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
            queryset.container = 'init'
            queryset.container_status = ''

            queryset.save()
        else:
            queryset.target = target
            queryset.dataset = dataset
            queryset.task_type = task_type
            queryset.autonn_dataset_file = autonn_dataset_file
            queryset.autonn_basemodel = autonn_basemodel
            queryset.nas_type = nas_type
            queryset.deploy_weight_level = ''
            queryset.deploy_precision_level = ''
            queryset.deploy_processing_lib = ''
            queryset.deploy_user_edit = ''
            queryset.deploy_input_method = ''
            queryset.deploy_input_data_path = ''
            queryset.deploy_output_method = ''
            queryset.container = 'init'
            queryset.container_status = ''

            queryset.save()


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
                                   "target_hostip : {7}\n" \
                                   "target_hostport : {8}\n" \
                                   "target_serviceport : {9}\n\n" \
                                   "#for autonn\n" \
                                   "dataset : {10}\n" \
                                   "basemodel : {11}\n" \
                                   "nas_type : {12}\n\n" \
                                   "#for deploy\n" \
                                   "# lightweight_level : {13}\n" \
                                   "# precision_level : {14}\n" \
                                   "# preprocessing_lib : {15}\n" \
                                   "# input_method : {16}\n" \
                                   "# input_data_location : {17}\n" \
                                   "# output_method : {18}\n" \
                                   "# user_editing : {19}\n".format(task_type,
                                                                  data.target_info,
                                                                  data.target_cpu,
                                                                  data.target_acc,
                                                                  data.target_memory,
                                                                  data.target_os,
                                                                  data.target_engine,
                                                                  data.target_host_ip,
                                                                  data.target_host_port,
                                                                  data.target_host_service_port,
                                                                  autonn_dataset_file,
                                                                  autonn_basemodel,
                                                                  nas_type,
                                                                  deploy_weight_level,
                                                                  deploy_precision_level,
                                                                  deploy_processing_lib,
                                                                  deploy_input_method,
                                                                  deploy_input_data_path,
                                                                  deploy_output_method,
                                                                  deploy_user_edit)
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
                                    "# target_hostip : {7}\n" \
                                    "# target_hostport : {8}\n" \
                                    "# target_serviceport : {9}\n\n" \
                                    "#for autonn\n" \
                                    "dataset : {10}\n" \
                                    "basemodel : {11}\n" \
                                    "nas_type : {12}\n\n" \
                                    "#for deploy\n" \
                                    "# lightweight_level : {13}\n" \
                                    "# precision_level : {14}\n" \
                                    "# preprocessing_lib : {15}\n" \
                                    "# input_method : {16}\n" \
                                    "# input_data_location : {17}\n" \
                                    "# output_method : {18}\n" \
                                    "# user_editing : {19}\n".format(task_type,
                                                                     data.target_info,
                                                                     data.target_cpu,
                                                                     data.target_acc,
                                                                     data.target_memory,
                                                                     data.target_os,
                                                                     data.target_engine,
                                                                     data.target_host_ip,
                                                                     data.target_host_port,
                                                                     data.target_host_service_port,
                                                                     autonn_dataset_file,
                                                                     autonn_basemodel,
                                                                     nas_type,
                                                                     deploy_weight_level,
                                                                     deploy_precision_level,
                                                                     deploy_processing_lib,
                                                                     deploy_input_method,
                                                                     deploy_input_data_path,
                                                                     deploy_output_method,
                                                                     deploy_user_edit)

        print(project_info_content)

        # project_info.yaml 파일 생성
        common_path = os.path.join(root_path, "shared/common/{0}/{1}".format(str(request.user), str(request.data['project_id'])))

        print(common_path)

        # 디렉토리 유뮤 확인
        if os.path.isdir(common_path) is False:
            os.makedirs(common_path)

        f = open(os.path.join(common_path, 'project_info.yaml'), 'w+')
        f.write(project_info_content)
        f.close()

        return Response(status=200)

    except Exception as e:
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
