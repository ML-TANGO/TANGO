import os
import json
import math
import random
import socket
import threading
import requests

from datetime import datetime

import django.middleware.csrf
from django.http import HttpResponse

from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from oauth2_provider.contrib.rest_framework import OAuth2Authentication

from .models import Target

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# 타겟 생성
@api_view(['GET', 'POST'])
@authentication_classes([OAuth2Authentication])   # 토큰 확인
def target_create(request):
    """
    target_create _summary_

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """

    try:
        target = Target(target_name=request.data['name'],
                        create_user=str(request.user),
                        create_date=str(datetime.now()),
                        target_info=request.data['info'],
                        target_engine=request.data['engine'],
                        target_os=request.data['os'],
                        target_cpu=request.data['cpu'],
                        target_acc=request.data['acc'],
                        target_memory=request.data['memory'],
                        nfs_ip=request.data['nfs_ip'],
                        nfs_path=request.data['nfs_path'],
                        target_host_ip=request.data['host_ip'],
                        target_host_port=request.data['host_port'],
                        target_host_service_port=request.data['host_service_port'],
                        target_image=str(request.data['image']))

        target.save()

        return Response(status=200)

    except Exception as e:
        print(e)


# 타겟 조회 (리스트)
@api_view(['GET', 'POST'])
@authentication_classes([OAuth2Authentication])   # 토큰 확인
def target_read(request):
    """
    target_read _summary_

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """

    try:
        # 모든 사용자가 타겟을 확인할 수 있도록 수정
        queryset = Target.objects.filter()
        data = list(queryset.values())

        data_list = []

        for i in data:

            target_data = {'id': i['id'],
                           'name': i['target_name'],
                           'create_user': i['create_user'],
                           'create_date': i['create_date'],
                           'info': i['target_info'],
                           'engine': i['target_engine'],
                           'os': i['target_os'],
                           'cpu': i['target_cpu'],
                           'acc': i['target_acc'],
                           'memory': i['target_memory'],
                           'nfs_ip': i['nfs_ip'],
                           'nfs_path': i['nfs_path'],
                           'host_ip': i['target_host_ip'],
                           'host_port': i['target_host_port'],
                           'host_service_port': i['target_host_service_port'],
                           'image': str(i['target_image'])}

            data_list.append(target_data)

        return HttpResponse(json.dumps(data_list))

    except Exception as e:
        print(e)


# 타겟 수정
@api_view(['GET', 'POST', 'PUT'])
@authentication_classes([OAuth2Authentication])   # 토큰 확인
def target_update(request):
    """
    target_update _summary_

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """

    try:
        queryset = Target.objects.get(id=int(request.data['id']),
                                      create_user=request.user)

        queryset.target_name = request.data['name']
        queryset.target_info = request.data['info']
        queryset.target_engine = request.data['engine']
        queryset.target_os = request.data['os']
        queryset.target_cpu = request.data['cpu']
        queryset.target_acc = request.data['acc']
        queryset.target_memory = request.data['memory']
        queryset.nfs_ip = request.data['nfs_ip']
        queryset.nfs_path = request.data['nfs_path']
        queryset.target_host_ip = request.data['host_ip']
        queryset.target_host_port = request.data['host_port']
        queryset.target_host_service_port = request.data['host_service_port']
        queryset.target_image = str(request.data['image'])

        queryset.save()

        return Response(status=200)

    except Exception as e:
        print(e)


# 타겟 삭제
@api_view(['GET', 'POST', 'DELETE'])
@authentication_classes([OAuth2Authentication])   # 토큰 확인
def target_delete(request):
    """
    target_delete _summary_

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """

    try:

        queryset = Target.objects.get(id=request.data['id'],
                                      create_user=request.user)  # Project ID로 검색
        queryset.delete()

        return Response(status=200)

    except Exception as e:
        print(e)

# Target 정보 조회
@api_view(['GET', 'POST'])
@authentication_classes([OAuth2Authentication])   # 토큰 확인
def target_info(request):
    """
    target_info _summary_

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """

    queryset = Target.objects.filter(id=request.data['id'])  # Target id로 검색
    data = list(queryset.values())

    target_data = {'id': data[0]['id'],
                           'name': data[0]['target_name'],
                           'create_user': data[0]['create_user'],
                           'create_date': data[0]['create_date'],
                           'info': data[0]['target_info'],
                           'engine': data[0]['target_engine'],
                           'os': data[0]['target_os'],
                           'cpu':data[0]['target_cpu'],
                           'acc': data[0]['target_acc'],
                           'memory': data[0]['target_memory'],
                           'nfs_ip': data[0]['nfs_ip'],
                           'nfs_path': data[0]['nfs_path'],
                           'host_ip': data[0]['target_host_ip'],
                           'host_port': data[0]['target_host_port'],
                           'host_service_port': data[0]['target_host_service_port'],
                           'image': str(data[0]['target_image'])}
    
    return HttpResponse(json.dumps(target_data))