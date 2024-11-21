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
from rest_framework.decorators import api_view, permission_classes, permission_classes

from rest_framework.permissions import AllowAny, IsAuthenticated

from .load_targets import convert_image_to_base64

from .models import Target

from django.db.models import Case, When, Value, IntegerField

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# 타겟 생성
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])   # 토큰 확인
def target_create(request):
    """
    target_create _summary_

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """

    image_base64 = request.data['image']

    if image_base64 == "" or image_base64 == None:
        image_path = os.path.join(BASE_DIR, "images", "default_target.png")
        image_base64 = convert_image_to_base64(image_path)

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
                        target_image=str(image_base64))

        target.save()

        return Response(status=200)

    except Exception as e:
        print(e)


# 타겟 조회 (리스트)
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])   # 토큰 확인
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
        queryset = Target.objects.annotate(
            # 정렬 우선순위 필드 추가
            order_priority=Case(
                When(order=0, then=Value(1)),  # 'order'가 0인 경우 우선순위를 1로 설정 (후순위)
                default=Value(0),              # 나머지는 우선순위를 0으로 설정 (정상 순위)
                output_field=IntegerField(),
            )
        ).order_by('order_priority', 'order')
        
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
                           'image': str(i['target_image']),
                           'order': int(i['order']),
                           }

            data_list.append(target_data)

        return HttpResponse(json.dumps(data_list))

    except Exception as e:
        print(e)


# 타겟 수정
@api_view(['GET', 'POST', 'PUT'])
@permission_classes([IsAuthenticated])   # 토큰 확인
def target_update(request):
    """
    target_update _summary_

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """

    try:
        # queryset = Target.objects.get(id=int(request.data['id']),
        #                               create_user=request.user)

        queryset = Target.objects.get(id=int(request.data['id']))

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
@permission_classes([IsAuthenticated])   # 토큰 확인
def target_delete(request):
    """
    target_delete _summary_

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """

    try:

        # queryset = Target.objects.get(id=request.data['id'],
        #                               create_user=request.user)  # Project ID로 검색
        queryset = Target.objects.get(id=request.data['id'])  # Project ID로 검색
        queryset.delete()

        return Response(status=200)

    except Exception as e:
        print(e)

# Target 정보 조회
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])   # 토큰 확인
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
    
    return HttpResponse(json.dumps(target_to_response(data[0])))


def target_to_response(target):
    to_dict = {
        'id': target['id'],
        'name': target['target_name'],
        'create_user': target['create_user'],
        'create_date': target['create_date'],
        'info': target['target_info'],
        'engine': target['target_engine'],
        'os': target['target_os'],
        'cpu':target['target_cpu'],
        'acc': target['target_acc'],
        'memory': target['target_memory'],
        'nfs_ip': target['nfs_ip'],
        'nfs_path': target['nfs_path'],
        'host_ip': target['target_host_ip'],
        'host_port': target['target_host_port'],
        'host_service_port': target['target_host_service_port'],
        'image': str(target['target_image'])
    }
    return to_dict