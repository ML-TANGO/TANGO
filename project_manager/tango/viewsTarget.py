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
        target = Target(name=request.data['name'],
                        create_user=str(request.user),
                        create_date=str(datetime.now()),
                        info=request.data['info'],
                        engine=request.data['engine'],
                        os=request.data['os'],
                        cpu=request.data['cpu'],
                        acc=request.data['acc'],
                        memory=request.data['memory'],
                        host_ip=request.data['host_ip'],
                        host_port=request.data['host_port'],
                        host_service_port=request.data['host_service_port'],
                        image=str(request.data['image']))

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
        queryset = Target.objects.all()
        target_all_list = list(queryset.values())

        return HttpResponse(json.dumps(target_all_list))

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

        queryset.name = request.data['name']
        queryset.info = request.data['info']
        queryset.engine = request.data['engine']
        queryset.os = request.data['os']
        queryset.cpu = request.data['cpu']
        queryset.acc = request.data['acc']
        queryset.memory = request.data['memory']
        queryset.host_ip = request.data['host_ip']
        queryset.host_port = request.data['host_port']
        queryset.host_service_port = request.data['host_service_port']
        queryset.image = str(request.data['image'])

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
