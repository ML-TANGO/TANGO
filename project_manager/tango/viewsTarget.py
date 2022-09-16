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
                        create_user=request.user,
                        create_date=str(datetime.now()),
                        target_cpu=request.data['cpu'],
                        target_gpu=request.data['gpu'],
                        target_memory=request.data['memory'],
                        target_model=request.data['model'],
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
        queryset = Target.objects.filter(create_user=str(request.user))
        data = list(queryset.values())

        data_list = []

        for i in data:

            target_data = {'id': i['id'], 'target_name': i['target_name'],
                           'create_user': i['create_user'], 'create_date': i['create_date'],
                           'target_cpu': i['target_cpu'], 'target_gpu': i['target_gpu'],
                           'target_memory': i['target_memory'], 'target_model': i['target_model'],
                           'target_image': str(i['target_image'])}

            data_list.append(target_data)

            # fPath = os.path.join(BASE_DIR, 'image')
            # with open(os.path.join(fPath, str(i['id'] + 1) + '.jpg'), mode='wb') as file:
            #     file.write(i['target_image'])

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
        queryset.target_image = request.data['image']
        queryset.target_cpu = request.data['cpu']
        queryset.target_gpu = request.data['gpu']
        queryset.target_memory = request.data['memory']
        queryset.target_model = request.data['model']
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
