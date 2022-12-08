"""viewsContainer module for tango
This module for vies.
Attributes:

Todo:

"""

import json

from django.http import HttpResponse
from rest_framework.permissions import AllowAny

from rest_framework.decorators import api_view, permission_classes, authentication_classes
from oauth2_provider.contrib.rest_framework import OAuth2Authentication

from .models import Project


# 컨테이너 상태 결과 응답
@api_view(['GET'])
@permission_classes([AllowAny])   # 토큰 확인
def status_report(request):

    try:
        # container_list = ['bms',
        #                   'vis2code',
        #                   'autonn',
        #                   'code_gen',
        #                   'cloud_deployment',
        #                   'ondevice_deployment']
        #
        # container_id = request.GET['container_id']
        #
        # if container_id not in container_list:
        #     return HttpResponse(status=400, content={'Container ID Not Find'})

        print('status_report')

        # container_level = 0
        # if 'bms' in request.GET['container_id']:
        #     container_level = 1
        # elif 'viz' in request.GET['container_id']:
        #     container_level = 2
        # elif 'auto' in request.GET['container_id']:
        #     container_level = 3
        # elif 'code' in request.GET['container_id']:
        #     container_level = 4
        # elif 'deploy' in request.GET['container_id']:
        #     container_level = 5

        user_id = request.GET['user_id']
        project_id = request.GET['project_id']

        container_id = str(request.GET['container_id'])
        result = request.GET['result']

        print('user_id : ' + user_id)
        print('project_id : ' + project_id)
        print('container_id : ' + container_id)
        print('result : ' + result)

        queryset = Project.objects.get(id=project_id, create_user=str(user_id))
        queryset.container = container_id
        queryset.container_status = result

        queryset.save()

        return HttpResponse(json.dumps({'status': 200}))

    except Exception as error:
        print(error)
        return HttpResponse(error)


# 컨테이너 정보 확인
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


# 컨테이너 업데이트
@api_view(['GET', 'POST'])
@authentication_classes([OAuth2Authentication])   # 토큰 확인
def status_update(request):
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

        # container_level = 0
        # if 'bms' in request.data['container']:
        #     container_level = 1
        # elif 'viz' in request.data['container']:
        #     container_level = 2
        # elif 'auto' in request.data['container']:
        #     container_level = 3
        # elif 'code' in request.data['container']:
        #     container_level = 4
        # elif 'deploy' in request.data['container']:
        #     container_level = 5

        queryset.container = str(request.data['container'])
        queryset.container_status = request.data['container_status']

        queryset.save()

        print(queryset.container)
        print(queryset.container_status)

        return HttpResponse(json.dumps({'container': queryset.container,
                                        'container_status': queryset.container_status}))

    except Exception as e:
        print(e)
