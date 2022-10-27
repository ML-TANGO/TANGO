"""views module for tango
This module for vies.
Attributes:

Todo:

"""

import json
import socket
import requests
import oauthlib.oauth2

from datetime import datetime

from django.http import HttpResponse, HttpResponseRedirect
from django.contrib.auth.models import User, Group
from django.contrib.auth import authenticate
from .serializers import UserSerializer, GroupSerializer

from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from rest_framework.permissions import IsAuthenticated, AllowAny

from .models import Oauth2ProviderAccesstoken, Oauth2ProviderRefreshtoken
from oauth2_provider.contrib.rest_framework import OAuth2Authentication
from oauth2_provider.models import AbstractAccessToken, AbstractRefreshToken, AbstractApplication
from oauth2_provider.views.application import ApplicationRegistration

from .models import Oauth2ProviderApplication
from oauthlib.common import generate_token
from oauth2_provider.models import AbstractApplication


class UserViewSet(viewsets.ModelViewSet):
    """UserViewSet class
    Note:
    Args:
        viewsets.ModelViewSet
    Attributes:
    """
    queryset = User.objects.all()
    serializer_class = UserSerializer


class GroupViewSet(viewsets.ModelViewSet):
    """GroupViewSet class
    Note:
    Args:
        viewsets.ModelViewSet
    Attributes:
    """

    queryset = Group.objects.all()
    serializer_class = GroupSerializer


# 서버 IP
@api_view(['GET', 'POST'])
@authentication_classes([OAuth2Authentication])   # 토큰 확인
def get_server_ip(request):
    """
    _summary_

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """

    try:
        weda_port_num = 8090

        return Response({'port': weda_port_num})

    except Exception as e:
        print(e)


def create_token(request, r_user_id, r_user_pw):
    """
    _summary_

    Args:
        request (_type_): _description_
        r_user_id (_type_): _description_
        r_user_pw (_type_): _description_

    Returns:
        _type_: _description_
    """

    # OAuth 애플리케이션 정보 GET
    oauth_app = Oauth2ProviderApplication.objects.get(name='deep_framework')

    # 현재 서버 IP 주소
    server_ip = request.get_host()
    url = "http://" + server_ip + "/o/token/"

    # OAuth 토큰 요청
    payload = 'client_id=' + oauth_app.client_id + \
              '&client_secret=' + oauth_app.client_secret + \
              '&grant_type=' + oauth_app.authorization_grant_type + \
              '&username=' + r_user_id + \
              '&password=' + r_user_pw
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return response.content


# 토큰 삭제
def delete_token(request, old_token):
    """
    _summary_

    Args:
        request (_type_): _description_
        old_token (_type_): _description_

    Returns:
        _type_: _description_
    """

    # OAuth 애플리케이션 정보 GET
    oauth_app = Oauth2ProviderApplication.objects.get(name='deep_framework')

    # 현재 서버 IP 주소
    server_ip = request.get_host()
    url = "http://" + server_ip + "/o/revoke_token/"

    # OAuth 토큰 요청
    payload = 'client_id=' + oauth_app.client_id + \
              '&client_secret=' + oauth_app.client_secret + \
              '&token=' + old_token

    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return response


# 로그인
@api_view(['POST'])
@permission_classes([AllowAny])
def login(request):
    """
    login _summary_

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """

    try:
        # TODO : 사용자 정보 확인 및 토큰 유무 확인
        user_id = request.data['user_id']
        user_pw = request.data['password']

        user_info = authenticate(username=user_id,
                                 password=user_pw)

        # 사용자 정보가 없는 경우
        if user_info is None:
            print('사용자 정보 없음')

            return Response({'result': False})

        else:
            # 토큰 유무 확인
            user_token_search = Oauth2ProviderAccesstoken.objects.filter(user=user_info.id)

            print(len(user_token_search))

            # 토큰 정보가 없는 경우
            if len(user_token_search) == 0:

                # 토큰 생성
                content = create_token(request, user_id, user_pw)

                return Response({'result': True, 'content': content})

            # 기존 토큰 정보가 있는 경우
            else:
                # 기존 토큰 삭제
                for i in user_token_search:
                    delete_token(request, i.token)

                user_refresh_token = Oauth2ProviderRefreshtoken.objects.filter(user=user_info)

                print(len(user_refresh_token))
                for r_token in user_refresh_token:
                    r_token.delete()

                # 토큰 생성
                content = create_token(request, user_id, user_pw)

                return Response({'result': True, 'content': content})

    except Exception as e:
        print(e)

        return Response(status=500)


# 로그 아웃
@api_view(['GET', 'POST'])
@permission_classes([OAuth2Authentication])
def logout(request):
    """
    logout _summary_

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """

    try:
        token = request.headers.get('Authorization', None)

        if token is not None:
            # OAuth 애플리케이션 정보 GET
            oauth_app = Oauth2ProviderApplication.objects.get(name='deep_framework')

            # 현재 서버 IP 주소
            server_ip = request.get_host()
            url = "http://" + server_ip + "/o/revoke_token/"

            # OAuth 토큰 요청
            payload = 'client_id=' + oauth_app.client_id + \
                      '&client_secret=' + oauth_app.client_secret + \
                      '&token=' + token
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded'
            }

            response = requests.request("POST", url, headers=headers, data=payload)
            print(response)

        else:
            print('token 없음')

        return Response(status=200)

    except Exception as e:
        print(e)


# 회원 가입
@api_view(['POST'])
@permission_classes([AllowAny])
def signup(request):
    """
    signup _summary_

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """

    try:
        user = User.objects.create_user(username=request.data['id'],
                                        email=request.data['email'],
                                        password=request.data['password'])
        user.save()

    except Exception as e:
        print(e)

    return Response(status=200)


# ID 중복 체크
@api_view(['GET', 'POST'])
@permission_classes([AllowAny])
def user_id_check(request):
    """
    user_id_check _summary_

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """

    try:
        # Get ID Info
        duplicate_check = User.objects.get(username=request.data['id'])

    except Exception as e:
        print(e)
        duplicate_check = None

    # 중복 아이디 없는 경우
    if duplicate_check is None:
        return Response({'result': True})
    else:
        return Response({'result': False})
