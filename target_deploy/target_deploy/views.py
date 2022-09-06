"""
view.py
"""
import os
import json
import requests
from django.shortcuts import render

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt, csrf_protect


@csrf_exempt
def test(request):
    """
    test
    """
    try:
        return HttpResponse('test')

    except requests.exceptions.RequestException as err: # Exception as e:  ## khlee 
        print(err)

@csrf_exempt
def token_check(token):
    """
    token_check
    """
    try:

        # 전달 파라미터 raw image 경로, annotation data 경로, task 정보 : detection
        url = 'http://0.0.0.0:8085/api/user/'

        headers = {
            'Content-Type': 'application/json',
            'Authorization': str(token)
        }

        response = requests.request("get", url, headers=headers)

        if response.status_code == 200:
            # 레이블링 저작도구에 전달한 파라미터 재수신 완료
            result = json.loads(response.content)
            request_username = result[0]['username']
            print('요청 사용자 : ' + request_username)

            return True

        else:
            print('유저 정보가 유효하지 않음')
            return False

    except requests.exceptions.RequestException as err: # Exception as e:  ## khlee 
        print(err)

        return False


# 이미지 배포
@csrf_exempt
def deploy_image(request):
    """
    deploy_image
    """

    try:
        print('이미지 배포')

        # 헤더 정보
        # token_check_result = token_check(request.META['HTTP_AUTHORIZATION'])
        #
        # if token_check_result is True:
        #
        #     data = json.loads(request.body)
        #
        #     target_image_save_path = data['target_image_save_path']     # 타겟 이미지 경로
        #     target_url = data['target_url']                             # 타겟 URL
        #     startup_command = data['startup_command']                   # startup 명령
        #
        #     print(target_image_save_path)
        #     print(target_url)
        #     print(startup_command)
        #
        #     # TODO - 1 : backend.ai manager 신경망 실행 [ 컨테이너 생성 요청 ]
        #
        #     # TODO - 2 : backend.ai agent 신경망 실행 [ 컨테이너 생성 ]
        #
        #     # Deep Framework 서버에 컨테이너 생성 성공여부 반환
        #     return HttpResponse(status=200)
        #
        # else:
        #     return HttpResponse(status=401)

        data = json.loads(request.body)

        target_image_save_path = data['target_image_save_path']     # 타겟 이미지 경로
        target_url = data['target_url']                             # 타겟 URL
        startup_command = data['startup_command']                   # startup 명령

        print(target_image_save_path)
        print(target_url)
        print(startup_command)

        # TODO - 1 : backend.ai manager 신경망 실행 [ 컨테이너 생성 요청 ]

        # TODO - 2 : backend.ai agent 신경망 실행 [ 컨테이너 생성 ]

        # Deep Framework 서버에 컨테이너 생성 성공여부 반환
        return HttpResponse(status=200)

    except ValueError as err:  # Exception as e:  ## khlee
        print(err)
