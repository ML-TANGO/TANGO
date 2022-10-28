"""
views.py
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

    except requests.exceptions.RequestException  as err: # Exception as e:  ## khlee 
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
# 이미지 생성
@csrf_exempt
def create_image(request):
    """
    create_image
    """

    try:
        print('이미지 생성')

        # 헤더 정보
        # token_check_result = token_check(request.META['HTTP_AUTHORIZATION'])
        #
        # if token_check_result is True:
        #
        #     data = json.loads(request.body)
        #
        #     source_image_path = data['source_image_path']                   # 소스 이미지 경로
        #     target_image_save_path = data['target_image_save_path']         # 타겟 이미지 저장 경로
        #     target_name = data['target_name']                               # 타겟 이름
        #     target_os = data['target_os']                                   # os
        #     target_engine = data['target_engine']                           # 가속 엔진
        #     target_ml_lib = data['target_ml_lib']                           # ML 라이브러리
        #     target_module = data['target_module']                           # 의존성 모듈
        #     neural_model_save_path = data['neural_model_save_path']         # 신경망 모델 저장 경로
        #     neural_run_app_path = data['neural_run_app_path']               # 신경망 실행 app 저장 경로
        #
        #     print(source_image_path)
        #     print(target_image_save_path)
        #     print(target_name)
        #     print(target_os)
        #     print(target_engine)
        #     print(target_ml_lib)
        #     print(target_module)
        #     print(neural_model_save_path)
        #     print(neural_run_app_path)
        #
        #     # TODO : 실행 이미지 생성
        #
        #     # Deep Framework 서버에 타겟 이미지 저장 경로 전달
        #     return HttpResponse(json.dumps({'target_image_save_path': target_image_save_path}))
        #
        # else:
        #     return HttpResponse(status=401)


        data = json.loads(request.body)

        source_image_path = data['source_image_path']                   # 소스 이미지 경로
        target_image_save_path = data['target_image_save_path']         # 타겟 이미지 저장 경로
        target_name = data['target_name']                               # 타겟 이름
        target_os = data['target_os']                                   # os
        target_engine = data['target_engine']                           # 가속 엔진
        target_ml_lib = data['target_ml_lib']                           # ML 라이브러리
        target_module = data['target_module']                           # 의존성 모듈
        neural_model_save_path = data['neural_model_save_path']         # 신경망 모델 저장 경로
        neural_run_app_path = data['neural_run_app_path']               # 신경망 실행 app 저장 경로

        print(source_image_path)
        print(target_image_save_path)
        print(target_name)
        print(target_os)
        print(target_engine)
        print(target_ml_lib)
        print(target_module)
        print(neural_model_save_path)
        print(neural_run_app_path)

        # TODO : 실행 이미지 생성

        # Deep Framework 서버에 타겟 이미지 저장 경로 전달
        return HttpResponse(json.dumps({'target_image_save_path': target_image_save_path}))

    except ValueError as err:  # Exception as e:  ## khlee
        print(err)
