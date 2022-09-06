'''
views.py
'''

import json
import requests

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def test(request):
    '''
    test
    '''
    try:
        return HttpResponse('test')

    except requests.exceptions.RequestException as err: # Exception as e:  ## khlee 
        print(err)


@csrf_exempt
def token_check(token):
    '''
    token_check
    '''
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

    except ValueError as e:
        print(e)

        return False


# 신경망 생성
@csrf_exempt
def create_neural(request):
    '''
    create_neural
    '''
    try:
        print('신경망 생성')

        # # 헤더 정보
        # token_check_result = token_check(request.META['HTTP_AUTHORIZATION'])
        #
        # if token_check_result is True:
        #
        #     data = json.loads(request.body)
        #
        #     print(data)
        #
        #     data_yaml_path = data['data_yaml_path']
        #     target_yaml_path = data['target_yaml_path']
        #
        #     print(data_yaml_path)
        #     print(target_yaml_path)
        #
        #
        #     # TODO : 신경망 생성
        #
        #
        #     # 신경망 모델 저장 경로 및 모델 이름 전달
        #     data_path_info = data_yaml_path.split('/')
        #     data_name = data_path_info[
        #           len(data_path_info) - 1].replace('.yaml', '')
        #
        #     target_path_info = target_yaml_path.split('/')
        #     target_name = \
        #       target_path_info[
        #               len(target_path_info) - 1].replace('.yaml', '')
        #
        #     neural_model_name = \
        #               'best_detnn-' + data_name + '-' + target_name + '.onnx'
        #     neural_model_path = \
        #               target_yaml_path.replace('.yaml', '') \
        #               + '/model/' + neural_model_name
        #
        #     print(neural_model_name)
        #     print(neural_model_path)
        #
        #     return HttpResponse(json.dumps({
        #           'neural_model_path':neural_model_path,
        #           'neural_model_name':neural_model_name}))
        #
        # else:
        #     return HttpResponse(status=401)

        data = json.loads(request.body)

        print(data)

        data_yaml_path = data['data_yaml_path']
        target_yaml_path = data['target_yaml_path']

        print(data_yaml_path)
        print(target_yaml_path)

        # TODO : 신경망 생성

        # 신경망 모델 저장 경로 및 모델 이름 전달
        data_path_info = data_yaml_path.split('/')
        data_name = data_path_info[
            len(data_path_info) - 1].replace('.yaml', '')

        target_path_info = target_yaml_path.split('/')
        target_name = target_path_info[
            len(target_path_info) - 1].replace('.yaml', '')

        neural_model_name = ('best_detnn-' + data_name +
                             '-' + target_name + '.onnx')
        neural_model_path = (target_yaml_path.replace('.yaml', '') +
                             '/model/' + neural_model_name)

        print(neural_model_name)
        print(neural_model_path)

        return HttpResponse(json.dumps(
            {'neural_model_path': neural_model_path,
             'neural_model_name': neural_model_name}))

    except ValueError as e:
        print(e)
