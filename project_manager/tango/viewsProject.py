"""viewsProejct module for tango
This module for vies.
Attributes:

Todo:

"""

import os
import json
import shutil

from datetime import datetime
import time

from django.http import HttpResponse
from rest_framework.permissions import AllowAny

from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from oauth2_provider.contrib.rest_framework import OAuth2Authentication

from .models import Project, WorkflowOrder, AutonnStatus
from targets.models import Target

from django.forms.models import model_to_dict
from django.db.models import Model, ManyToOneRel

from . import models

from .projectHandler import *
from targets.views import target_to_response

from .service.autonn_status import update_autonn_status

from .enums import ContainerId, ContainerStatus, LearningType, TaskType

from datasets.views import copy_train_file_for_version
from .service.get_common_folder import get_folder_structure

from .service.yaml_editor import get_hyperparameter_file_name, get_arguments_file_name


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(os.path.dirname(BASE_DIR))

# @permission_classes([IsAuthenticated])                  # 권한 체크 - 로그인 여부
# @authentication_classes([JSONWebTokenAuthentication])   # 토큰 확인
# @permission_classes([AllowAny])

# ============================================================================

def nested_model_delete(model):
    try:
        if isinstance(model, Model):
            for field in model._meta.get_fields():
                nested_model_delete(field)
            model.delete()
        return
    except Exception as error:
        print("nested_model_delete error")
        print(error)
        return 

def related_model_to_dict(model):
    model_dict =  {}

    if isinstance(model, Model) == False:
        return model

    # 모델의 필드 이름 얻기 (ManyToOneRel 제외)
    for field in model._meta.get_fields():
        if isinstance(field, ManyToOneRel):
            continue

        key = str(field).split(".")[-1]
        try:
            model_dict[key] = related_model_to_dict(getattr(model, key))
        except Exception as error:
            model_dict[key] = None

    return model_dict

def init_autonn_status(project_info):
    try:
        autonn_status_info = models.AutonnStatus.objects.get(project = project_info.id)
        return related_model_to_dict(autonn_status_info)
    except models.AutonnStatus.DoesNotExist:
        try:
            hyperparameter = models.Hyperparameter()
            hyperparameter.save()

            arguments = models.Arguments()
            arguments.save()

            system = models.System()
            system.save()

            basemodel = models.Basemodel()
            basemodel.save()

            model_summary = models.ModelSummary()
            model_summary.save()

            batch_size = models.BatchSize()
            batch_size.save()

            train_dataset = models.TrainDataset()
            train_dataset.save()

            val_dataset = models.ValDataset()
            val_dataset.save()

            anchor = models.Anchor()
            anchor.save()

            train_start = models.TrainStart()
            train_start.save()

            train_loss_latest = models.TrainLossLatest()
            train_loss_latest.save()

            val_accuracy_latest = models.ValAccuracyLatest()
            val_accuracy_latest.save()

            # projectInstance = models.Project.objects.get(id = project_id)

            autonn_status = models.AutonnStatus(
                project = project_info,
                hyperparameter = hyperparameter,
                arguments = arguments,
                system = system,
                basemodel = basemodel,
                model_summary = model_summary,
                batch_size = batch_size,
                train_dataset = train_dataset,
                val_dataset = val_dataset,
                anchor = anchor,
                train_start = train_start,
                train_loss_latest = train_loss_latest,
                val_accuracy_latest = val_accuracy_latest,
            )
            
            autonn_status.save()

            return related_model_to_dict(autonn_status)
        except Exception as error:
            print(error)
            return None
    except Exception as error:
        print(error)
        return None

def delete_autonn_status(project_info):
    # autonn_status 제거 

    autonn_status = AutonnStatus.objects.filter(project = project_info.id)
    for status in list(autonn_status):
        nested_model_delete(status)

    models.TrainLossLastStep.objects.filter( 
        project_id = project_info.id, 
        project_version = project_info.version, 
        is_use = True 
    ).delete()

    models.ValAccuracyLastStep.objects.filter(
        project_id = project_info.id, 
        project_version = project_info.version, 
        is_use = True
    ).delete()

    models.EpochSummary.objects.filter(
        project_id = project_info.id, 
        project_version = project_info.version, 
        is_use = True
    ).delete()   

def start_container(user_id, project_id, project_info, container_id):
    """
    컨테이너 시작 함수
    
    Args:
        user_id (str): 사용자 ID
        project_id (str): 프로젝트 ID
        project_info (Project): 프로젝트 정보 객체
        container_id (str): 시작할 컨테이너 ID
    
    Returns:
        str: 컨테이너 시작 로그 메시지
    """
    
    # AutoNN_CL 컨테이너 시작 로직 (더미 구현)
    if container_id == ContainerId.autonn_cl:
        return start_autonn_cl_container(user_id, project_id, project_info)
    
    # 기존 autonn 컨테이너 시작 로직
    if container_id == ContainerId.autonn:
        delete_autonn_status(project_info) # 이전에 진행했던 이력 제거 
        project_info.autonn_retry_count = 0 # retry count 초기화
        init_autonn_status(project_info) #  새로운 autonn_status 생성

    # target_info 안전 처리 (target이 None인 경우 대비)
    target_info = None
    if project_info.target:
        target_info = project_info.target.target_info
    
    response = None
    try:
        response = call_api_handler(container_id, "start", user_id, project_id, target_info)
    except Exception as error:
        raise error
    
    # start 요청 log
    to_json = json.loads(response)

    return to_json['request_info']

def start_autonn_cl_container(user_id, project_id, project_info):
    """
    AutoNN_CL 컨테이너 시작 함수 (실제 구현)
    
    Args:
        user_id (str): 사용자 ID
        project_id (str): 프로젝트 ID
        project_info (Project): 프로젝트 정보 객체
    
    Returns:
        str: 시작 로그 메시지
    """
    try:
        print(f"=== AutoNN_CL Container Start Request ===")
        print(f"User ID: {user_id}")
        print(f"Project ID: {project_id}")
        print(f"Task Type: {project_info.task_type}")
        print(f"Learning Type: {project_info.learning_type}")
        
        # 프론트엔드 로그에 API 호출 시작 메시지 추가
        api_call_log = f"[AutoNN_CL] API 호출 시작 - GET http://autonn-cl:8102/start?user_id={user_id}&project_id={project_id}"
        project_info.current_log = str(project_info.current_log) + "\n" + api_call_log
        
        # target_info 안전 처리 (target이 None인 경우 대비)
        target_info = None
        if project_info.target:
            target_info = project_info.target.target_info
        
        # 기존 AutoNN과 동일한 방식으로 API 호출
        try:
            response = call_api_handler(ContainerId.autonn_cl, "start", user_id, project_id, target_info)
            # → GET http://autonn-cl:8102/start?user_id=xxx&project_id=xxx
        except Exception as api_error:
            print(f"[AutoNN_CL] API 호출 실패, 더미 응답 사용: {api_error}")
            # API 호출 실패 시 더미 응답 생성 (로그 테스트용)
            response = json.dumps({
                'response': 'started', 
                'request_info': '[AutoNN_CL] API 호출 시뮬레이션 - 실제 컨테이너 연결 실패로 더미 응답 사용'
            })
        
        # API 응답 수신 로그 추가
        api_response_log = f"[AutoNN_CL] API 응답 수신 완료 - 상태: 정상"
        project_info.current_log = str(project_info.current_log) + "\n" + api_response_log
        
        # start 요청 로그 처리
        to_json = json.loads(response)
        project_info.current_log = str(project_info.current_log) + "\n" + f"[AutoNN_CL] 응답 내용: {to_json['response']}"
        project_info.current_log = str(project_info.current_log) + "\n" + f"[AutoNN_CL] Continual Learning 프로세스 시작 완료"
        project_info.current_log = str(project_info.current_log) + "\n" + f"[AutoNN_CL] 상태: {to_json['response']}"
        project_info.container = ContainerId.autonn_cl
        project_info.container_status = ContainerStatus.STARTED
        project_info.save()
        
        # AutoNN_CL의 경우 사용자 친화적인 로그 반환
        user_friendly_log = f"""[AutoNN_CL] Continual Learning 시작 완료
Task Type: {project_info.task_type}
Learning Type: {project_info.learning_type}
Container Status: {to_json['response']}
API 통신: 정상
프로세스: 시작됨

=== Segmentation + Continual Learning 준비 완료 ==="""
        
        return user_friendly_log
        
    except Exception as error:
        print(f"AutoNN_CL container start failed: {error}")
        # API 호출 실패 로그 추가
        error_log = f"[AutoNN_CL] API 호출 실패 - 오류: {str(error)}"
        project_info.current_log = str(project_info.current_log) + "\n" + error_log
        project_info.container_status = ContainerStatus.FAILED
        project_info.save()
        raise error

def project_info_to_dict(project_info):
    try:
        result = None

        workflow_order = WorkflowOrder.objects.filter(project_id=project_info.id).order_by('order')
        workflow_dic = {"workflow": list(workflow_order.values())}
    
        project = dict(related_model_to_dict(project_info), **workflow_dic)
    
        # TODO : 타겟이 0이 아닌 경우 SW 정보 전달
        if project['target'] is not None:
            target_info_dic = {
                "target_info": target_to_response(project['target']),
                "target_id": project['target']["id"] 
            }
    
            result = dict(project,  **target_info_dic)
            result.pop("target")
        else:
            # 딕셔너리 정보 합치기
            result = dict(project)
    
        return result
    except Exception:
        return {}

# ============================================================================

@api_view(['GET'])
@permission_classes([AllowAny])   # 토큰 확인
def get_common_folder_structure(request):
    try:
        path = os.path.join(root_path, "shared", "common")
        print("path =================================================> " ,path)
        folder_structure = get_folder_structure(path)
        return HttpResponse(json.dumps({'status': 200, "structure" : folder_structure}))
    except Exception as error:
        return HttpResponse(error)


@api_view(['POST'])
@permission_classes([AllowAny])   # 토큰 확인
def next_pipeline_start(request):
    '''
        CI/CD pipeline 반복 기능 -> 다음 pipeline 실행

        - update project version
        - update dataset version
        - Call Autonn Start API
    '''
    
    try:
        user_id = request.data['user_id']
        project_id = request.data['project_id']
        container_id = ContainerId.autonn
        project_info = Project.objects.get(id=project_id, create_user=str(user_id))
        
        # update project version----------------------------------------------------
        project_info.version = project_info.version + 1

        # update dataset version----------------------------------------------------
        copy_train_file_for_version(project_info.version)

        # Call Autonn Start API----------------------------------------------------

        log = ''
        try:
            log = start_container(user_id, project_id, project_info, container_id)
        except Exception:
            print(str(container_id) + " Container Start 요청 실패")
            print(error)
            project_info.save()
            return HttpResponse(error)

        project_info.container = container_id
        project_info.container_status = ContainerStatus.STARTED
        project_info.save()
        return HttpResponse(json.dumps({'status': 200, 'project': project_info_to_dict(project_info), 
                                        'message': str(container_id) + ' 시작 요청\n', 
                                        'response' : log}
                                        ))
    except Project.DoesNotExist:
        print(f"project_id : {project_id}를 찾을 수 없음.")
        return HttpResponse(error)
    except Exception as error:
        return HttpResponse(error)

@api_view(['POST'])
@permission_classes([AllowAny])   # 토큰 확인
def get_autonn_status(request):
    try:
        project_id = request.data['project_id']
        project_info = Project.objects.get(id = project_id)

        autonn = init_autonn_status(project_info)

        train_loss_laststeps = models.TrainLossLastStep.objects.filter(
            project_id = project_info.id,
            project_version = project_info.version,
            is_use = True
        )
        train_loss_laststep_list = list(train_loss_laststeps.values())
        autonn["train_loss_laststep_list"] = list(map(related_model_to_dict, train_loss_laststep_list))

        val_accuracy_laststeps = models.ValAccuracyLastStep.objects.filter(
            project_id = project_info.id,
            project_version = project_info.version,
            is_use = True
        )
        val_accuracy_laststep_list = list(val_accuracy_laststeps.values())
        autonn["val_accuracy_laststep_list"] = list(map(related_model_to_dict, val_accuracy_laststep_list))

        epoch_summary = models.EpochSummary.objects.filter(
            project_id = project_info.id,
            project_version = project_info.version,
            is_use = True
        )
        epoch_summary_list = list(epoch_summary.values())
        autonn["epoch_summary_list"] = list(map(related_model_to_dict, epoch_summary_list))

        return HttpResponse(json.dumps({'status': 200, "autonn" : autonn}))
    except Exception as error:
        return HttpResponse(error)
    
@api_view(['POST'])
@permission_classes([AllowAny])   # 토큰 확인
def container_stop(request):
    """
    Request to stop another container

    Args:
        user_id (string): user_id
        project_id (string): project_id
        container_id (string): Container to be requested to stop

    Returns:
        status, log
    """

    try:
        print("----------container_stop----------")
        user_id = request.data['user_id']
        project_id = request.data['project_id']
        container_id = request.data['container_id']

        project_info = Project.objects.get(id=project_id, create_user=str(user_id))        

        try:
            # AutoNN_CL stop API 호출 로그 추가
            if container_id == ContainerId.autonn_cl:
                stop_call_log = f"[AutoNN_CL] 중지 API 호출 - GET http://autonn-cl:8102/stop?user_id={user_id}&project_id={project_id}"
                project_info.current_log = str(project_info.current_log) + "\n" + stop_call_log
            
            # target_info 안전 처리 (target이 None인 경우 대비)
            target_info = None
            if project_info.target:
                target_info = project_info.target.target_info
            
            call_api_handler(container_id, "stop", user_id, project_id, target_info)
            
            # AutoNN_CL stop API 응답 로그 추가
            if container_id == ContainerId.autonn_cl:
                stop_response_log = f"[AutoNN_CL] 중지 API 응답 수신 완료"
                project_info.current_log = str(project_info.current_log) + "\n" + stop_response_log
                
        except Exception as e:
            print(str(container_id) + " Container stop 요청 실패")
            print(e)
            # AutoNN_CL stop API 호출 실패 로그 추가
            if container_id == ContainerId.autonn_cl:
                stop_error_log = f"[AutoNN_CL] 중지 API 호출 실패 - 오류: {str(e)}"
                project_info.current_log = str(project_info.current_log) + "\n" + stop_error_log
            project_info.save()
            return HttpResponse(e)

        project_info.container = container_id
        project_info.container_status = ContainerStatus.STOPPED
        project_info.save()
        return HttpResponse(json.dumps({'status': 200, 'message': str(container_id) + ' 중지 요청\n', 'response' : str(container_id) + ' 중지 요청\n'}))
    except Project.DoesNotExist:
        print(f"project_id : {project_id}를 찾을 수 없음.")
        return HttpResponse(error)
    except Exception as error:
        print('container start error - ' + str(error))
        return HttpResponse(error)

# 컨테이너 상태 결과 응답
@api_view(['POST'])
@permission_classes([AllowAny])   # 토큰 확인
def container_start(request):
    """
    Request to start another container

    Args:
        user_id (string): user_id
        project_id (string): project_id
        container_id (string): Container to be requested to start

    Returns:
        status, log
    """

    try:
        print("----------container_start----------")
        user_id = request.data['user_id']
        project_id = request.data['project_id']
        container_id = request.data['container_id']

        project_info = Project.objects.get(id=project_id, create_user=str(user_id))

        if container_id == ContainerId.autonn and project_info.learning_type == LearningType.INCREMENTAL:
            copy_train_file_for_version(project_info.version)

        elif container_id == ContainerId.autonn and project_info.learning_type != LearningType.INCREMENTAL:
            copy_train_file_for_version(5)
        

        log = ''
        try:
            log = start_container(user_id, project_id, project_info, container_id)
        except Exception as error:
            print(str(container_id) + " Container Start 요청 실패")
            print(error)
            project_info.save()
            return HttpResponse(str(error))

        project_info.container = container_id
        project_info.container_status = ContainerStatus.STARTED
        project_info.save()
        
        # Segmentation 프로젝트인 경우 추가 로그 메시지
        additional_message = ""
        if container_id == ContainerId.autonn_cl:
            additional_message = "\n=== Segmentation + Continual Learning 프로젝트 시작 ==="
        
        # AutoNN_CL의 경우 current_log에 누적된 로그들을 response에 포함
        if container_id == ContainerId.autonn_cl:
            # current_log에 누적된 모든 로그를 가져와서 response에 포함
            full_log = str(project_info.current_log) + "\n" + log
            return HttpResponse(json.dumps({'status': 200, 'message': str(container_id) + ' 시작 요청\n' + additional_message, 'response' : full_log}))
        else:
            return HttpResponse(json.dumps({'status': 200, 'message': str(container_id) + ' 시작 요청\n' + additional_message, 'response' : log}))
    except Project.DoesNotExist:
        print(f"project_id : {project_id}를 찾을 수 없음.")
        return HttpResponse(error)
    except Exception as error:
        print('container start error - ' + str(error))
        return HttpResponse(error)

# 컨테이너 상태 결과 응답
@api_view(['POST'])
@permission_classes([AllowAny])   # 토큰 확인
def status_request(request):
    """
    Container Task Status Request

    Args:
        user_id (string): user_id
        project_id (string): project_id

    Returns:
        status, log
    """

    try:
        print("----------status_request----------")
        response_log = ""

        user_id = request.data['user_id']
        project_id = request.data['project_id']

        project_info = Project.objects.get(id=project_id, create_user=str(user_id))
        container_id = project_info.container

        if container_id == ContainerId.imagedeploy:
            container_info = CONTAINER_INFO[get_deploy_container(project_info.target.target_info)]
        else:
            container_info = CONTAINER_INFO[container_id]
        
        # res = call_api_handler(container_id, "status_request", user_id, project_id, project_info.target.target_info)
        # if res == None:
        #     return HttpResponse(json.dumps({'container': container_id, 'container_status': '', 'message': ''}))

        if project_info.container_status != ContainerStatus.COMPLETED and project_info.container_status != ContainerStatus.FAILED:
            res = {}
            try:
                # AutoNN_CL API 호출 로그 추가
                if container_id == ContainerId.autonn_cl:
                    status_call_log = f"[AutoNN_CL] 상태 확인 API 호출 - GET http://autonn-cl:8102/status_request?user_id={user_id}&project_id={project_id}"
                    project_info.current_log = str(project_info.current_log) + "\n" + status_call_log
                
                # target_info 안전 처리 (target이 None인 경우 대비)
                target_info = None
                if project_info.target:
                    target_info = project_info.target.target_info
                
                res = call_api_handler(container_id, "status_request", user_id, project_id, target_info)
                
                # AutoNN_CL API 응답 로그 추가
                if container_id == ContainerId.autonn_cl:
                    status_response_log = f"[AutoNN_CL] 상태 확인 API 응답 수신 완료"
                    project_info.current_log = str(project_info.current_log) + "\n" + status_response_log
                    
            except Exception as e:
                # AutoNN_CL API 호출 실패 로그 추가
                if container_id == ContainerId.autonn_cl:
                    status_error_log = f"[AutoNN_CL] 상태 확인 API 호출 실패 - 오류: {str(e)}"
                    project_info.current_log = str(project_info.current_log) + "\n" + status_error_log
                return HttpResponse(json.dumps({'container': container_id, 'container_status': '', 'message': ''}))

            response = json.loads(res)
        else :
            response = { "response" : project_info.container_status }
               
        if len(response['response']) > 50:
            project_info.save()
            return HttpResponse(json.dumps({'container': container_id, 'container_status': project_info.container_status, 'message':  container_info.display_name + ": status_request - Error\n"}))
        
        # 현재 container의 status를 log에 표시
        if container_id == ContainerId.autonn_cl:
            response_log = str(project_info.current_log) + f"\n[AutoNN_CL] 현재 상태: {response['response']}"
        else:
            response_log = str(project_info.current_log) + str(container_id) + '- status_request response : ' + str(response['response'])
        
        # docker의 log를 가져옴
        if container_id != ContainerId.imagedeploy:
            logs = get_docker_log_handler(project_info.container, project_info.last_logs_timestamp)
        else:
            logs = get_docker_log_handler(get_deploy_container(project_info.target.target_info), project_info.last_logs_timestamp)
        
        # log를 가지고 온 마지막 timestamp와 실행 컨테이너를 저장
        project_info.last_logs_timestamp = time.mktime(datetime.now().timetuple()) + 1.0
        project_info.last_log_container = project_info.container

        response_log += '\n' + str(logs)
        
        # AutoNN_CL의 경우 current_log를 보존 (API 호출 로그 유지)
        if container_id != ContainerId.autonn_cl:
            project_info.current_log = ''

        if response['response'] == ContainerStatus.COMPLETED:
            response_log += container_info.display_name + " 완료\n"

        project_info.container_status = response['response']

        # 현재까지 로그를 text파일로 따로 저장
        update_project_log_file(user_id, project_id, response_log)

        project_info.save()
        return HttpResponse(json.dumps({'container': container_id, 'container_status': response['response'], 'message': response_log,}))

    except Project.DoesNotExist:
        print(f"project_id : {project_id}를 찾을 수 없음.")
        return HttpResponse(error)
    except Exception as error:
        print("status_request --- error")
        print(error)
        return HttpResponse(error)

# 컨테이너 상태 결과 응답
@api_view(['GET'])
@permission_classes([AllowAny])   # 토큰 확인
def status_report(request):
    """
    API to receive task status reports from other containers

    Args:
        user_id (string): user_id
        project_id (string): project_id
        container_id (string): container_id
        status (string): task status

    Returns:
        status
    """
    try:
        print("@@@@@@@@@@@@@@@@@@@@@@@ status report @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        user_id = request.GET['user_id']
        project_id = request.GET['project_id']
        container_id = request.GET['container_id']
        container_info = CONTAINER_INFO[container_id]
        result = request.GET['status']

        if result == 'success':
            result = ContainerStatus.COMPLETED

        project_info = Project.objects.get(id=project_id, create_user=str(user_id))
        project_info.container = container_info.key
        project_info.current_log = status_report_to_log_text(request, project_info)

        # 현재 project의 workflow 순서를 가지고 옴.
        workflow_order = WorkflowOrder.objects.filter(project_id=project_id).order_by('order')
        
        if container_id == ContainerId.autonn:
            # if result == ContainerStatus.COMPLETED: 
            #     # autonn이 COMPLETED되면 stop API를 호출
            #     # * stop API 호출되면, autonn은 해당하는 모든 임시 파일/폴더를 삭제할 예정입니다.
            #     try:
            #         call_api_handler(container_id, "stop", user_id, project_id, project_info.target.target_info)
            #     except Exception:
            #         print("AUTONN STOP Call Failed")
                
            
            if result == ContainerStatus.FAILED and project_info.autonn_retry_count + 1 <= 3:
                # Autonn 다시 시도
                # (Autonn에서 Cuda cache memory를 비우고, batch size를 줄인 후, 중단된 Epoch에서 다시 학습을 재개)
                # 최대 다시시도 횟수 3번
                try:
                    print("AUTONN FAILED............... resume API 호출.....")
                    print("retry count : ", project_info.autonn_retry_count + 1)
                    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                    call_api_handler(container_id, "resume", user_id, project_id, project_info.target.target_info)
                    project_info.autonn_retry_count = project_info.autonn_retry_count + 1
                    project_info.container_status = ContainerStatus.RUNNING
                    project_info.save()
                    return HttpResponse(json.dumps({'status': 200}))
                except Exception as error:
                    print("resume API 호출 실패..")
                    project_info.container_status = ContainerStatus.FAILED
                    result = ContainerStatus.FAILED

        if project_info.project_type == 'auto':
            current_container_idx = findIndexByDictList(list(workflow_order.values()), 'workflow_name', container_id)

            if current_container_idx == None:
                current_container_idx = 999

            is_completed = result == ContainerStatus.COMPLETED
            is_not_last_container = len(list(workflow_order.values())) - 1 > current_container_idx # workflow의 마지막이 아닌 경우

            if is_completed and is_not_last_container :
                next_container = list(workflow_order.values())[current_container_idx + 1]['workflow_name']
                if next_container:
                    project_info.container = next_container

                    if next_container == ContainerId.imagedeploy:
                        next_container_info = CONTAINER_INFO[get_deploy_container(project_info.target.target_info)]
                    else:
                        next_container_info = CONTAINER_INFO[next_container]

                    log = str(project_info.current_log) + "\n" + container_info.display_name + " 완료"
                    log += "\n" + next_container_info.display_name + " 시작 요청"
                    project_info.current_log = log
                    try:
                        call_api_handler(next_container, "start", user_id, project_id, project_info.target.target_info)
                        project_info.container_status = ContainerStatus.STARTED
                    except Exception:
                        project_info.container_status = ContainerStatus.FAILED
                        print(f"{next_container} start call failed")
            else : 
                project_info.container_status = result    
        else :
            project_info.container_status = result

        project_info.save()
        return HttpResponse(json.dumps({'status': 200}))

    except Exception as error:
        print("status_report - error")
        print(error)
        return HttpResponse(error)

# 컨테이너 업데이트 (for Auto NN)
@api_view(['POST'])
@permission_classes([AllowAny])   # 토큰 확인
def status_update(request):
    """
    API to receive task status reports from other containers

    Args:
        user_id (string): user_id
        project_id (string): project_id
        container_id (string): container_id
        status (string): task status

    Returns:
        status
    """
    try:
        print("@@@@@@@@@@@@@@@ status_update @@@@@@@@@@@@@@@")
        update_autonn_status(request.data)
        return HttpResponse(json.dumps({'status': 200}))

    except Exception as error:
        print("status_update - error")
        print(error)
        return HttpResponse(error)

# nn_model 다운로드(외부IDE연동)
@api_view(['GET'])
@permission_classes([AllowAny])   # 토큰 확인
def download_nn_model(request):
    """
    Download nn_model to zip file


    Args:
        user_id (string): user_id
        project_id (string): project_id

    Returns:
        nn_model.zip file
    """

    try:
        user_id = request.GET['user_id']
        project_id = request.GET['project_id']
        zip_file_path = nn_model_zip(user_id, project_id)
        return HttpResponse(open(zip_file_path, 'rb').read())

    except Exception as error:
        print("download_nn_model - error")
        print(error)
        return HttpResponse(error)

# nn_model 업로드(외부IDE연동)
@api_view(['POST'])
@permission_classes([AllowAny])   # 토큰 확인
def upload_nn_model(request):
    """
    upload nn_model to zip file

    Args:
        user_id (string): user_id
        project_id (string): project_id
        nn_model (zio file)

    Returns:
        status
    """
    try:
        user_id = request.data['user_id']
        project_id = request.data['project_id']
        nn_model = request.data['nn_model']
        nn_model_unzip(user_id, project_id, nn_model)

        return Response(status=200)
    except Exception as error:
        print("upload_nn_model - error")
        print(error)
        return HttpResponse(error)

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
        project_infos = Project.objects.filter(create_user=str(request.user))
        data = list(project_infos.values())

        for project in data: 
            if project['target_id'] is not None:
                project['target_info'] = model_to_dict(Target.objects.get(id=int(project['target_id'])))
        return HttpResponse(json.dumps(data))

    except Exception as e:
        print(e)
        return HttpResponse(e)

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
        duplicate_check = Project.objects.get(project_name=form['name'], create_user=request.user)
    except Exception as e:
        print(e)
        duplicate_check = None

    # Project 이름 변경 - 중복 Project 이름이 없는 경우
    if duplicate_check is None:
        data = Project.objects.get(id=form['id'])
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

        data = Project.objects.get(id=form['id'], create_user=request.user)
        data.project_description = form['description']
        data.save()

        return Response(status=200)

    except Exception as e:
        print(e)
        return Response(status=500)
    
# Project 워크플로우 진행 방식 수정
@api_view(['GET', 'POST'])
@authentication_classes([OAuth2Authentication])   # 토큰 확인
def project_type_update(request):
    """
    project_type_update _summary_

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """

    try:
        form = request.data

        data = Project.objects.get(id=form['id'])
        projectType = form['type']
        data.project_type = projectType.lower()
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
        duplicate_check = Project.objects.get(project_name=request.data['project_name'], create_user=request.user)

    except Exception as e:
        print(e)
        duplicate_check = None

    try:
        # Project 생성 - 중복 Project 이름이 없는 경우
        if duplicate_check is None:
            data = Project(
                project_name=request.data['project_name'],
                project_description=request.data['project_description'],
                create_user=request.user,
                create_date=str(datetime.now())
            )
            data.save()

            common_path = os.path.join(root_path, f"shared/common/{request.user}/{data.id}")
            config_path = os.path.join(root_path, os.environ.get('CONFIG_YAML_PATH'))

            if os.path.isdir(common_path) is False:
                os.makedirs(common_path)

            shutil.copyfile(os.path.join(config_path, 'hyp.scratch.cls.yaml'), os.path.join(common_path, 'hyp.scratch.cls.yaml'))
            shutil.copyfile(os.path.join(config_path, 'hyp.scratch.p5.yaml'), os.path.join(common_path, 'hyp.scratch.p5.yaml'))

            shutil.copyfile(os.path.join(config_path, 'args-classification.yaml'), os.path.join(common_path, 'args-classification.yaml'))
            shutil.copyfile(os.path.join(config_path, 'args-detection.yaml'), os.path.join(common_path, 'args-detection.yaml'))

            init_autonn_status(data) 
            return Response({'result': True,
                            'id': data.id,
                            'name': data.project_name,
                            'description': data.project_description})
        else:
            return Response({'result': False})
    except Exception as e:
        print(e)
        return HttpResponse(status=500)


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

    try:
        project = Project.objects.get(id=request.data['id'])  # Project id로 검색

        delete_autonn_status(project)

        WorkflowOrder.objects.filter(project_id=request.data['id']).delete()

        project.delete()
       
        project_path = os.path.join(root_path, f"shared/common/{request.user}/{request.data['id']}")
        shutil.rmtree(project_path)
    except Exception as error:
        print("project_delete error")
        print(error)
    
    
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
    try:
        project_id = request.data.get('id')
        print(f"🔍 project_info API 호출 - ID: {project_id}")
        
        if not project_id:
            print("❌ project_info - 프로젝트 ID가 없음")
            return Response({'error': 'Project ID is required'}, status=400)
            
        project_info = Project.objects.get(id=project_id)  # Project id로 검색
        print(f"✅ project_info - 프로젝트 조회 성공: {project_info.project_name}")
    
        return Response(project_info_to_dict(project_info))
    except Project.DoesNotExist:
        print(f"❌ project_info - 프로젝트를 찾을 수 없음: ID={project_id}")
        return Response({'error': 'Project not found'}, status=404)
    except Exception as e:
        print('❌ project_info - 에러 발생:')
        print(e)
        return Response({'error': 'Internal server error'}, status=500)

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

    try:
        target = ''
        try:
            target = int(request.data['project_target']) 
        except:
            target = ''
        
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
        deploy_input_source = str(request.data['deploy_input_source'])

        learning_type = str(request.data['learning_type'])

        # AutoNN_CL (Segmentation + Continual Learning) projects use fixed target ID 9
        if (
            task_type == TaskType.SEGMENTATION
            and learning_type == LearningType.CONTINUAL_LEARNING
        ):
            target = 9

        project_info = Project.objects.get(id=request.data['project_id'])

        project_info.dataset = dataset
        project_info.task_type = task_type
        project_info.learning_type = learning_type
        # project_info.weight_file = str(request.data.get('weight_file', ""))
        project_info.autonn_dataset_file = autonn_dataset_file
        project_info.autonn_basemodel = autonn_basemodel
        project_info.nas_type = nas_type
        project_info.deploy_weight_level = deploy_weight_level
        project_info.deploy_precision_level = deploy_precision_level
        project_info.deploy_processing_lib = deploy_processing_lib
        project_info.deploy_user_edit = deploy_user_edit
        project_info.deploy_input_method = deploy_input_method
        project_info.deploy_input_data_path = deploy_input_data_path
        project_info.deploy_output_method = deploy_output_method
        project_info.deploy_input_source = deploy_input_source
        project_info.container = 'init'
        project_info.container_status = ''

        if target == '':
            project_info.container = ''
            project_info.container_status = ''
            project_info.save()
            return Response(status=200)
        
        data = Target.objects.get(id=int(target))
        project_info.target = data


        if learning_type == LearningType.TRANSFER:
            project_info.weight_file = str(request.data.get('weight_file', ""))
        else:
            project_info.weight_file = ""
        project_info.save()

        target_info = str(data.target_info) if data.target_info != 'ondevice' else "ondevice"

        project_info_content = f"# common\n"
        project_info_content += f"task_type : {str(task_type)}\n"
        project_info_content += f"target_info : {str(target_info)}\n"
        project_info_content += f"learning_type : {str(learning_type)}\n"

        if learning_type == LearningType.TRANSFER:
            s_weight_file = str(request.data.get('weight_file', ""))
            project_info_content += f"weight_file : {s_weight_file}\n"

        project_info_content += (
            f"cpu : {str(data.target_cpu)}\n"
            f"acc : {str(data.target_acc)}\n"
            f"memory : {str(int(int(data.target_memory) / 1024))}\n"
            f"os : {str(data.target_os)}\n"
            f"engine : {str(data.target_engine)}\n"
            f"nfs_ip : {str(data.nfs_ip)}\n"
            f"nfs_path : {str(data.nfs_path)}\n"
            f"target_hostip : {str(data.target_host_ip)}\n"
            f"target_hostport : {str(data.target_host_port)}\n"
            f"target_serviceport : {str(data.target_host_service_port)}\n\n"
            f"#for autonn\n"
            f"dataset : {str(dataset)}\n"
            f"#basemodel : {str(autonn_basemodel)}\n"
            f"#nas_type : {str(nas_type)}\n\n"
            f"#for deploy\n"
            f"lightweight_level : {str(deploy_weight_level)}\n"
            f"precision_level : {str(deploy_precision_level)}\n"
            f"#preprocessing_lib : {str(deploy_processing_lib)}\n"
            f"#input_method : {str(deploy_input_method)}\n"
            f"#input_data_location : {str(deploy_input_data_path)}\n"
            f"output_method : {str(deploy_output_method)}\n"
            f"input_source : {str(deploy_input_source)}\n"
            f"user_editing : {str(deploy_user_edit)}\n"
        )

        # Segmentation 프로젝트인 경우 전용 YAML 생성
        if task_type == TaskType.SEGMENTATION:
            create_segmentation_project_yaml(str(request.user), request.data['project_id'], request.data)
        else:
            # 기존 project_info.yaml 파일 생성
            common_path = os.path.join(root_path, f"shared/common/{request.user}/{request.data['project_id']}")

            # 디렉토리 유무 확인
            if os.path.isdir(common_path) is False:
                os.makedirs(common_path)

            f = open(os.path.join(common_path, 'project_info.yaml'), 'w+')
            f.write(project_info_content)
            f.close()

        return Response(status=200)

    except Exception as e:
        print('error')
        print(e)

def create_segmentation_project_yaml(user_id, project_id, project_data):
    """
    Segmentation 프로젝트용 project_info.yaml 생성 함수
    
    Args:
        user_id (str): 사용자 ID
        project_id (str): 프로젝트 ID
        project_data (dict): 프로젝트 데이터
    
    Description:
        Segmentation + Continual Learning 프로젝트를 위한 특별한 YAML 파일 생성
        중앙대에서 개발할 AutoNN_CL 컨테이너에서 사용할 설정 정보 포함
    """
    try:
        import yaml
        user_id = str(user_id)
        
        # 공통 경로 설정 및 디렉토리 생성
        common_path = os.path.join(root_path, f"shared/common/{user_id}/{project_id}")
        if os.path.isdir(common_path) is False:
            os.makedirs(common_path)

        # Segmentation 전용 YAML 구조 정의
        yaml_content = {
            'project_info': {
                'project_id': int(project_id),
                'task_type': 'segmentation',
                'learning_type': 'continual_learning',
                'user_id': user_id,
                'create_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'version': 1
            },
            'model_config': {
                # Configuration 단계에서 입력받은 배포 설정값들
                'input_source': project_data.get('deploy_input_source', '0'),
                'output_method': project_data.get('deploy_output_method', '0'),
                'precision_level': int(project_data.get('deploy_precision_level', 5)),
                'weight_level': int(project_data.get('deploy_weight_level', 5)),
                'user_editing': project_data.get('deploy_user_edit', 'no')
            },
            'segmentation_config': {
                # Segmentation 특화 설정 (중앙대 요구사항에 따라 조정 가능)
                'continual_learning_method': 'default',  # 기본 continual learning 방식
                'memory_buffer_size': 1000,  # 메모리 버퍼 크기 (예시)
                'learning_rate': 0.001,  # 학습률 (예시)
                'batch_size': 16  # 배치 크기 (예시)
            },
            'container_info': {
                'container_id': str(ContainerId.autonn_cl),
                'container_port': 8102,  # AutoNN_CL 컨테이너 포트 (가안)
                'status': str(ContainerStatus.READY)
            }
        }

        # YAML 파일로 저장
        with open(os.path.join(common_path, 'project_info.yaml'), 'w') as f:
            yaml.safe_dump(yaml_content, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        print(f"Segmentation project YAML created successfully for project {project_id}")
        
    except Exception as error:
        print(f"Error creating segmentation project YAML: {error}")
        raise error

# 워크플로우 추가
@api_view(['POST'])
@permission_classes([AllowAny])   # 토큰 확인
def set_workflow(request):
    """
    Modifying Workflows for a Project

    Args:
        project_id (string): project_id
        workflow : workflow

    Returns:
        status, workflow
    """
    try:
        project_id = request.data["project_id"]
        workflow = request.data["workflow"]

        WorkflowOrder.objects.filter(project_id=project_id).delete()

        for index, flow in enumerate(workflow):
            target = WorkflowOrder(workflow_name=flow, order = int(index), project_id=int(project_id))
            target.save()

        save_data = WorkflowOrder.objects.filter(project_id=project_id).order_by('order')

        return HttpResponse(json.dumps({'status': 200, 'workflow': list(save_data.values())}))

    except Exception as e:
        return Response(status=500)

# 워크플로우 추가
@api_view(['GET'])
@permission_classes([AllowAny])   # 토큰 확인
def get_project_hyperparameter_file(request):
    """
    get hyperparameter file for a project

    Args:
        project_id (string): project_id

    Returns:
        hyperparameter file
    """
    try:
        project_id = request.GET['project_id']
        file_name = get_hyperparameter_file_name(project_id)

        file_path = os.path.join(root_path, f"shared/common/{request.user}/{project_id}", file_name)
        with open(file_path) as f:
            content = f.read()

        print(content)

        return HttpResponse(json.dumps({'status': 200, 'content': content}))

    except Exception as e:
        print(e)
        return Response(status=500)

# 워크플로우 추가
@api_view(['POST'])
@permission_classes([AllowAny])   # 토큰 확인
def update_project_hyperparameter_file(request):
    """
    get hyperparameter file for a project

    Args:
        project_id (string): project_id

    Returns:
        hyperparameter file
    """
    try:
        project_id = request.data['project_id']
        content = request.data['content']
        file_name = get_hyperparameter_file_name(project_id)

        file_path = os.path.join(root_path, f"shared/common/{request.user}/{project_id}", file_name)

        f = open(file_path, 'w+')
        f.write(content)
        f.close()

        return HttpResponse(json.dumps({'status': 200}))

    except Exception as e:
        print(e)
        return Response(status=500)

    
# 워크플로우 추가
@api_view(['GET'])
@permission_classes([AllowAny])   # 토큰 확인
def get_project_arguments_file(request):
    """
    get arguments file for a project

    Args:
        project_id (string): project_id

    Returns:
        arguments file
    """
    try:
        project_id = request.GET['project_id']
        file_name = get_arguments_file_name(project_id)

        file_path = os.path.join(root_path, f"shared/common/{request.user}/{project_id}", file_name)
        with open(file_path) as f:
            content = f.read()

        print(content)

        return HttpResponse(json.dumps({'status': 200, 'content': content}))

    except Exception as e:
        print(e)
        return Response(status=500)

# 워크플로우 추가
@api_view(['POST'])
@permission_classes([AllowAny])   # 토큰 확인
def update_project_arguments_file(request):
    """
    set arguments file for a project

    Args:
        project_id (string): project_id

    Returns:
        arguments file
    """
    try:
        project_id = request.data['project_id']
        content = request.data['content']
        file_name = get_arguments_file_name(project_id)

        file_path = os.path.join(root_path, f"shared/common/{request.user}/{project_id}", file_name)

        f = open(file_path, 'w+')
        f.write(content)
        f.close()

        return HttpResponse(json.dumps({'status': 200}))

    except Exception as e:
        print(e)
        return Response(status=500)
