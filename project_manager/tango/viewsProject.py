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

from .enums import ContainerId, ContainerStatus


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

# ============================================================================

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

        # autonn을 시작할때 setting
        # 1. 이전에 진행했던 이력 제거 
        # 2. retry count 초기화
        if container_id == ContainerId.autonn:
            delete_autonn_status(project_info) # 이전에 진행했던 이력 제거 
            project_info.autonn_retry_count = 0 # retry count 초기화
            init_autonn_status(project_info) #  새로운 autonn_status 생성

        response = None
        try:
            response = call_api_handler(container_id, "start", user_id, project_id, project_info.target.target_info)
        except Exception as error:
            print(str(container_id) + " Container Start 요청 실패")
            print(error)
            project_info.save()
            return HttpResponse(error)
        
        to_json = json.loads(response)

        project_info.container = container_id
        project_info.container_status = ContainerStatus.STARTED
        project_info.save()
        return HttpResponse(json.dumps({'status': 200, 'message': str(container_id) + ' 시작 요청\n', 'response' : to_json['request_info']}))
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
        
        res = call_api_handler(container_id, "status_request", user_id, project_id, project_info.target.target_info)
        if res == None:
            return HttpResponse(json.dumps({'container': container_id, 'container_status': '', 'message': ''}))

        response = json.loads(res)
               
        if len(response['response']) > 50:
            project_info.save()
            return HttpResponse(json.dumps({'container': container_id, 'container_status': project_info.container_status, 'message':  container_info.display_name + ": status_request - Error\n"}))
        
        # 현재 container의 status를 log에 표시
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
        project_info.current_log = ''

        if project_info.container_status == ContainerStatus.COMPLETED:
            response_log += container_info.display_name + " 완료\n"
            response['response'] = ContainerStatus.COMPLETED

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

        project_info = Project.objects.get(id=project_id, create_user=str(user_id))
        project_info.container = container_info.key
        project_info.current_log = status_report_to_log_text(request, project_info)

        # 현재 project의 workflow 순서를 가지고 옴.
        workflow_order = WorkflowOrder.objects.filter(project_id=project_id).order_by('order')
        if container_id == ContainerId.autonn:
            if result == ContainerStatus.COMPLETED: 
                # autonn이 COMPLETED되면 stop API를 호출
                # * stop API 호출되면, autonn은 해당하는 모든 임시 파일/폴더를 삭제할 예정입니다.
                call_api_handler(container_id, "stop", user_id, project_id, project_info.target.target_info)
            
            elif result == ContainerStatus.FAILED and project_info.autonn_retry_count + 1 <= 3:
                # Autonn 다시 시도
                # (Autonn에서 Cuda cache memory를 비우고, batch size를 줄인 후, 중단된 Epoch에서 다시 학습을 재개)
                # 최대 다시시도 횟수 3번
                try:
                    call_api_handler(container_id, "resume", user_id, project_id, project_info.target.target_info)
                    project_info.autonn_retry_count = project_info.autonn_retry_count + 1
                    project_info.container_status = ContainerStatus.RUNNING
                    project_info. save()
                    return HttpResponse(json.dumps({'status': 200}))
                except Exception as error:
                    print("resume API 호출 실패..")
                    project_info.container_status = ContainerStatus.FAILED

        if project_info.project_type == 'auto':
            current_container_idx = findIndexByDictList(list(workflow_order.values()), 'workflow_name', container_id)

            is_completed = result == ContainerStatus.COMPLETED
            is_valid_idx = current_container_idx != None
            is_not_last_container = len(list(workflow_order.values())) - 1 > current_container_idx # workflow의 마지막이 아닌 경우

            if is_completed and is_valid_idx and is_not_last_container :
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
                    call_api_handler(next_container, "start", user_id, project_id, project_info.target.target_info)
                    project_info.container_status = ContainerStatus.STARTED

        else:
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

    # Project 생성 - 중복 Project 이름이 없는 경우
    if duplicate_check is None:
        data = Project(
            project_name=request.data['project_name'],
            project_description=request.data['project_description'],
            create_user=request.user,
            create_date=str(datetime.now())
        )
        data.save()

        init_autonn_status(data) 
        return Response({'result': True,
                         'id': data.id,
                         'name': data.project_name,
                         'description': data.project_description})
    else:
        return Response({'result': False})

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
        result = None

        project_info = Project.objects.get(id=request.data['id'])  # Project id로 검색
        
        workflow_order = WorkflowOrder.objects.filter(project_id=request.data['id']).order_by('order')
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

        return Response(result)
    except Exception as e:
        print('error - project_info-=============')
        print(e)

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

        project_info = Project.objects.get(id=request.data['project_id'])

        project_info.dataset = dataset
        project_info.task_type = task_type
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
        project_info.save()

        target_info = str(data.target_info) if data.target_info != 'ondevice' else "ondevice"
        project_info_content = (
            f"# common\n"
            f"task_type : {str(task_type)}\n"
            f"target_info : {str(target_info)}\n"
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

        # project_info.yaml 파일 생성
        common_path = os.path.join(root_path, f"shared/common/{request.user}/{request.data['project_id']}")

        # 디렉토리 유뮤 확인
        if os.path.isdir(common_path) is False:
            os.makedirs(common_path)

        f = open(os.path.join(common_path, 'project_info.yaml'), 'w+')
        f.write(project_info_content)
        f.close()

        return Response(status=200)

    except Exception as e:
        print('error')
        print(e)

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
    
