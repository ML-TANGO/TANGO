import json, yaml
import multiprocessing as mp
import random
from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.decorators import api_view
from rest_framework.response import Response
import requests
from . import models
from .tango.main.select import run_autonn

PROCESSES = {}

@api_view(['GET', 'POST'])
def InfoList(request):
    '''
        General information list
    '''
    if request.method == 'GET':
        infoList = models.Info.objects.all()
        return Response(infoList, status=HTTP_200_OK)

    elif request.method == 'POST':
        # Fetching the form data
        uploadedFile = request.FILES["data_yaml"]
        usrId = request.data['user_id']
        prjId = request.data['project_id']
        target = request.data['target']
        task = request.data['task']
        sts = request.data['status']
        prcId = request.data['process_id']

        # Saving the information in the database
        updatedInfo = models.Info(
            userid=usrId,
            project_id=prjId,
            target_device=target,
            data_yaml=uploadedFile,
            task=task,
            status=sts,
            process_id=prcId
        )
        updatedInfo.save()

        return Response("created", status=status.HTTP_201_CREATED)


@api_view(['GET'])
def start(request):
    """
        API for project manager having autonn start
    """
    # print("_________GET /start_____________")
    params = request.query_params
    userid = params['user_id']
    project_id = params['project_id']

    try:
        info = models.Info.objects.get(userid=userid, project_id=project_id)
        if info.status in ['started', 'running']:
            # duplicate project
            return Response("failed", status=status.HTTP_406_NOT_ACCEPTABLE, content_type="text/plain")
    except models.Info.DoesNotExist:
        # new project
        info = models.Info(userid=userid, project_id=project_id)

    try:
        # data_yaml, proj_yaml = get_user_requirements(userid, project_id)

        pr = mp.Process(target = process_autonn, args=(userid, project_id))
        pr_id = get_process_id()
        PROCESSES[pr_id] = pr
        PROCESSES[pr_id].start()

        # info.target_device=str(proj_yaml)
        # info.data_yaml=str(data_yaml)
        info.status="started"
        info.process_id = pr_id
        info.save()
        return Response("started", status=status.HTTP_200_OK, content_type="text/plain")
    except Exception as e:
        print(f"[AutoNN GET/start] exception: {e}")
        info.status="failed"
        info.save()
        return Response("failed", status=status.HTTP_400_BAD_REQUEST, content_type="text/plain")


@api_view(['GET'])
def status_request(request):
    """
        API for project manager pooling autonn status
    """
    # print("_________GET /status_request_____________")
    params = request.query_params
    userid = params['user_id']
    project_id = params['project_id']

    try:
        info = models.Info.objects.get(userid=userid,
                                          project_id=project_id)
    except models.Info.DoesNotExist:
        # empty project
        return Response("ready", status=status.HTTP_204_NO_CONTENT, content_type='text/plain')

    try:
        if PROCESSES[str(info.process_id)].is_alive():
            # the project is running
            info.status = "running"
            info.save()
            # print("_____running_______")
            return Response("running", status=status.HTTP_200_OK, content_type='text/plain')
        else:
            # the project is not running
            if info.status == "completed":
                # print("_____completed_______")
                return Response("completed", status=status.HTTP_208_ALREADY_REPORTED, content_type='text/plain')
            else:
                info.status = "failed"
                info.save()
                # print("_____failed(dead process)_______")
                return Response("failed", status=status.HTTP_410_GONE, content_type='text/plain')
    except KeyError as e:
        print(f"[AutoNN GET/status_request] exception: {e}")
        info.status = "failed"
        info.save()
        # print("_____failed(empty process)_______")
        return Response("failed", status=status.HTTP_400_BAD_REQUEST, content_type='text/plain')


def status_report(userid, project_id, status="success"):
    """
        Report status to project manager when the autonn process ends
    """
    try:
        url = 'http://projectmanager:8085/status_report'
        headers = {
            'Content-Type' : 'text/plain'
        }
        payload = {
            'container_id' : "autonn",
            'user_id' : userid,
            'project_id' : project_id,
            'status' : status
        }
        response = requests.get(url, headers=headers, params=payload)

        info = models.Info.objects.get(userid=userid, project_id=project_id)
        info.status = status
        # process_done = PROCESSES.pop(str(info.process_id))
        # process_done.close()
        # info.process_id = ''
        info.save()
    except Exception as e:
        print(f"[AutoNN status_report] exception: {e}")


def process_autonn(userid, project_id):
    '''
        select basemodel
        run autonn (setup - train - nas - hpo)
        export weights
        export neural net info
        status report
    '''
    try:
        # ------- actual process --------
        fanal_model = run_autonn(userid, project_id, viz2code="False", nas="False", hpo="False")
        # export_model(final_model, userid, project_id)
        # export_nn_info(userid, project_id)
        # status_report(userid, project_id, "completed")

        # ------- temp for test ---------
        # import time
        # time.sleep(15)
        status_report(userid, project_id, "completed")
        return
    except Exception as e:
        print(f"[AutoNN process_autonn] exception: {e}")
        status_report(userid, project_id, "failed")



def get_process_id():
    """
        Assign a new random number into a process
    """
    while True:
        pr_num = str(random.randint(100000, 999999))
        try:
            temp = PROCESSES[pr_num]
        except KeyError:
            break
    return pr_num
