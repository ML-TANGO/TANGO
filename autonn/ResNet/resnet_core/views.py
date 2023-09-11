"""autonn/ResNet/resnet_core/views.py
"""

import os
import json
import torch
import requests
import shutil
import multiprocessing
import yaml
import random

from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.response import Response
from rest_framework.decorators import api_view
from pathlib import Path
from distutils.dir_util import copy_tree
import argparse

# from .yolov7_utils.train import run_yolo
from . import models


PROCESSES = {}


def index(request):
    """index"""
    return render(request, "resnet_core/index.html")


@api_view(["GET", "POST"])
def InfoList(request):
    """Information List for Neck NAS"""
    if request.method == "POST":
        # Fetching the form data
        uploadedFile = request.FILES["data_yaml"]
        usrId = request.data["user_id"]
        prjId = request.data["project_id"]
        target = request.data["target"]
        task = request.data["task"]
        sts = request.data["status"]
        prcId = request.data["process_id"]

        # Saving the information in the database
        updatedInfo = models.Info(
            userid=usrId,
            project_id=prjId,
            target_device=target,
            data_yaml=uploadedFile,
            task=task,
            status=sts,
            process_id=prcId,
        )
        updatedInfo.save()

        return render(request, "resnet_core/index.html")


@api_view(["GET"])
def start(request):
    print("_________GET /start_____________")
    params = request.query_params
    userid = params["user_id"]
    project_id = params["project_id"]

    # check user id & project id
    try:
        nasinfo = models.Info.objects.get(userid=userid, project_id=project_id)
    except models.Info.DoesNotExist:
        print("new user or project")
        nasinfo = models.Info(userid=userid, project_id=project_id)

    data_yaml, proj_yaml = get_user_requirements(userid, project_id)
    print(data_yaml, proj_yaml)

    pr = multiprocessing.Process(target=process_resnet, args=(userid, project_id, data_yaml, proj_yaml))
    pr_id = get_process_id()
    PROCESSES[pr_id] = pr
    print(f"{len(PROCESSES)}-th process is starting")
    PROCESSES[pr_id].start()

    nasinfo.target_device = str(proj_yaml)
    nasinfo.data_yaml = str(data_yaml)
    nasinfo.status = "started"
    nasinfo.process_id = pr_id
    print(
        "PROCESS ID TYPE CHECK(before save): ",
        type(nasinfo.process_id),
        nasinfo.process_id,
    )
    nasinfo.save()
    print(
        "PROCESS ID TYPE CHECK(after save) : ",
        type(nasinfo.process_id),
        nasinfo.process_id,
    )
    return Response("started", status=200, content_type="text/plain")


@api_view(["GET"])
def stop(request):
    print("_________GET /stop_____________")
    params = request.query_params
    userid = params["user_id"]
    project_id = params["project_id"]

    try:
        nasinfo = models.Info.objects.get(userid=userid, project_id=project_id)
    except models.Info.DoesNotExist:
        print("no such user or project...")
        return Response("failed", status=200, content_type="text/plain")

    PROCESSES[str(nasinfo.process_id)].terminate()
    nasinfo.status = "stopped"
    nasinfo.save()
    return Response("stopped", status=200, content_type="text/plain")


@api_view(["GET"])
def status_request(request):
    print("_________GET /status_request_____________")
    params = request.query_params
    userid = params["user_id"]
    project_id = params["project_id"]
    print("user_id: ", userid)
    print("project_id: ", project_id)
    print("------------------------------------------")

    try:
        nasinfo = models.Info.objects.get(userid=userid, project_id=project_id)
    except models.Info.DoesNotExist:
        print("new user or project")
        nasinfo = models.Info(userid=userid, project_id=project_id)
        nasinfo.status = "ready"
        nasinfo.save()
        return Response("ready", status=200, content_type="text/plain")

    try:
        if PROCESSES[str(nasinfo.process_id)].is_alive():
            print("found thread running ResNet")
            nasinfo.status = "running"
            nasinfo.save()
            return Response("running", status=200, content_type="text/plain")
        else:
            print("tracked ResNet process you want, but not running anymore")
            if nasinfo.status == "running":
                nasinfo.status = "failed"
                nasinfo.save()
            print(f"nasinfo.status: {nasinfo.status}")
            return Response(nasinfo.status, status=200, content_type="text/plain")
    except KeyError:
        return Response("ready", status=200, content_type="text/plain")


def get_user_requirements(userid, projid):
    common_root = Path("/shared/common/")
    proj_path = common_root / userid / projid
    proj_yaml_path = proj_path / "project_info.yaml"  # 'target.yaml'

    ##### Changed Code #####
    with open(proj_yaml_path, "r") as f:
        proj_info = yaml.safe_load(f)
    dataset_on_proj = proj_info["dataset"]
    if os.path.isdir("/shared/datasets/" + dataset_on_proj):
        dataset_yaml_path = Path("/shared/datasets/") / dataset_on_proj / "dataset.yaml"
    else:
        print(f"There is no /shared/datasets/{dataset_on_proj}. Instead COCO dataset will be used.")
        dataset_yaml_path = Path("/shared/datasets/coco/") / "dataset.yaml"

    return dataset_yaml_path, proj_yaml_path


def status_report(userid, project_id, status="success"):
    try:
        url = "http://projectmanager:8092/status_report"
        headers = {"Content-Type": "text/plain"}
        payload = {
            "container_id": "autonn_resnet",
            "user_id": userid,
            "project_id": project_id,
            "status": status,
        }
        response = requests.get(url, headers=headers, params=payload)
        print(response.text)

        nasinfo = models.Info.objects.get(userid=userid, project_id=project_id)
        nasinfo.status = "completed"
        nasinfo.save()
        print(f"status_report function:{nasinfo.status}")
    except BaseException as e:
        print(e)


def process_resnet(userid, project_id, data_yaml, proj_yaml):
    try:
        common_root = Path("/shared/common/")
        proj_path = os.path.dirname(proj_yaml)

        with open(proj_yaml, "r") as f:
            proj_info = yaml.safe_load(f)
        print(proj_info)

        status_report(userid, project_id, status="success")
        print("process_resnet ends")
    except ValueError as e:
        print(e)

@api_view(["GET"])
def get_ready_for_test(request):
    try:
        print("_______GET /get_ready_for_test________")
        params = request.query_params
        userid = params["user_id"]
        project_id = params["project_id"]

        print("user_id: ", userid)
        print("project_id: ", project_id)
        print("------------------------------------------")

        return Response("ready_for_v7_test", status=200, content_type="text/plain")
        # check user id & project id
        try:
            nasinfo = models.Info.objects.get(userid=userid, project_id=project_id)
        except models.Info.DoesNotExist:
            print("new user or project")
            nasinfo = models.Info(userid=userid, project_id=project_id)

        common_root = Path("/shared/common/")
        proj_path = common_root / userid / project_id

        Path(proj_path).mkdir(parents=True, exist_ok=True)
        Path("/shared/datasets/").mkdir(parents=True, exist_ok=True)

        shutil.copy("sample_yaml/project_info.yaml", proj_path)
        shutil.copytree("sample_data/coco128", Path("/shared/") / "datasets" / "coco")
        with open("sample_yaml/dataset.yaml") as f:
            data_yaml = yaml.load(f, Loader=yaml.FullLoader)
        data_yaml["train"] = str(Path("/shared/") / "datasets" / "coco" / "images" / "train2017")
        data_yaml["test"] = str(Path("/shared/") / "datasets" / "coco" / "images" / "train2017")
        data_yaml["val"] = str(Path("/shared/") / "datasets" / "coco" / "images" / "train2017")
        with open(Path("/shared/datasets/coco/") / "dataset.yaml", "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        return Response("ready_for_v7_test", status=200, content_type="text/plain")
    except Exception as e:
        print(e)


# def make_directory(path_list):
#     path = Path('')
#     for path_temp in path_list:
#         path = path / path_temp
#         if not os.path.isdir(path):
#             os.mkdir(path)


def get_process_id():  # Assign Blank Process Number
    while True:
        pr_num = str(random.randint(100000, 999999))
        try:
            temp = PROCESSES[pr_num]
        except KeyError:
            break
    return pr_num
