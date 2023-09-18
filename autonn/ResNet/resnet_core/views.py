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

from . import models
from .resnet_utils import train

from typing import Dict, List, Tuple

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
        url = "http://projectmanager:8085/status_report"
        headers = {"Content-Type": "text/plain"}
        payload = {
            "container_id": "autonn-resnet",  # 추후 autonn_resnet로 변경 필요
            "user_id": userid,
            "project_id": project_id,
            "status": status,
        }
        response = requests.get(url, headers=headers, params=payload)

        nasinfo = models.Info.objects.get(userid=userid, project_id=project_id)
        nasinfo.status = "completed"
        nasinfo.save()
        print(f"status_report function:{nasinfo.status}")
    except BaseException as e:
        print(e)


def process_resnet(userid, project_id, data_yaml, proj_yaml):
    try:
        proj_path = Path(proj_yaml).parent
        Path(proj_path).mkdir(parents=True, exist_ok=True)

        with open(proj_yaml, "r") as f:
            proj_info = yaml.safe_load(f)
        with open(proj_path / "basemodel.yaml", "r") as f:
            basemodel_info = yaml.safe_load(f)

        final_model = train.run_resnet(proj_path, data_yaml)
        if not Path("/source/pretrained/kagglecxr_resnet152_normalize.pt").exists():
            final_model = "/source/pretrained/kagglecxr_resnet152_normalize.pt"
        print("process_resnet: train done")

        best_pt_path = str(Path(proj_path) / "resnet.pt")
        if Path(final_model).exists():
            Path(best_pt_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(final_model, best_pt_path)
            os.remove(final_model)
            print(f"saved the best model: {str(best_pt_path)}")

        src_yaml_root = Path("/source/sample_yaml/")
        prjct_path = Path("/shared/common/") / userid / project_id
        src_info_path = src_yaml_root / "neural_net_info.yaml"
        final_info_path = prjct_path / "neural_net_info.yaml"
        input_shape = basemodel_info.get("input_size", [256])[0]
        create_nn_info(src_info_path, final_info_path, best_pt_path, [input_shape, input_shape])

        exp_num = exp_num_check(proj_path)
        shutil.copy(proj_yaml, Path(proj_path) / str("exp" + str(exp_num) + "_project_info.yaml"))

        print("process_resnet: ends")

        autogen_resnet(userid, project_id)
        status_report(userid, project_id, status="success")
    except ValueError as e:
        print(e)


def autogen_resnet(userid, project_id):
    print("autogen_resnet: start")
    project_root = Path("/shared/common/") / userid / project_id
    source_root = Path("/source/") / "resnet_core"
    nn_model_path = project_root / "nn_model"

    nn_model_path.mkdir(parents=True, exist_ok=True)
    (nn_model_path / "models").mkdir(parents=True, exist_ok=True)
    model_file = source_root / "pretrained/kagglecxr_resnet152_normalize.pt"
    shutil.copy(model_file, nn_model_path / "resnet.pt")
    shutil.copy(project_root / "basemodel.yaml", nn_model_path / "basemodel.yaml")
    shutil.copy(source_root / "resnet_utils/inference.py", nn_model_path / "inference.py")
    shutil.copy(source_root / "resnet_utils/models/resnet_cifar10.py", nn_model_path / "models/resnet_cifar10.py")
    print("autogen_resnet: ends")
    return


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
    except Exception as e:
        print(e)


def create_nn_info(src_info_path: str, final_info_path: str, final_pt_path: str, input_shape: Tuple[int, int]):
    nn_info = dict()
    with open(src_info_path) as f:
        nn_yaml = yaml.load(f, Loader=yaml.FullLoader)
    py_list = ["models/resnet_cifar10.py"]
    for k in nn_yaml.keys():
        if type(nn_yaml[k]) == str:
            nn_info[str(k)] = str(nn_yaml[k])
        else:
            nn_info[str(k)] = nn_yaml[k]
    nn_info["class_name"] = str("ResNet.load_config('basemodel.yaml')")
    nn_info["weight_file"] = str("resnet.pt")
    nn_info["input_tensor_shape"] = [1, 1, input_shape[0], input_shape[1]]
    with open(final_info_path, "w") as file:
        yaml.dump(nn_info, file, default_flow_style=False)


def exp_num_check(proj_path):
    current_filelist = os.listdir(proj_path)
    exp_num_list = []
    for filename in current_filelist:
        if "exp" in filename[:3]:
            exp_num_list.append(int(filename.split("_")[0][3:]))
    if len(exp_num_list) == 0:
        return 0
    else:
        return max(exp_num_list) + 1


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
