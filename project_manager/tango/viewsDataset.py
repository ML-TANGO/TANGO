import os
import glob
import json
import time
import math
import random
import socket
import threading
import requests

from datetime import datetime

import django.middleware.csrf
from django.http import HttpResponse

from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from oauth2_provider.contrib.rest_framework import OAuth2Authentication

from .models import Target

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(os.path.dirname(BASE_DIR))

def get_folder_info(folder_path):
    folder_info = {}

    # 폴더 경로가 유효한지 확인합니다.
    if os.path.isdir(folder_path):
        folder_info['name'] = os.path.basename(folder_path)
        folder_info['path'] = folder_path
        folder_info['size'] = get_folder_size(folder_path)
        folder_info['creation_time'] = get_folder_creation_date(folder_path)
        folder_info['last_modified_time'] = get_folder_last_modified_date(folder_path)
        folder_info['file_count'] = get_file_count(folder_path)
    else:
        print("유효한 폴더 경로가 아닙니다.")

    return folder_info

def get_folder_size(folder_path):
    total_size = 0
    for path, dirs, files in os.walk(folder_path):
        for f in files:
            file_path = os.path.join(path, f)
            total_size += os.path.getsize(file_path)
    return total_size

def get_folder_creation_date(folder_path):
    creation_time = os.path.getctime(folder_path)
    formatted_creation_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(creation_time))
    return formatted_creation_time

def get_folder_last_modified_date(folder_path):
    modified_time = os.path.getmtime(folder_path)
    formatted_modified_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(modified_time))
    return formatted_modified_time

def get_file_count(folder_path):
    file_count = 0
    for path, dirs, files in os.walk(folder_path):
        file_count += len(files)
    return file_count


# dataset list get -> /shared/datasets 경로의 폴더 list
@api_view(['GET'])
@authentication_classes([OAuth2Authentication])   # 토큰 확인
def get_dataset_list(request):
    """
    get_dataset_list _summary_

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """

    try:
        path = os.path.join(root_path, "shared/datasets/*")
        # try:
        #     if not os.path.exists(path):
        #         datasetPath = path = os.path.join(root_path, "shared/datasets/")
        #         os.makedirs(datasetPath)
        # except OSError:
        #     print("Error: Failed to create the directory.")

        dir_list = glob.glob(path)
        
        dir_info_list = []
        for dir_path in dir_list:
            name = os.path.basename(dir_path)
            if name != '*' :
                info = get_folder_info(dir_path)
                dir_info_list.append(info)

        return HttpResponse(json.dumps({'status': 200,
                                        'datasets': dir_info_list }))
        # return HttpResponse(json.dumps({'status': 200}))
    except Exception as e:
        print(e)