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

from PIL import Image

import cv2
import base64

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(os.path.dirname(BASE_DIR))

def get_folder_info(folder_path):
    """
    Returns the folder info

    Args:
        folder_path (string): folder_path

    Returns:
        folder name, folder path, folder size, folder creation_time, 
        folder last_modified_time, file_count, thumbnail
    """
    folder_info = {}

    # 폴더 경로가 유효한지 확인합니다.
    if os.path.isdir(folder_path):
        folder_info['name'] = os.path.basename(folder_path)
        folder_info['path'] = folder_path
        folder_info['size'] = get_folder_size(folder_path)
        folder_info['creation_time'] = get_folder_creation_date(folder_path)
        folder_info['last_modified_time'] = get_folder_last_modified_date(folder_path)
        folder_info['file_count'] = get_file_count(folder_path)
        folder_info['thumbnail'] = get_folder_thumbnail(folder_path)
    else:
        print("유효한 폴더 경로가 아닙니다.")

    return folder_info

def get_folder_size(folder_path):
    """
    Returns the folder size

    Args:
        folder_path (string): folder_path

    Returns:
        folder size
    """
    total_size = 0
    for path, dirs, files in os.walk(folder_path):
        for f in files:
            file_path = os.path.join(path, f)
            total_size += os.path.getsize(file_path)
    return total_size

def get_folder_creation_date(folder_path):
    """
    Returns the creation date

    Args:
        folder_path (string): folder_path

    Returns:
        creation date
    """
    creation_time = os.path.getctime(folder_path)
    formatted_creation_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(creation_time))
    return formatted_creation_time

def get_folder_last_modified_date(folder_path):
    """
    Returns the last modification date

    Args:
        folder_path (string): folder_path

    Returns:
        last modification date
    """
    modified_time = os.path.getmtime(folder_path)
    formatted_modified_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(modified_time))
    return formatted_modified_time

def get_file_count(folder_path):
    """
    Returns the number of files

    Args:
        folder_path (string): folder_path

    Returns:
        int : file count
    """
    file_count = 0
    for path, dirs, files in os.walk(folder_path):
        file_count += len(files)
    return file_count

def get_folder_thumbnail(folder_path):
    """
    Create thumbnails after randomly extracting 4 images from a folder

    Args:
        folder_path (string): Image folder path

    Returns:
        Returns Thumbnails to base64
    """

    file_list = []

    # get image path in folder
    for path, dirs, files in os.walk(folder_path):
        images = [ fi for fi in files if fi.endswith(('.jpg','.jpeg', '.png')) ]
        for image in images:
            file_list.append(os.path.join(path, image))

    # random choice
    if len(file_list) > 0:
        random_images = random.sample(file_list,4) if len(file_list) >= 4  else random.sample(file_list, len(file_list))
        thumbnail_list = []
        for image in random_images:
            thumbnail_list.append(make_image_thumbnail(image))
        thumb = cv2.hconcat(thumbnail_list)
        jpg_img = cv2.imencode('.jpg', thumb)
        return "data:image/jpg;base64," + str(base64.b64encode(jpg_img[1]).decode('utf-8'))
            
    else :
        return None

def make_image_thumbnail(path):
    """
    Create Thumbnails

    Args:
        path (string): File path to create thumbnails

    Returns:
        Thumbnails
    """
    maxsize = (128, 128) 
    img = cv2.imread(path, 1);
    thumbnail = cv2.resize(img, maxsize, interpolation=cv2.INTER_AREA)
    return thumbnail

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