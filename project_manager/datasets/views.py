import os
import glob
import json
import time
import random

from django.http import HttpResponse

from rest_framework.decorators import api_view, permission_classes, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated

from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import base64

import multiprocessing
import threading
import time
import zipfile
import shutil

from .enums import DATASET_STATUS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(os.path.dirname(BASE_DIR))

RUN_COCO_THREAD="RUN_COCO_THREAD"
RUN_IMAGENET_THREAD="RUN_IMAGENET_THREAD"
RUN_VOC_THREAD="RUN_VOC_THREAD"
RUN_CHEST_XRAY_THREAD="RUN_CHEST_XRAY_THREAD"

COMMON_DATASET_INFO = {
    "COCO": {
        "name": "coco",
        "path": os.path.join(root_path, "shared/datasets/coco"),
        "thread_name": RUN_COCO_THREAD,
        "script_path": os.path.join(BASE_DIR, "download_scripts", "get_coco.sh")
    },
    "IMAGE_NET": {
        "name": "imagenet",
        "path": os.path.join(root_path, "shared/datasets/imagenet"),
        "thread_name": RUN_IMAGENET_THREAD,
        "script_path": os.path.join(BASE_DIR, "download_scripts", "get_imagenet.sh")
    },
    "VOC": {
        "name": "VOC",
        "path": os.path.join(root_path, "shared/datasets/VOC"),
        "thread_name": RUN_VOC_THREAD,
        "script_path": os.path.join(BASE_DIR, "download_scripts", "get_voc.sh")
    },
    "CHEST_XRAY": {
        "name": "ChestXRay",
        "path": os.path.join(root_path, "shared/datasets/ChestXRay"),
        "thread_name": RUN_CHEST_XRAY_THREAD
    }
}

#region get dataset ......................................................................................

# dataset list get -> /shared/datasets 경로의 폴더 list
@api_view(['GET'])
@permission_classes([IsAuthenticated])   # 토큰 확인
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
        dir_list = glob.glob(path)

        dataset_dir_list = []        
        for dir_path in dir_list:
            name = os.path.basename(dir_path)
            if name != '*' :
                dataset_dir_list.append(dir_path)

        dir_info_list = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(get_folder_info, dir_path) for dir_path in dataset_dir_list]
            
            for future in as_completed(futures):
                result = future.result()
                if result != None :
                    dir_info_list.append(result)

        dir_info_list = sorted(dir_info_list, key= lambda x: x["name"])
        return HttpResponse(json.dumps({'status': 200, 'datasets': dir_info_list }))
        # return HttpResponse(json.dumps({'status': 200}))
    except Exception as e:
        print("get_dataset_list error ---------------------------------\n")
        print(e)
        print("\n ------------------------------------------------------")
        return HttpResponse(json.dumps({'status': 404, 'datasets' : []}))

@api_view(['GET'])
@permission_classes([IsAuthenticated])   # 토큰 확인
def get_dataset_info(request):
    """
    get_dataset_list _summary_

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """

    try:
        dataset_name = request.GET["name"]

        print("dataset_name : " + str(dataset_name))

        path = os.path.join(root_path, "shared/datasets/*")
        dir_list = glob.glob(path)

        dataset_info = None

        for dir_path in dir_list:
            name = os.path.basename(dir_path)
            if name ==  dataset_name:
                print("dir_path : " + str(dir_path))
                dataset_info = get_folder_info(dir_path)
                break

        return HttpResponse(json.dumps({'status': 200, 'dataset': dataset_info }))
    except Exception as e:
        return HttpResponse(json.dumps({'status': 404}))


@api_view(['POST'])
@permission_classes([IsAuthenticated])   # 토큰 확인
def get_folders_size(request):
    try:
        folder_list = request.data['folder_list']

        num_processes = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(get_dir_size_handler, folder_list)

        return HttpResponse(json.dumps({'status': 200, 'datas': results }))
    except Exception as e:
        print(e)
        return HttpResponse(json.dumps({'status': 404}))

@api_view(['POST'])
@permission_classes([IsAuthenticated])   # 토큰 확인
def get_folders_file_count(request):
    try:
        folder_list = request.data['folder_list']

        num_processes = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(get_file_count, folder_list)

        return HttpResponse(json.dumps({'status': 200, 'datas': results }))
    except Exception as e:
        print(e)
        return HttpResponse(json.dumps({'status': 404}))        
    

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
        # folder_info['size'] = get_folder_size(folder_path) # 계산하는데 시간이 오래걸려 따로 수행..
        folder_info['creation_time'] = get_folder_creation_date(folder_path)
        # folder_info['last_modified_time'] = get_folder_last_modified_date(folder_path)
        # folder_info['file_count'] = get_file_count(folder_path) # 계산하는데 시간이 오래걸려 따로 수행..
        folder_info['thumbnail'] = get_folder_thumbnail(folder_path)
        # folder_info['isDownload'] = is_download_complete_dataset(folder_path)
        folder_info['status'] = check_dataset_status(folder_info)
        
    else:
        print("유효한 폴더 경로가 아닙니다.")
        return None

    return folder_info

def get_dir_size_handler(path):
    return {"folder_path": path, "size": get_dir_size(path)}

def get_dir_size(path='.'):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total

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
    # return file_count
    return { "folder_path": folder_path, "count": file_count}

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
        images = [ fi for fi in files if str(fi).lower().endswith(('.jpg','.jpeg', '.png',)) ]
        for image in images:
            file_list.append(os.path.join(path, image))
            if len(file_list)>4:
                break
        if len(file_list)>4:
            break

    # random choice
    if len(file_list) >= 4:
        # random_images = random.sample(file_list,4) if len(file_list) >= 4  else random.sample(file_list, len(file_list))
        random_images = file_list[0:4]
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

#endregion

#region Dataset Download .................................................................................

@api_view(['POST'])
@permission_classes([AllowAny])   # 토큰 확인
def download_coco(request):
    """
    download_coco _summary_

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """

    try:
        if is_thread_name_active(COMMON_DATASET_INFO["COCO"]["thread_name"]):
            print("[RUN_COCO_THREAD] Already running...")
            return HttpResponse(json.dumps({'status': 200, 'isAlready':True}))
        elif is_download_complete_dataset(COMMON_DATASET_INFO["COCO"]["path"]):
            print("COCO DATASET Download Complete")
            return HttpResponse(json.dumps({'status': 200, 'complete':True}))
        
        data = request.data
        args = [data["isTrain"], data["isVal"], data["isTest"], data["isSegments"], data["isSama"]]
        thread_1 = threading.Thread(target = download_coco_handler, args=args, name=COMMON_DATASET_INFO["COCO"]["thread_name"])
        thread_1.start()
        return HttpResponse(json.dumps({'status': 200}))
    except Exception as e:
        print(e)

def download_coco_handler(is_train, is_val, is_test, is_segments, is_sama):
    coco_script_file_path = COMMON_DATASET_INFO["COCO"]["script_path"]
    print(coco_script_file_path)
    if os.path.isfile(coco_script_file_path):
        os.chmod(coco_script_file_path, 0o755)

        # fix_path = root_path if os.environ.get('IS_DOCKER_COMPOSE') else BASE_DIR
        fix_path = root_path
        labels_unzip_path = os.path.join(fix_path, "shared/datasets")

        sh_run = str(coco_script_file_path)
        sh_run += " " + labels_unzip_path

        if is_train: sh_run += " --train"
        if is_val: sh_run += " --val"
        if is_test: sh_run += " --test"
        if is_segments: sh_run += " --segments"
        if is_sama: sh_run += " --sama"

        if os.path.exists(os.path.join(labels_unzip_path, "coco", ".DS_Store")):
            os.remove(os.path.join(labels_unzip_path, "coco", ".DS_Store"))
            print(f"Removed the .DS_Store file")
        else:
            print(f".DS_Store file does not exist")

        os.system(sh_run)

        print("coco_dataset download done")

        coco_yaml_file_path = os.path.join(BASE_DIR, "datasets_yaml", "coco", "coco_dataset.yaml") 
        coco_datasets_path = os.path.join(COMMON_DATASET_INFO["COCO"]["path"], "dataset.yaml")
        shutil.copy(coco_yaml_file_path, coco_datasets_path)


@api_view(['POST'])
@permission_classes([AllowAny])   # 토큰 확인
def download_imagenet(request):
    """
    download_coco _summary_

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """

    try:
        if is_thread_name_active(COMMON_DATASET_INFO["IMAGE_NET"]["thread_name"]):
            print("[RUN_IMAGENET_THREAD] Already running...")
            return HttpResponse(json.dumps({'status': 200, 'isAlready':True}))
        
        elif is_download_complete_dataset(COMMON_DATASET_INFO["IMAGE_NET"]["path"]):
            print("imagenet DATASET Download Complete")
            return HttpResponse(json.dumps({'status': 200, 'complete':True}))
        
        data = request.data
        args = [data["isTrain"], data["isVal"]]
        thread_1 = threading.Thread(target = download_imagenet_handler, args=args, name=COMMON_DATASET_INFO["IMAGE_NET"]["thread_name"])
        thread_1.start()
        return HttpResponse(json.dumps({'status': 200}))
    except Exception as e:
        print(e)

def download_imagenet_handler(is_train, is_val):
    imagenet_script_file_path = COMMON_DATASET_INFO["IMAGE_NET"]["script_path"]
    print(imagenet_script_file_path)
    if os.path.isfile(imagenet_script_file_path):
        os.chmod(imagenet_script_file_path, 0o755)

        labels_unzip_path = COMMON_DATASET_INFO["IMAGE_NET"]["path"]

        sh_run = str(imagenet_script_file_path)
        sh_run += " " + labels_unzip_path

        if is_train: sh_run += " --train"
        if is_val: sh_run += " --val"
        os.system(sh_run)

        print("imagenet download done")

        imagenet_yaml_file_path = os.path.join(BASE_DIR, "datasets_yaml", "imagenet", "imagenet_dataset.yaml") 
        imagenet_datasets_path = os.path.join(COMMON_DATASET_INFO["IMAGE_NET"]["path"], "dataset.yaml")
        shutil.copy(imagenet_yaml_file_path, imagenet_datasets_path)


@api_view(['POST'])
@permission_classes([AllowAny])   # 토큰 확인
def download_voc(request):
    """
    download_coco _summary_

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """

    try:
        if is_thread_name_active(COMMON_DATASET_INFO["VOC"]["thread_name"]):
            print("[RUN_VOC_THREAD] Already running...")
            return HttpResponse(json.dumps({'status': 200, 'isAlready':True}))
        elif is_download_complete_dataset(COMMON_DATASET_INFO["VOC"]["path"]):
            print("VOC DATASET Download Complete")
            return HttpResponse(json.dumps({'status': 200, 'complete':True}))
        thread_1 = threading.Thread(target = download_voc_handler, name=COMMON_DATASET_INFO["VOC"]["thread_name"])
        thread_1.start()
        return HttpResponse(json.dumps({'status': 200}))
    except Exception as e:
        print(e)

def download_voc_handler():
    voc_script_file_path = COMMON_DATASET_INFO["VOC"]["script_path"]
    print(voc_script_file_path)
    if os.path.isfile(voc_script_file_path):
        os.chmod(voc_script_file_path, 0o755)

        # fix_path = root_path if os.environ.get('IS_DOCKER_COMPOSE') else BASE_DIR
        fix_path = root_path
        labels_unzip_path = COMMON_DATASET_INFO["VOC"]["path"]

        sh_run = str(voc_script_file_path)
        sh_run += " " + labels_unzip_path

        os.system(sh_run)

        print("voc download done")

        voc_yaml_file_path = os.path.join(BASE_DIR, "datasets_yaml", "VOC", "voc_dataset.yaml") 
        voc_datasets_path = os.path.join(COMMON_DATASET_INFO["VOC"]["path"], "dataset.yaml")
        shutil.copy(voc_yaml_file_path, voc_datasets_path)


@api_view(['POST'])
@permission_classes([AllowAny])   # 토큰 확인
def download_chest_xray_dataset(request):
    """
    download_kaggle_dataset _summary_

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """

    try:
        if is_thread_name_active(COMMON_DATASET_INFO["CHEST_XRAY"]["thread_name"]):
            print("[RUN_KAGGLE_THREAD] Already running...")
            return HttpResponse(json.dumps({'status': 200, 'isAlready':True}))
        elif is_download_complete_dataset(COMMON_DATASET_INFO["CHEST_XRAY"]["path"]):
            print("KAGGLE DATASET Download Complete")
            return HttpResponse(json.dumps({'status': 200, 'complete':True}))
        thread_1 = threading.Thread(target = download_chest_xray_handler, args=[request.user], name=COMMON_DATASET_INFO["CHEST_XRAY"]["thread_name"])
        thread_1.start()
        return HttpResponse(json.dumps({'status': 200}))
    except Exception as e:
        print(e)

def download_chest_xray_handler(user_id):
    api = authenticate_kaggle(user_id)
    # 데이터셋 다운로드
    dataset = 'paultimothymooney/chest-xray-pneumonia'

    # fix_path = root_path if os.environ.get('IS_DOCKER_COMPOSE') else BASE_DIR
    dataset_path =COMMON_DATASET_INFO["CHEST_XRAY"]["path"]

    # 데이터셋 다운로드
    api.dataset_download_files(dataset, path=dataset_path, unzip=True)

    move_files_recursive(os.path.join(dataset_path, "chest_xray"), dataset_path)
    print(dataset + ".zip unzip done")

    if os.path.exists(os.path.join(dataset_path, "chest_xray")):
        shutil.rmtree(os.path.join(dataset_path, "chest_xray"))
    else:
        print(f"chest_xray folder does not exist")


def load_kaggle_credentials(user_id):
    # 사용자 이름을 기반으로 해당 사용자의 설정 파일 경로를 결정
    # user_kaggle_json_path = f'/path/to/{user_id}_kaggle.json'
    print("load_kaggle_credentials - user_id : " + str(user_id))
    home_dir = os.path.expanduser("~")
    kaggle_dir = os.path.join(home_dir, ".kaggle")
    user_kaggle_json_path = os.path.join(kaggle_dir, str(user_id)+"_kaggle.json")

    # 해당 사용자의 설정 파일이 있는지 확인
    if os.path.exists(user_kaggle_json_path):
        with open(user_kaggle_json_path, 'r') as f:
            kaggle_json = json.load(f)
            return kaggle_json['username'], kaggle_json['key']
    else:
        raise FileNotFoundError(f"Kaggle JSON file for user '{user_id}' not found.")

def authenticate_kaggle(user_id):
    # 해당 사용자의 Kaggle 인증 정보 로드
    kaggle_username, kaggle_key = load_kaggle_credentials(user_id)

    print("authenticate_kaggle - kaggle_username", kaggle_username)
    print("authenticate_kaggle - kaggle_key", kaggle_key)

    # Kaggle API에 인증
    os.environ['KAGGLE_USERNAME'] = kaggle_username
    os.environ['KAGGLE_KEY'] = kaggle_key

    # Kaggle API를 사용하여 데이터셋 다운로드
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    return api

def setup_kaggle_api(user_id, kaggle_userid, kaggle_key):
    """
    Kaggle User ID와 API Key를 입력받아 kaggle.json 파일을 생성하고,
    ~/.kaggle/ 디렉토리에 저장하며 적절한 권한을 설정합니다.
    """
    
    # kaggle.json 데이터 생성
    kaggle_data = {
        "username": kaggle_userid,
        "key": kaggle_key
    }

    # 홈 디렉토리 경로 가져오기
    home_dir = os.path.expanduser("~")
    kaggle_dir = os.path.join(home_dir, ".kaggle")

    # .kaggle 디렉토리 존재 여부 확인 후 생성
    if not os.path.exists(kaggle_dir):
        os.makedirs(kaggle_dir)

    # kaggle.json 파일 경로 설정
    kaggle_json_path = os.path.join(kaggle_dir, str(user_id)+"_kaggle.json")

    # kaggle.json 파일 작성
    with open(kaggle_json_path, "w") as f:
        json.dump(kaggle_data, f)

    # 파일 권한 설정
    os.chmod(kaggle_json_path, 0o600)

    print(f'kaggle.json 파일이 {kaggle_json_path}에 저장되었습니다. ')

@api_view(['GET'])
@permission_classes([AllowAny])   # 토큰 확인
def is_exist_user_kaggle_json(request):
    try:
        print("request.user : " + str(request.user))
        username, key = load_kaggle_credentials(request.user)
        print("username : " + str(username))
        print("key : " + str(key))
        return HttpResponse(json.dumps({'status': 200, 'isExist':True, "username":username, "key":key}))
    except FileNotFoundError:
        return HttpResponse(json.dumps({'status': 200, 'isExist':False}))
    except Exception:
        return HttpResponse(json.dumps({'status': 404}))
    
@api_view(['POST'])
@permission_classes([AllowAny])   # 토큰 확인
def setup_user_kaggle_api(request):
    try:
        print('request.data["username"] : ' + str(request.data["username"]))
        print('request.data["key"] : ' + str(request.data["key"]))
        setup_kaggle_api(request.user, request.data["username"], request.data["key"])
        return HttpResponse(json.dumps({'status': 200, 'isExist':True}))
    except Exception:
        return HttpResponse(json.dumps({'status': 404}))
        

#endregion

#region Common Func....................................................................................... 
def create_folder_if_not_exists(path):
    """
    주어진 경로에 폴더가 없으면 생성하는 함수.
    
    :param path: 폴더를 확인하고 생성할 경로
    """
    if not os.path.isdir(path):
        os.makedirs(path)
        print(f"{path} 폴더를 생성했습니다.")
    else:
        print(f"{path} 폴더가 이미 존재합니다.")

def dataset_start_scirpt():
    for common_dataset in COMMON_DATASET_INFO.values():
        create_folder_if_not_exists(common_dataset["path"])

def is_thread_name_active(name):
    """
    주어진 이름을 가진 쓰레드가 활성 상태인지 확인하는 함수

    :param name: 확인할 쓰레드 이름
    :return: 쓰레드 이름이 활성 상태이면 True, 그렇지 않으면 False
    """
    return any(thread.name == name for thread in threading.enumerate())

def delete_all_files_in_directory(directory):
    # 디렉토리가 존재하는지 확인
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return

    # 디렉토리 내 모든 파일과 하위 디렉토리를 반복
    for root, dirs, files in os.walk(directory):
        for name in files:
            file_path = os.path.join(root, name)
            try:
                os.remove(file_path)
                print(f"File {file_path} has been deleted.")
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

        for name in dirs:
            dir_path = os.path.join(root, name)
            try:
                shutil.rmtree(dir_path)
                print(f"Directory {dir_path} has been deleted.")
            except Exception as e:
                print(f"Failed to delete {dir_path}. Reason: {e}")

def move_files_recursive(source_folder, destination_folder):
    # 소스 폴더의 모든 파일과 폴더 리스트
    items = os.listdir(source_folder)
    
    # 대상 폴더가 존재하지 않으면 생성
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    for item in items:
        # 소스 아이템의 절대 경로
        source_item = os.path.join(source_folder, item)
        # 대상 아이템의 절대 경로
        destination_item = os.path.join(destination_folder, item)
        
        if os.path.isdir(source_item):
            # 만약 소스 아이템이 폴더라면, 재귀적으로 함수 호출하여 해당 폴더의 내용을 복사
            move_files_recursive(source_item, destination_item)
        else:
            # 파일을 복사
            shutil.move(source_item, destination_item)

def is_download_complete_dataset(folder_path):
    file_count = 0
    for path, dirs, files in os.walk(folder_path):
        file_count += len(files)
        if file_count > 2:
            return True
    
    if file_count > 2:
        return True
    else: 
        return False

def check_dataset_status(dataset):
    folder_name = dataset["name"]
    folder_path = dataset["path"]

    common_dataset = next((common_dataset for common_dataset in COMMON_DATASET_INFO.values() if common_dataset["name"] == folder_name), None)

    # Common Dataset이 아닌 경우 (= 사용자가 직접 업로드한 경우? 등등 이미 다운완료 되었을 것)
    if(common_dataset == None):
        return DATASET_STATUS.COMPLETE.value

    # 다운로드 중일 경우
    if is_thread_name_active(common_dataset["thread_name"]) == True:
        return DATASET_STATUS.DOWNLOADING.value

    # 데이터 셋 다운로드가 완료된 경우
    if is_download_complete_dataset(folder_path) == True:
        return DATASET_STATUS.COMPLETE.value

    # 데이터셋 다운로드 전.
    return DATASET_STATUS.NONE.value

#endregion



