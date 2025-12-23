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
import numpy as np

import multiprocessing
import threading
import time
import zipfile
import shutil
import yaml

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

IMAGENET_SYNONYM_CACHE = {}
IMAGENET_SIMPLE_LABELS_CACHE = {}
IMAGENET_CLASS_INDEX_CACHE = {}

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


@api_view(['GET'])
@permission_classes([IsAuthenticated])   # 토큰 확인
def get_dataset_preview(request):
    try:
        dataset_name = request.GET.get("name")
        count = int(request.GET.get("count", 5))
        if not dataset_name:
            return HttpResponse(json.dumps({'status': 400, 'message': 'dataset name is required'}))

        folder_path = None
        path = os.path.join(root_path, "shared/datasets/*")
        for dir_path in glob.glob(path):
            if os.path.basename(dir_path) == dataset_name:
                folder_path = dir_path
                break

        if folder_path is None:
            return HttpResponse(json.dumps({'status': 404, 'message': 'dataset not found'}))

        dataset_yaml_path = os.path.join(folder_path, "dataset.yaml")
        if not os.path.isfile(dataset_yaml_path):
            return HttpResponse(json.dumps({'status': 404, 'message': 'dataset.yaml not found'}))

        dataset_info = load_dataset_yaml(dataset_yaml_path)
        train_sources = extract_train_sources(dataset_info, dataset_yaml_path)
        if not train_sources:
            return HttpResponse(json.dumps({'status': 404, 'message': 'train sources not found'}))

        sample_paths = reservoir_sample(iter_image_paths(train_sources, dataset_yaml_path), count)
        samples = []
        detected_task = None
        names = dataset_info.get("names")

        for image_path in sample_paths:
            sample = build_sample_preview(image_path, names, dataset_yaml_path)
            if sample is None:
                continue
            samples.append(sample)
            sample_task = sample.get("task")
            if sample_task == "segmentation":
                detected_task = "segmentation"
            elif sample_task == "detection" and detected_task != "segmentation":
                detected_task = "detection"
            elif detected_task is None and sample_task == "classification":
                detected_task = "classification"

        return HttpResponse(json.dumps({'status': 200, 'task': detected_task, 'samples': samples }))
    except Exception as e:
        print("get_dataset_preview error ---------------------------------\n")
        print(e)
        print("\n ------------------------------------------------------")
        return HttpResponse(json.dumps({'status': 500, 'samples': []}))


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
    dataset_yaml_path = os.path.join(path, "dataset.yaml")
    if os.path.isfile(dataset_yaml_path):
        train_size = get_train_image_size_from_yaml(dataset_yaml_path)
        if train_size is not None:
            return {"folder_path": path, "size": train_size}
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
    dataset_yaml_path = os.path.join(folder_path, "dataset.yaml")
    if os.path.isfile(dataset_yaml_path):
        train_count = get_train_image_count_from_yaml(dataset_yaml_path)
        if train_count is not None:
            return { "folder_path": folder_path, "count": train_count}

    file_count = 0
    for path, dirs, files in os.walk(folder_path):
        file_count += len(files)
    return { "folder_path": folder_path, "count": file_count}

def get_train_image_count_from_yaml(dataset_yaml_path):
    try:
        with open(dataset_yaml_path, "r") as f:
            dataset_info = yaml.safe_load(f) or {}
    except Exception as error:
        print(f"dataset.yaml parse error: {dataset_yaml_path}")
        print(error)
        return None

    train_value = dataset_info.get("train")
    if train_value is None:
        return None

    train_sources = train_value if isinstance(train_value, list) else [train_value]
    base_dir = os.path.dirname(dataset_yaml_path)

    total = 0
    for source in train_sources:
        if not isinstance(source, str) or not source.strip():
            continue
        resolved = resolve_dataset_path(source, base_dir)
        if os.path.isdir(resolved):
            total += count_images_in_dir(resolved)
        elif os.path.isfile(resolved):
            total += count_images_in_list_file(resolved)
    return total

def get_train_image_size_from_yaml(dataset_yaml_path):
    try:
        with open(dataset_yaml_path, "r") as f:
            dataset_info = yaml.safe_load(f) or {}
    except Exception as error:
        print(f"dataset.yaml parse error: {dataset_yaml_path}")
        print(error)
        return None

    train_value = dataset_info.get("train")
    if train_value is None:
        return None

    train_sources = train_value if isinstance(train_value, list) else [train_value]
    base_dir = os.path.dirname(dataset_yaml_path)

    total = 0
    for source in train_sources:
        if not isinstance(source, str) or not source.strip():
            continue
        resolved = resolve_dataset_path(source, base_dir)
        if os.path.isdir(resolved):
            total += get_image_dir_size(resolved)
        elif os.path.isfile(resolved):
            total += get_image_list_file_size(resolved)
    return total

def resolve_dataset_path(path_value, base_dir):
    if os.path.isabs(path_value):
        return path_value
    return os.path.normpath(os.path.join(base_dir, path_value))

def load_dataset_yaml(dataset_yaml_path):
    try:
        with open(dataset_yaml_path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as error:
        print(f"dataset.yaml parse error: {dataset_yaml_path}")
        print(error)
        return {}

def extract_train_sources(dataset_info, dataset_yaml_path):
    train_value = dataset_info.get("train")
    if train_value is None:
        return []
    train_sources = train_value if isinstance(train_value, list) else [train_value]
    base_dir = os.path.dirname(dataset_yaml_path)
    resolved_sources = []
    for source in train_sources:
        if not isinstance(source, str) or not source.strip():
            continue
        resolved_sources.append(resolve_dataset_path(source, base_dir))
    return resolved_sources

def iter_image_paths(train_sources, dataset_yaml_path):
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    for source in train_sources:
        if os.path.isdir(source):
            for root, dirs, files in os.walk(source):
                for file_name in files:
                    if file_name.lower().endswith(image_exts):
                        yield os.path.join(root, file_name)
        elif os.path.isfile(source):
            base_dir = os.path.dirname(source)
            with open(source, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    image_path = line.split()[0]
                    resolved = resolve_dataset_path(image_path, base_dir)
                    if os.path.isfile(resolved) and resolved.lower().endswith(image_exts):
                        yield resolved

def reservoir_sample(iterator, count):
    if count <= 0:
        return []
    samples = []
    for idx, item in enumerate(iterator):
        if len(samples) < count:
            samples.append(item)
            continue
        replace_idx = random.randint(0, idx)
        if replace_idx < count:
            samples[replace_idx] = item
    return samples

def build_sample_preview(image_path, names, dataset_yaml_path=None):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"[preview] image load failed: {image_path}")
        return None
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    label_path = guess_label_path(image_path)
    labels = []
    label_items = []
    task = "classification"
    legend = []
    if label_path and os.path.isfile(label_path):
        annotations, task = parse_yolo_label_file(label_path)
        if annotations:
            image, legend = draw_annotations(image, annotations, names)
            labels = [get_class_name(item["class_id"], names) for item in annotations]
        else:
            print(f"[preview] no annotations parsed: {label_path}")
    else:
        print(f"[preview] label file missing: {label_path} (image={image_path})")

    if not labels and task == "classification":
        class_label = get_class_label_from_path(image_path, names, dataset_yaml_path)
        if class_label:
            labels = [class_label]
            label_items = [
                {
                    "text": class_label,
                    "color_hex": color_to_hex(color_for_label(class_label))
                }
            ]
        else:
            print(f"[preview] classification label missing: {image_path}")

    success, jpg_img = cv2.imencode('.jpg', image)
    if not success:
        return None

    return {
        "file": os.path.basename(image_path),
        "labels": labels,
        "label_items": label_items,
        "legend": legend,
        "task": task,
        "image": "data:image/jpg;base64," + str(base64.b64encode(jpg_img).decode('utf-8'))
    }

def guess_label_path(image_path):
    parts = image_path.split(os.sep)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        base_path = os.path.splitext(join_path_parts(parts))[0]
    else:
        base_path = os.path.splitext(image_path)[0]
    label_path = base_path + ".txt"
    print(f"[preview] guessed label path: {label_path}")
    return label_path

def join_path_parts(parts):
    if parts and parts[0] == "":
        return os.sep + os.path.join(*parts[1:])
    return os.path.join(*parts)

def parse_yolo_label_file(label_path):
    annotations = []
    task = "detection"
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                print(f"[preview] invalid label line (too short): {label_path} :: {line}")
                continue
            try:
                values = [float(part) for part in parts]
            except ValueError:
                print(f"[preview] invalid label line (non-numeric): {label_path} :: {line}")
                continue
            class_id = int(values[0])
            coords = values[1:]
            if len(coords) > 4:
                task = "segmentation"
                if len(coords) % 2 != 0:
                    print(f"[preview] invalid segmentation coords: {label_path} :: {line}")
                    continue
                points = list(zip(coords[0::2], coords[1::2]))
                annotations.append({"class_id": class_id, "points": points})
            else:
                annotations.append({"class_id": class_id, "bbox": tuple(coords)})
    print(f"[preview] parsed {len(annotations)} annotations from {label_path}")
    return annotations, task

def draw_annotations(image, annotations, names):
    height, width = image.shape[:2]
    legend = []
    largest_by_class = {}
    for ann in annotations:
        if "bbox" not in ann:
            continue
        cx, cy, bw, bh = ann["bbox"]
        area = bw * bh
        class_id = ann["class_id"]
        if class_id not in largest_by_class or area > largest_by_class[class_id]["area"]:
            largest_by_class[class_id] = {
                "area": area,
                "bbox": ann["bbox"]
            }

    for ann in annotations:
        class_id = ann["class_id"]
        label = get_class_name(class_id, names)
        color = color_for_class(class_id)
        if label.strip().lower() == "person":
            color = (255, 255, 255)
        if "bbox" in ann:
            cx, cy, bw, bh = ann["bbox"]
            x1 = int((cx - bw / 2) * width)
            y1 = int((cy - bh / 2) * height)
            x2 = int((cx + bw / 2) * width)
            y2 = int((cy + bh / 2) * height)
            x1 = max(0, min(width - 1, x1))
            x2 = max(0, min(width - 1, x2))
            y1 = max(0, min(height - 1, y1))
            y2 = max(0, min(height - 1, y2))
            overlay = image.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.18, image, 0.82, 0, image)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            largest = largest_by_class.get(class_id)
            if largest and largest["bbox"] == ann["bbox"]:
                abbr = get_label_abbreviation(label)
                draw_abbr_circle(image, abbr, (x1, y1), color)
                rgb_color = bgr_to_rgb(color)
                legend.append({
                    "abbr": abbr,
                    "label": label,
                    "color": list(rgb_color),
                    "color_hex": color_to_hex(rgb_color)
                })
        elif "points" in ann:
            points = [(int(x * width), int(y * height)) for x, y in ann["points"]]
            if len(points) < 3:
                continue
            pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            overlay = image.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
            cv2.polylines(image, [pts], True, color, 2)
            label_pos = points[0]
            draw_label_box(image, label, label_pos, color)
    legend = dedupe_legend(legend)
    return image, legend

def get_class_name(class_id, names):
    if isinstance(names, dict):
        return names.get(class_id) or names.get(str(class_id)) or str(class_id)
    if isinstance(names, list):
        if 0 <= class_id < len(names):
            return names[class_id]
    return str(class_id)

def get_class_label_from_path(image_path, names, dataset_yaml_path=None):
    parent = os.path.basename(os.path.dirname(image_path))
    synset_label = resolve_imagenet_label_by_synset(parent, dataset_yaml_path)
    if synset_label:
        return synset_label
    if parent.isdigit():
        idx = int(parent)
        imagenet_label = resolve_imagenet_label_by_index(idx, dataset_yaml_path, names)
        if imagenet_label:
            return imagenet_label
        if isinstance(names, dict):
            return names.get(parent) or names.get(idx) or parent
        if isinstance(names, list) and 0 <= idx < len(names):
            return names[idx]
    if isinstance(names, dict):
        return names.get(parent) or names.get(str(parent)) or parent
    if isinstance(names, list):
        if parent in names:
            return parent
    if dataset_yaml_path:
        synset_label = resolve_imagenet_label_by_synset(parent, dataset_yaml_path)
        if synset_label:
            return synset_label
    if dataset_yaml_path:
        synset_label = resolve_synset_label(parent, dataset_yaml_path)
        if synset_label:
            return synset_label
    return parent

def draw_label_box(image, text, origin, color, font_scale=0.75, thickness=2, padding=4):
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    box_x1 = max(0, x)
    box_y1 = max(0, y - text_h - padding * 2)
    box_x2 = min(image.shape[1] - 1, x + text_w + padding * 2)
    box_y2 = min(image.shape[0] - 1, y + baseline + padding)
    cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), color, -1)
    text_x = box_x1 + padding
    text_y = box_y2 - padding
    cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def get_label_abbreviation(label):
    label = label.strip()
    if not label:
        return "?"
    parts = [part for part in label.split(" ") if part]
    if len(parts) > 1:
        return "".join(part[0].upper() for part in parts[:2])
    return label[0].upper()

def dedupe_legend(legend):
    seen = set()
    result = []
    for item in legend:
        key = item.get("abbr")
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def color_for_class(class_id):
    return (
        50 + int((class_id * 37) % 180),
        50 + int((class_id * 17) % 180),
        50 + int((class_id * 29) % 180)
    )

def color_to_hex(color):
    try:
        return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))
    except Exception:
        return "#444444"

def bgr_to_rgb(color):
    try:
        return (int(color[2]), int(color[1]), int(color[0]))
    except Exception:
        return color

def color_for_label(label):
    if label.strip().lower() == "person":
        return (255, 255, 255)
    total = 0
    for ch in label:
        total = (total * 31 + ord(ch)) % 100000
    return (
        50 + (total * 37) % 180,
        50 + (total * 17) % 180,
        50 + (total * 29) % 180
    )

def resolve_synset_label(synset, dataset_yaml_path):
    if not synset or not synset.startswith("n"):
        return None
    base_dir = os.path.dirname(dataset_yaml_path)
    cache_key = base_dir
    if cache_key not in IMAGENET_SYNONYM_CACHE:
        IMAGENET_SYNONYM_CACHE[cache_key] = load_synset_map(base_dir)
    return IMAGENET_SYNONYM_CACHE[cache_key].get(synset)

def resolve_imagenet_label_by_index(idx, dataset_yaml_path, names):
    if not dataset_yaml_path:
        return None
    base_dir = os.path.dirname(dataset_yaml_path)
    cache_key = base_dir
    if cache_key not in IMAGENET_SIMPLE_LABELS_CACHE:
        IMAGENET_SIMPLE_LABELS_CACHE[cache_key] = load_imagenet_simple_labels(base_dir)
    labels = IMAGENET_SIMPLE_LABELS_CACHE[cache_key]
    if labels and 0 <= idx < len(labels):
        return labels[idx]
    return None

def resolve_imagenet_label_by_synset(synset, dataset_yaml_path):
    if not synset or not synset.startswith("n"):
        return None
    base_dir = os.path.dirname(dataset_yaml_path) if dataset_yaml_path else os.path.join(BASE_DIR, "datasets_yaml", "imagenet")
    cache_key = base_dir
    if cache_key not in IMAGENET_CLASS_INDEX_CACHE:
        IMAGENET_CLASS_INDEX_CACHE[cache_key] = load_imagenet_class_index(base_dir)
    synset_to_idx = IMAGENET_CLASS_INDEX_CACHE[cache_key]
    idx = synset_to_idx.get(synset)
    if idx is None:
        print(f"[preview] imagenet synset not found: {synset}")
        return None
    if cache_key not in IMAGENET_SIMPLE_LABELS_CACHE:
        IMAGENET_SIMPLE_LABELS_CACHE[cache_key] = load_imagenet_simple_labels(base_dir)
    labels = IMAGENET_SIMPLE_LABELS_CACHE[cache_key]
    if labels and 0 <= idx < len(labels):
        print(f"[preview] imagenet synset mapped: {synset} -> {idx} -> {labels[idx]}")
        return labels[idx]
    print(f"[preview] imagenet label missing for index: {idx} (synset={synset})")
    return None

def load_synset_map(base_dir):
    mapping = {}
    candidates = ["synset_words.txt", "words.txt", "imagenet_words.txt"]
    for filename in candidates:
        path = os.path.join(base_dir, filename)
        if not os.path.isfile(path):
            continue
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) != 2:
                    continue
                synset_id, label = parts[0], parts[1]
                label = label.split(",")[0].strip()
                mapping[synset_id] = label
        break
    return mapping

def load_imagenet_simple_labels(base_dir):
    candidates = [
        os.path.join(base_dir, "imagenet-simple-labels.json"),
        os.path.join(BASE_DIR, "datasets_yaml", "imagenet", "imagenet-simple-labels.json")
    ]
    for json_path in candidates:
        if not os.path.isfile(json_path):
            continue
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception as error:
            print(f"imagenet simple labels load error: {json_path}")
            print(error)
    return []

def load_imagenet_class_index(base_dir):
    candidates = [
        os.path.join(base_dir, "imagenet_class_index.json"),
        os.path.join(BASE_DIR, "datasets_yaml", "imagenet", "imagenet_class_index.json")
    ]
    for json_path in candidates:
        if not os.path.isfile(json_path):
            continue
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                mapping = {}
                for idx_str, value in data.items():
                    if not isinstance(value, list) or len(value) < 1:
                        continue
                    synset = value[0]
                    try:
                        idx = int(idx_str)
                    except ValueError:
                        continue
                    mapping[synset] = idx
                return mapping
        except Exception as error:
            print(f"imagenet class index load error: {json_path}")
            print(error)
    return {}

def draw_abbr_circle(image, text, origin, color):
    x, y = origin
    radius = 14
    center = (x + radius, max(radius + 2, y - radius))
    avg = (color[0] + color[1] + color[2]) / 3
    text_color = (0, 0, 0) if avg > 200 else (255, 255, 255)
    cv2.circle(image, center, radius, color, -1)
    if avg > 200:
        cv2.circle(image, center, radius, (120, 120, 120), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6 if len(text) <= 1 else 0.5
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = center[0] - text_size[0] // 2
    text_y = center[1] + text_size[1] // 2
    cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

def count_images_in_dir(folder_path):
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    count = 0
    for path, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.lower().endswith(image_exts):
                count += 1
    return count

def get_image_dir_size(folder_path):
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    total = 0
    for path, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.lower().endswith(image_exts):
                file_path = os.path.join(path, file_name)
                try:
                    total += os.path.getsize(file_path)
                except OSError:
                    continue
    return total

def count_images_in_list_file(list_path):
    count = 0
    with open(list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            count += 1
    return count

def get_image_list_file_size(list_path):
    total = 0
    base_dir = os.path.dirname(list_path)
    with open(list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            resolved = resolve_dataset_path(line, base_dir)
            if os.path.isfile(resolved):
                try:
                    total += os.path.getsize(resolved)
                except OSError:
                    continue
    return total

def get_folder_thumbnail(folder_path):
    """
    Create thumbnails after randomly extracting 4 images from a folder

    Args:
        folder_path (string): Image folder path

    Returns:
        Returns Thumbnails to base64
    """

    # file_list = []

    # # get image path in folder
    # for path, dirs, files in os.walk(folder_path):
    #     images = [ fi for fi in files if str(fi).lower().endswith(('.jpg','.jpeg', '.png',)) ]
    #     for image in images:
    #         file_list.append(os.path.join(path, image))
    #         if len(file_list)>4:
    #             break
    #     if len(file_list)>4:
    #         break

    # # random choice
    # if len(file_list) >= 4:
    #     # random_images = random.sample(file_list,4) if len(file_list) >= 4  else random.sample(file_list, len(file_list))
    #     random_images = file_list[0:4]
    #     thumbnail_list = []
    #     for image in random_images:
    #         try:
    #             thumbnail = make_image_thumbnail(image)
    #             thumbnail_list.append(thumbnail)
    #         except Exception:
    #             print("thumbnail error")

    # get image path in folder

    thumbnail_list = []
    error_count = 0
    for path, dirs, files in os.walk(folder_path):
        images = [ fi for fi in files if str(fi).lower().endswith(('.jpg','.jpeg', '.png',)) ]
        for image in images:
            try:
                thumbnail = make_image_thumbnail(os.path.join(path, image))
                thumbnail_list.append(thumbnail)
            except Exception as error:
                print(f"thumbnail 생성 실패 : {os.path.join(path, image)}")
                print(error)
                print("=====================================================")

                error_count += 1
                if error_count > 1000:
                    return None
                else:
                    continue

            if len(thumbnail_list)>=4:
                break
        if len(thumbnail_list)>=4:
            break

    if len(thumbnail_list) <= 0:
        print(f"{folder_path}에서 생성된 thumbnail_list가 존재하지 않음.\n")
        return None

    try:
        thumb = cv2.hconcat(thumbnail_list)
        jpg_img = cv2.imencode('.jpg', thumb)
        return "data:image/jpg;base64," + str(base64.b64encode(jpg_img[1]).decode('utf-8'))
    
    except Exception as error:
        print(f" get_folder_thumbnail error : {folder_path}\n")
        print(error)
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
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise Exception("Image file not found in path.")

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

    if os.path.exists(os.path.join(dataset_path, "__MACOSX")):
        shutil.rmtree(os.path.join(dataset_path, "__MACOSX"))

    if os.path.exists(os.path.join(dataset_path, "train", ".DS_Store")):
        os.remove(os.path.join(dataset_path, "train", ".DS_Store"))

    if os.path.exists(os.path.join(dataset_path, "test", ".DS_Store")):
        os.remove(os.path.join(dataset_path, "test", ".DS_Store"))

    if os.path.exists(os.path.join(dataset_path, "val", ".DS_Store")):
        os.remove(os.path.join(dataset_path, "val", ".DS_Store"))

    chest_xray_yaml_file_path = os.path.join(BASE_DIR, "datasets_yaml", "ChestXRay", "ChestXRay_dataset.yaml") 
    chest_xray_datasets_path = os.path.join(COMMON_DATASET_INFO["CHEST_XRAY"]["path"], "dataset.yaml")
    shutil.copy(chest_xray_yaml_file_path, chest_xray_datasets_path)


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
        if file_count > 5:
            return True
    
    if file_count > 5:
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

def copy_train_file_for_version(version):
    try:
        print("copy file name : ", f"train_{str(version).zfill(4)}.txt")
        src = os.path.join(BASE_DIR, "datasets_yaml", "coco", f"train_{str(version).zfill(4)}.txt") 
        print("copy_train_file_for_version - src : ", src)
    
        dst = os.path.join(COMMON_DATASET_INFO["COCO"]["path"], f"train2017.txt")
        print("copy_train_file_for_version - dst : ", dst)
        
        shutil.copy(src, dst)
    except Exception as error:
        print(f"copy_train_file_for_version ERROR : {error}")
        raise error
    

#endregion
