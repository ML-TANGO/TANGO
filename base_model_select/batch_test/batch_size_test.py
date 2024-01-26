# Simplified Version of autobatch.py of Yolov5, AGPL-3.0 license
import torch
import gc
import yaml
import os

from copy import deepcopy
from pathlib import Path

from .yolo.models.yolo import Model as yolo_model
from .yolo.nas.supernet.supernet_yolov7 import YOLOSuperNet
from .resnet.resnet_cifar10 import ResNet as resnet_model
from .resnet.resnet_cifar10 import BasicBlock


def run_batch_test(basemodel_yaml, task, imgsz, hyp_yaml=None, nas=None):
    prefix = '[ AutoBatch ]'
    if task=='detection':
        with open(hyp_yaml, 'r') as f:
            hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
        if nas:
            model = YOLOSuperNet(str(Path(os.path.dirname(__file__))/'yolo'/'nas'/'supernet'/'yolov7_supernet.yml'), ch=3, nc=80, anchors=hyp.get('anchors'))
            model.set_max_net()
            print(f'{prefix} YOLOSuperNet is used for AutoBatch.')
        else:
            model = yolo_model(basemodel_yaml, ch=3, nc=80, anchors=hyp.get('anchors'))
            print(f'YOLO is used for AutoBatch.')

    elif task=='classification':
        with open(basemodel_yaml, 'r') as f:
            basemodel_dict = yaml.load(f, Loader=yaml.SafeLoader)
        model = resnet_model(BasicBlock, 
                             basemodel_dict.get('layers', [3,3,3]), 
                             basemodel_dict.get('num_classes', 2))
        print(f'{prefix} {task} model is used for AutoBatch.')

    else:
        print(f'{prefix} task is unknown ({task})')
        return None

    batch_size = int(get_batch_size_for_gpu(model, 3 if task=='detection' else 1, imgsz, amp=True) * 0.8)
    # It assumes that the memory sizes of all gpus in a machine are same.
    # 0.8 is multiplied by batch size to prevent cuda memory error due to a memory leak of yolov7

    del model
    torch.cuda.empty_cache()
    gc.collect()

    batch_size = False if batch_size < 2 else batch_size * torch.cuda.device_count()
    print(f'{prefix} Selcted Batch Size - {batch_size} (0.8 * batch size for a gpu * gpu_num)')

    return batch_size


def get_batch_size_for_gpu(model, ch, imgsz, amp):
    with torch.cuda.amp.autocast(amp):
        return autobatch(model, ch, imgsz)


def autobatch(model, ch, imgsz, batch_size=4):
    prefix = '[ AutoBatch ]'
    if torch.cuda.is_available():
       num_dev = torch.cuda.device_count() 
    else:
       print(f'{prefix} CUDA not detected, using default CPU batch size {batch_size}')
       return batch_size

    device = torch.device(f'cuda:0')
    model.to(device)
    model.train()
    batch_size = 2
    while True:
        img = torch.zeros(batch_size, ch, imgsz, imgsz).float()
        img = img.to(device)
        try:
            model(img)
            print(f'{prefix} success: ', batch_size)
            batch_size = batch_size * 2
        except RuntimeError as e:
            print(e)
            print(f'{prefix} fail: ', batch_size)
            final_batch_size = int(batch_size / 2.)
            del img
            break

    return final_batch_size
