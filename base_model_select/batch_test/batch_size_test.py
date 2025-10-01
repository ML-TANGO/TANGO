# Simplified Version of autobatch.py of Yolov5, AGPL-3.0 license
import torch
import gc
import yaml
import os

from copy import deepcopy
from pathlib import Path

from autonn.autonn.autonn_core.tango.common.models.yolo import Model as yolo_model
from autonn.autonn.autonn_core.tango.common.models.supernet_yolov9 import NASModel as YOLOSuperNet
from .resnet.resnet_cifar10 import ResNet as resnet_model
from .resnet.resnet_cifar10 import BasicBlock

from .binary_search import TestFuncGen, binary_search


PREFIX = '[ BMS - AutoBatch ]'
DEBUG = False


def run_batch_test(basemodel_yaml, task, imgsz, hyp_yaml=None, nas=None):
    print(f'{PREFIX} Start AutoBatch')
    if task=='detection':
        with open(hyp_yaml, 'r') as f:
            hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
        if nas:
            cfg_path = Path(__file__).resolve().parents[3] / 'autonn' / 'autonn' / 'autonn_core' / 'tango' / 'common' / 'cfg' / 'yolov9' / 'yolov9-supernet.yml'
            model = YOLOSuperNet(str(cfg_path), ch=3, nc=80, anchors=hyp.get('anchors'))
            model.set_max_net()
            if DEBUG: print(f'{PREFIX} YOLOSuperNet is used for AutoBatch.')
        else:
            model = yolo_model(basemodel_yaml, ch=3, nc=80, anchors=hyp.get('anchors'))
            if DEBUG: print(f'{PREFIX} YOLO is used for AutoBatch.')

    elif task=='classification':
        with open(basemodel_yaml, 'r') as f:
            basemodel_dict = yaml.load(f, Loader=yaml.SafeLoader)
        model = resnet_model(BasicBlock,
                             basemodel_dict.get('layers', [3,3,3]),
                             basemodel_dict.get('num_classes', 2))
        if DEBUG: print(f'{PREFIX} {task} model is used for AutoBatch.')

    else:
        if DEBUG: print(f'{PREFIX} task is unknown ({task})')
        return None

    batch_size = int(get_batch_size_for_gpu(model, 3 if task=='detection' else 1, imgsz, amp=True) * 0.9)
    # It assumes that the memory sizes of all gpus in a machine are same.
    # 0.8 safety factor keeps headroom to avoid CUDA OOM during training

    del model
    torch.cuda.empty_cache()
    gc.collect()

    batch_size = False if batch_size < 2 else batch_size * torch.cuda.device_count()
    print(f'{PREFIX} Selcted Batch Size - {batch_size} (0.9 * batch size for a gpu * gpu_num)')

    return batch_size


def get_batch_size_for_gpu(model, ch, imgsz, amp):
    with torch.cuda.amp.autocast(amp):
        return autobatch(model, ch, imgsz)


def autobatch(model, ch, imgsz, batch_size=4):
    if torch.cuda.is_available():
       num_dev = torch.cuda.device_count()
    else:
       print(f'{PREFIX} CUDA not detected, using default CPU batch size {batch_size}')
       return batch_size

    device = torch.device(f'cuda:0')
    model.to(device)
    model.train()
    batch_size = 2
    while True:
        img = torch.zeros(batch_size, ch, imgsz, imgsz).float()
        img = img.to(device)
        try:
            y = model(img)
            y = y[1] if isinstance(y, list) else y
            loss = y.mean()
            loss.backward() # need to free the variables of the graph
            if DEBUG: print(f'{PREFIX} success: ', batch_size)
            batch_size = batch_size * 2
            del img
        except RuntimeError as e:
            if DEBUG: print(f'{PREFIX} fail: ', batch_size)
            final_batch_size = int(batch_size / 2.)
            del img
            break

    torch.cuda.empty_cache()
    test_func = TestFuncGen(model, ch, imgsz)
    final_batch_size = binary_search(final_batch_size, batch_size, test_func, want_to_get=True)

    return final_batch_size
