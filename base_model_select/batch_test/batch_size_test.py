# Simplified Version of autobatch.py of Yolov5, AGPL-3.0 license
import torch
import gc
import yaml

from copy import deepcopy

from .utils.general import colorstr
from .models.yolo import Model


def run_batch_test(basemodel_yaml, hyp_yaml, imgsz):
    with open(hyp_yaml, 'r') as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    model = Model(basemodel_yaml, ch=3, nc=80, anchors=hyp.get('anchors'))
    batch_size = int(get_batch_size_for_gpu(model, imgsz, amp=True) * 0.8)
    # It assumes that the memory sizes of all gpus in a machine are same.
    # 0.8 is multiplied by batch size to prevent cuda memory error due to a memory leak of yolov7

    del model
    torch.cuda.empty_cache()
    gc.collect()

    batch_size = False if batch_size < 2 else batch_size * torch.cuda.device_count()
    prefix = '[ AutoBatch ]'
    print(f'{prefix} Selcted Batch Size - {batch_size} (0.8 * batch size for a gpu * gpu_num)')

    return batch_size


def get_batch_size_for_gpu(model, imgsz, amp):
    with torch.cuda.amp.autocast(amp):
        return autobatch(model, imgsz)


def autobatch(model, imgsz, batch_size=16):
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
        img = torch.zeros(batch_size, 3, imgsz, imgsz).float()
        img = img.to(device)
        try:
            model(img)
            print(f'{prefix} success: ', batch_size)
            batch_size = batch_size * 2
        except RuntimeError as e:
            print(f'{prefix} fail: ', batch_size)
            final_batch_size = int(batch_size / 2.)
            del img
            break

    return final_batch_size
