'''
nas search
'''

import os
import sys

import numpy as np

from .datasets import load_dataset
from .models.model import load_models
from .trainers.trainer import train
from .trainers.eval import fine_tune
from .utils.accelerate import check_amp, check_train_batch_size
from .utils.general import (check_dataset, check_img_size,
                            labels_to_class_weights)
from .utils.torch_utils import torch_distributed_zero_first

BASEPATH = os.path.dirname(os.path.abspath(__file__))

if str(BASEPATH) not in sys.path:
    sys.path.append(str(BASEPATH))


# https://pytorch.org/docs/stable/elastic/run.html
LOCAL_RANK = int(os.environ.get('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def arch_search(
            data_path, 
            weights, 
            batch_size, 
            max_latency,
            pop_size,
            niter,
            device):
    '''arch_search'''
    data_dict = None
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data_path)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = int(data_dict['nc'])  # the number of classes
    names = data_dict['names']
    assert len(
        names) == nc, f'{len(names)} names found \
            for nc={nc} dataset in {data_path}'  # check

    base_model, supernet = load_models(weights, nc, device)
    stride = base_model.stride

    # Image size
    gs = max(int(stride.max()), 32)  # grid size (max stride)
    # verify imgsz is gs-multiple
    imgsz = check_img_size(224, gs, floor=gs * 2)

    # Batch size
    # DDP mode TODO
    if batch_size == -1:  # single-GPU only, estimate best batch size
        amp = check_amp(base_model)  # check AMP
        batch_size = check_train_batch_size(base_model, imgsz, amp)
    
    print("auto batch size: ", batch_size)

    train_loader, dataset = load_dataset(train_path,
                                        imgsz,
                                        batch_size // WORLD_SIZE,
                                        gs,
                                        cache="ram", # or disk
                                        rect=False,
                                        rank=LOCAL_RANK,
                                        workers=8,
                                        image_weights=True,
                                        quad=True,
                                        prefix='train: ',
                                        shuffle=True)

    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data_path}. \
        Possible class labels are 0-{nc - 1}'

    val_loader = load_dataset(val_path,
                            imgsz,
                            batch_size // WORLD_SIZE * 2,
                            gs,
                            cache="ram",
                            rect=True,
                            rank=-1,
                            workers=8 * 2,
                            pad=0.5,
                            prefix='val: ')[0]

    # yolov5
    base_model = base_model.model[10:]
    base_model.class_weights = labels_to_class_weights(
        dataset.labels, nc).to(device) * nc
    base_model.nc = nc
    base_model.names = names
    base_model.stride = stride

    finalmodel = train(
        train_loader,
        val_loader,
        base_model,
        supernet,
        nc,
        names,
        max_latency,
        pop_size,
        niter,
        device)

    # amp = False
    # model = fine_tune(val_loader, model, amp)

    return finalmodel
