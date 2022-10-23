'''
nas search
'''

import os
import sys

from easydict import EasyDict as edict

from .datasets import load_dataset
from .models.model import load_models
from .trainers.trainer import train
from .utils.general import (check_dataset, check_img_size,
                            labels_to_class_weights)
from .utils.torch_utils import torch_distributed_zero_first

BASEPATH = os.path.dirname(os.path.abspath(__file__))

if str(BASEPATH) not in sys.path:
    sys.path.append(str(BASEPATH))


# https://pytorch.org/docs/stable/elastic/run.html
LOCAL_RANK = int(os.environ.get('LOCAL_RANK', -1))


def arch_search(data_path, weights, batch_size, device):
    '''arch_search'''
    data_dict = None
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data_path)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']
    _nc = int(data_dict['nc'])  # the number of classes
    names = data_dict['names']
    assert len(
        names) == _nc, f'{len(names)} names found \
            for nc={_nc} dataset in {data_path}'  # check

    base_model, supernet = load_models(weights, _nc, device)

    # Image size
    _gs = max(int(base_model.stride.max()), 32)  # grid size (max stride)
    # verify imgsz is gs-multiple
    imgsz = check_img_size(224, _gs, floor=_gs * 2)

    config = edict()
    config.imgsz = imgsz
    config.batch_size = batch_size
    config.stride = _gs
    config.workers = 8
    config.quad = False

    # config for train-set
    config.data_path = train_path
    config.cache = "disk"
    config.rect = False
    config.pad = 0.0
    config.rank = LOCAL_RANK
    config.prefix = 'train: '
    config.shuffle = False

    # train_loader, dataset = load_dataset(config)
    # mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())
    # max label class
    # nb = len(train_loader)  # number of batches
    # assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data_path}. \
    # Possible class labels are 0-{nc - 1}'

    config.data_path = val_path
    config.cache = "ram"
    config.rect = True
    config.pad = 0.5
    config.rank = -1
    config.prefix = 'val: '

    val_loader, dataset = load_dataset(config)  # [0]

    # yolov5
    base_model = base_model.model[10:]
    base_model.class_weights = labels_to_class_weights(
        dataset.config.labels, _nc).to(device) * _nc
    base_model.nc = _nc
    base_model.names = names

    finalmodel = train(  # train_loader,
        dataset,
        val_loader,
        base_model,
        supernet,
        _nc,
        names,
        device)
    return finalmodel
