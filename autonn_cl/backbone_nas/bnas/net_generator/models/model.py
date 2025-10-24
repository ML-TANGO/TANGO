'''
Load supernet and head
'''

import os

import torch

from utils.downloads import attempt_download
from utils.general import check_suffix, intersect_dicts
from utils.torch_utils import torch_distributed_zero_first

from .supernet.model_zoo import ofa_net
from .yolo import Model

# from models.supernet.search_space import BackBoneMobileNetV3


# https://pytorch.org/docs/stable/elastic/run.html
LOCAL_RANK = int(os.environ.get('LOCAL_RANK', -1))


def load_models(weights, nc, device):
    '''
    Load supernet and head
    '''
    check_suffix(weights, '.pt')  # check weights
    with torch_distributed_zero_first(LOCAL_RANK):
        weights = attempt_download(weights)  # download if not found locally
    # load checkpoint to CPU to avoid CUDA memory leak
    ckpt = torch.load(weights, map_location='cpu')
    
    model = Model(ckpt['model'].yaml, ch=3, nc=nc).to(device)  # create
    exclude = ['anchor']  # exclude keys
    csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
    csd = intersect_dicts(csd, model.state_dict(),
                          exclude=exclude)  # intersect
    model.load_state_dict(csd, strict=False)  # load

    # supernet
    supernet_path = (os.path.dirname(weights) + '/ofa_nets').replace("\\", '/')
    supernet = ofa_net('ofa_mbv3_d234_e346_k357_w1.0',
                       model_dir=supernet_path).cuda()

    return model, supernet
