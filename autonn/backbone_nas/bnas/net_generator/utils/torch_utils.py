# References:
# [1] YOLOv5 🚀 by Ultralytics, GPL-3.0 license
'''
utils related to torch
'''

import time
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn as nn


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    '''
    Decorator to make all processes
    in distributed training wait
    for each local_master to do something
    '''
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])


def copy_attr(_a, _b, include=(), exclude=()):
    '''
    Copy attributes from b to a,
    options to only include [...] and to exclude [...]
    '''
    for _k, _v in _b.__dict__.items():
        if ((include and _k not in include)
                or _k.startswith('_') or _k in exclude):
            continue
        setattr(_a, _k, _v)


def time_sync():
    '''PyTorch-accurate time'''
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def fuse_conv_and_bn(conv, _bn):
    '''
    Fuse Conv2d() and BatchNorm2d() layers
    https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    '''
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True
                          ).requires_grad_(False).to(conv.weight.device)

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(_bn.weight.div(torch.sqrt(_bn.eps + _bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(
        0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = _bn.bias - \
        _bn.weight.mul(_bn.running_mean).div(
            torch.sqrt(_bn.running_var + _bn.eps))
    fusedconv.bias.copy_(
        torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def initialize_weights(model):
    '''init weights'''
    for _m in model.modules():
        _t = type(_m)
        if _t is nn.Conv2d:
            # nn.init.kaiming_normal_(m.weight, \
            # mode='fan_out', nonlinearity='relu')
            pass
        elif _t is nn.BatchNorm2d:
            _m.eps = 1e-3
            _m.momentum = 0.03
        elif _t in [nn.Hardswish, nn.LeakyReLU,
                    nn.ReLU, nn.ReLU6, nn.SiLU]:
            _m.inplace = True


def is_parallel(model):
    '''
    Returns True if model is of type DP or DDP
    '''
    return isinstance(model,
                      (nn.parallel.DataParallel,
                       nn.parallel.DistributedDataParallel))


def de_parallel(model):
    '''
    De-parallelize a model:
    returns single-GPU model if model is of type DP or DDP
    '''
    return model.module if is_parallel(model) else model
