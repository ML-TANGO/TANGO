from copy import deepcopy

import numpy as np
import socket
import torch

from ..models.common import AutoShape

from .general import LOGGER, emojis
from .torch_utils import profile

def check_online():
    '''
    Check internet connectivity
    '''
    try:
        # check host accessibility
        socket.create_connection(("1.1.1.1", 443), 5)
        return True
    except OSError:
        return False

def check_amp(model, final=False):
    '''
    Check PyTorch Automatic Mixed Precision (AMP) functionality.
    Return True on correct operation
    '''
    def amp_allclose(model, img, final):
        '''
        All close FP32 vs AMP results
        '''
        m = AutoShape(model, final, verbose=False)  # model
        a = m(img).xywhn[0]  # FP32 inference
        m.amp = True
        b = m(img).xywhn[0]  # AMP inference
        # close to 10% absolute tolerance
        print(a.shape, b.shape)
        print(torch.allclose(a, b, atol=0.1))
        return a.shape == b.shape and torch.allclose(a, b, atol=0.1)

    prefix = 'AMP: '
    device = next(model.parameters()).device  # get model device
    if device.type == 'cpu':
        return False  # AMP disabled on CPU
    img = ('https://ultralytics.com/images/bus.jpg'
           if check_online() else np.ones((640, 640, 3)))
    try:
        assert amp_allclose(deepcopy(model), img, final)
        LOGGER.info(emojis(f'{prefix}checks passed ✅'))
        return True
    except Exception:
        help_url = 'https://github.com/ultralytics/yolov5/issues/7908'
        LOGGER.warning(emojis(
            f'{prefix}checks failed ❌, disabling \
                Automatic Mixed Precision. See {help_url}'))
        return False

def check_train_batch_size(model, imgsz=640, amp=True):
    # Check YOLOv5 training batch size
    with torch.cuda.amp.autocast(amp):
        return autobatch(deepcopy(model).train(), imgsz)  # compute optimal batch size

def autobatch(model, imgsz=640, fraction=0.8, batch_size=16):
    # Automatically estimate best YOLOv5 batch size to use `fraction` of available CUDA memory
    # Usage:
    #     import torch
    #     from utils.autobatch import autobatch
    #     model = torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False)
    #     print(autobatch(model))

    # Check device
    prefix = 'AutoBatch: '
    device = next(model.parameters()).device  # get model device
    if device.type == 'cpu':
        return batch_size
    if torch.backends.cudnn.benchmark:
        return batch_size

    # Inspect CUDA memory
    gb = 1 << 30  # bytes to GiB (1024 ** 3)
    d = str(device).upper()  # 'CUDA:0'
    properties = torch.cuda.get_device_properties(device)  # device properties
    t = properties.total_memory / gb  # GiB total
    r = torch.cuda.memory_reserved(device) / gb  # GiB reserved
    a = torch.cuda.memory_allocated(device) / gb  # GiB allocated
    f = t - (r + a)  # GiB free

    # Profile batch sizes
    batch_sizes = [1, 2, 4, 8, 16]
    try:
        img = [torch.empty(b, 3, imgsz, imgsz) for b in batch_sizes]
        results = profile(img, model, n=3, device=device)
    except Exception as e:
        LOGGER.warning(f'{prefix}{e}')

    # Fit a solution
    y = [x[2] for x in results if x]  # memory [2]
    p = np.polyfit(batch_sizes[:len(y)], y, deg=1)  # first degree polynomial fit
    b = int((f * fraction - p[1]) / p[0])  # y intercept (optimal batch size)
    if None in results:  # some sizes failed
        i = results.index(None)  # first fail index
        if b >= batch_sizes[i]:  # y intercept above failure point
            b = batch_sizes[max(i - 1, 0)]  # select prior safe point
    if b < 1 or b > 1024:  # b outside of safe range
        b = batch_size

    fraction = (np.polyval(p, b) + r + a) / t  # actual fraction predicted
    return b