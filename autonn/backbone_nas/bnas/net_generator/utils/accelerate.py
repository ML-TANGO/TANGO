import numpy as np
import socket
import torch

from ..models.common import AutoShape

from .general import LOGGER, emojis

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

def check_amp(model):
    '''
    Check PyTorch Automatic Mixed Precision (AMP) functionality.
    Return True on correct operation
    '''
    def amp_allclose(model, img):
        '''
        All close FP32 vs AMP results
        '''
        _m = AutoShape(model, verbose=False)  # model
        _a = _m(img).xywhn[0]  # FP32 inference
        _m.amp = True
        _b = _m(img).xywhn[0]  # AMP inference
        # close to 10% absolute tolerance
        return _a.shape == _b.shape and torch.allclose(_a, _b, atol=0.1)

    prefix = 'AMP: '
    device = next(model.parameters()).device  # get model device
    if device.type == 'cpu':
        return False  # AMP disabled on CPU
    img = ('https://ultralytics.com/images/bus.jpg'
           if check_online() else np.ones((640, 640, 3)))
    try:
        assert amp_allclose(model, img)
        LOGGER.info(emojis(f'{prefix}checks passed ✅'))
        return True
    except ValueError:
        help_url = 'https://github.com/ultralytics/yolov5/issues/7908'
        LOGGER.warning(emojis(
            f'{prefix}checks failed ❌, disabling \
                Automatic Mixed Precision. See {help_url}'))
        return False