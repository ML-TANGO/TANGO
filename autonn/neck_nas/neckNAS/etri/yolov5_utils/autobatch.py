"""
Auto-batch utils
"""

from copy import deepcopy

import numpy as np
import torch
from torch.cuda import amp

from yolov5_utils.general import LOGGER, colorstr
from yolov5_utils.torch_utils import profile


def check_train_batch_size(model, imgsz=640):
    # Check YOLOv5 training batch size
    with amp.autocast():
        # compute optimal batch size
        return autobatch(deepcopy(model).train(), imgsz)


def autobatch(model, imgsz=640, fraction=0.9, batch_size=16):
    """ Automatically estimate best batch size to use `fraction`
        of available CUDA memory
    """
    prefix = colorstr('AutoBatch: ')
    LOGGER.info(f'{prefix}Computing optimal batch size for --imgsz {imgsz}')
    device = next(model.parameters()).device  # get model device
    if device.type == 'cpu':
        LOGGER.info(f'{prefix}CUDA not detected,'
                    f' using default CPU batch-size {batch_size}')
        return batch_size

    gb = 1 << 30  # bytes to GiB (1024 ** 3)
    d = str(device).upper()  # 'CUDA:0'
    properties = torch.cuda.get_device_properties(device)  # device properties
    t = properties.total_memory / gb  # (GiB)
    r = torch.cuda.memory_reserved(device) / gb  # (GiB)
    a = torch.cuda.memory_allocated(device) / gb  # (GiB)
    f = t - (r + a)  # free inside reserved
    LOGGER.info(f'{prefix}{d} ({properties.name})'
                f' {t:.2f}G total, {r:.2f}G reserved,'
                f' {a:.2f}G allocated, {f:.2f}G free')

    batch_sizes = [1, 2, 4, 8, 16]
    try:
        img = [torch.zeros(b, 3, imgsz, imgsz) for b in batch_sizes]
        y = profile(img, model, n=3, device=device)
    except Exception as e:
        LOGGER.warning(f'{prefix}{e}')

    y = [x[2] for x in y if x]  # memory [2]
    batch_sizes = batch_sizes[:len(y)]
    p = np.polyfit(batch_sizes, y, deg=1)  # first degree polynomial fit
    b = int((f * fraction - p[1]) / p[0])  # y intercept (optimal batch size)
    LOGGER.info(f'{prefix}Using batch-size {b}'
                f' for {d} {t * fraction:.2f}G/{t:.2f}G'
                f' ({fraction * 100:.0f}%)')
    return b
