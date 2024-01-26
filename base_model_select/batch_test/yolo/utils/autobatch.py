# Simplified Version of autobatch.py of Yolov5, AGPL-3.0 license

import torch
import gc

from copy import deepcopy
from .general import colorstr


def get_batch_size_for_gpu(model, imgsz, amp):
    with torch.cuda.amp.autocast(amp):
        return autobatch(model, imgsz)

def autobatch(model, imgsz, batch_size=16):
    prefix = colorstr('AutoBatch: ')
    if torch.cuda.is_available():
       num_dev = torch.cuda.device_count() 
    else:
       print(f'{prefix} CUDA not detected, using default CPU batch size {batch_size}')
       return batch_size

    batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256]
    min_b = 9999
    img = [torch.empty(b, 3, imgsz, imgsz) for b in batch_sizes]
    # for device_idx in range(num_dev):						## TODO
    for device_idx in range(1):
        device = torch.device(f'cuda:{device_idx}')
        print(device)
        model.to(device)
        model.train()
        results = []
        for ii, img_dummy in enumerate(img):
            img_dummy_ = img_dummy.to(device)
            print(next(model.parameters()).device, img_dummy_.device)
            try:
                model(img_dummy_)
                results.append(True)
                print('success: ', ii)
            except RuntimeError as e:
                print(e)
                results.append(False)
                print('fail: ', ii)
                del img
                break
        
        if False in results:
            i = results.index(False)
            b = batch_sizes[max(i-1, 0)]
        else:
            b = max(batch_sizes)
        min_b = min(b, min_b)

    return min_b 								## TODO
