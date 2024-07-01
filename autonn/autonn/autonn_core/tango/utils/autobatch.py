# Simplified Version of autobatch.py of Yolov5, AGPL-3.0 license
import torch
import gc
import logging

from copy import deepcopy
from tango.main import status_update
from tango.utils.general import colorstr


DEBUG = False
PREFIX = colorstr('AutoBatch: ')

logger = logging.getLogger(__name__)

class TestFuncGen:
    def __init__(self, model, ch, imgsz):
        self.model = model
        self.ch = ch
        self.imgsz = imgsz

    def __call__(self, batch_size):
        img = torch.zeros(batch_size, self.ch, self.imgsz, self.imgsz).float()
        img = img.to(next(self.model.parameters()).device)

        try:
            y = self.model(img)
            y = y[1] if isinstance(y, list) else y
            loss = y.mean()
            loss.backward() # need to free the variables of the graph
            del img
            return True
        except RuntimeError as e:
            del img
            return False

def binary_search(uid, pid, low, high, test_func, want_to_get):
    logger.info(f'{PREFIX} Start Binary Search')
    low_result = test_func(low)
    high_result = test_func(high)

    batchsize_content = {}
    while True:
        next_test = int((low + high) / 2.)
        if next_test==low or next_test==high:
            print(f'{PREFIX} The result of Binary Search: {next_test}')
            return low if low_result==want_to_get else high

        judge = test_func(next_test)
        if judge==low_result:
            low = next_test
            low_result = judge
            if DEBUG: logger.debug(f'{PREFIX} low: {low} / high: {high}')
        elif judge == high_result:
            high = next_test
            high_result = judge
            if DEBUG: logger.debug(f'{PREFIX} low: {low} / high: {high}')

        batchsize_content['low'] = low
        batchsize_content['high'] = high
        status_update(uid, pid,
                      update_id="batchsize",
                      update_content=batchsize_content)

def get_batch_size_for_gpu(uid, pid, model, ch, imgsz, amp_enabled=True):
    with torch.cuda.amp.autocast(enabled=amp_enabled):
        return autobatch(uid, pid, model, ch, imgsz)

def autobatch(uid, pid, model, ch, imgsz, batch_size=16):
    # prefix = colorstr('AutoBatch: ')
    if torch.cuda.is_available():
       num_dev = torch.cuda.device_count() 
    else:
       logger.info(f'{PREFIX} CUDA not detected, using default CPU batch size {batch_size}')
       return batch_size

    device = torch.device(f'cuda:0')
    model.to(device)
    model.train()

    batchsize_content = {}
    batch_size = 2
    while True:
        img = torch.zeros(batch_size, ch, imgsz, imgsz).float()
        img = img.to(device)
        try:
            y = model(img)
            y = y[1] if isinstance(y, list) else y
            loss = y.mean()
            loss.backward() # need to free the variables of the graph
            if DEBUG: logger.debug(f'{PREFIX} success: ', batch_size)
            batchsize_content['low'] = batch_size
            status_update(uid, pid,
                          update_id="batchsize",
                          update_content=batchsize_content)
            batch_size = batch_size * 2
            del img
        except RuntimeError as e:
            if DEBUG: logger.debug(f'{PREFIX} fail: ', batch_size)
            final_batch_size = int(batch_size / 2.)
            batchsize_content['low'] = final_batch_size
            batchsize_content['high'] = batch_size
            status_update(uid, pid,
                          update_id="batchsize",
                          update_content=batchsize_content)
            del img
            break
    torch.cuda.empty_cache()

    test_func = TestFuncGen(model, ch, imgsz)
    final_batch_size = binary_search(uid, pid, final_batch_size, batch_size, test_func, want_to_get=True)
    torch.cuda.empty_cache()
    gc.collect()

    return final_batch_size * num_dev


