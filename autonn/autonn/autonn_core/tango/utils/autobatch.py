# Simplified Version of autobatch.py of Yolov5, AGPL-3.0 license
import torch
import gc
import logging

from copy import deepcopy
from tango.main import status_update
from tango.utils.general import colorstr


DEBUG = True
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
            # y = y[1] if isinstance(y, list) else y
            # y = y[-1] if isinstance(y, list) else y # one more time (just in case dual or triple heads: yolov9)
            # loss = y.mean()
            loss = (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum() # from yolov9 code
            loss.backward() # need to free the variables of the graph
            del img
            return True
        except RuntimeError as e:
            del img
            return False

def binary_search(uid, pid, low, high, test_func, want_to_get):
    logger.info(f'{PREFIX}Start Binary Search')
    low_result = test_func(low)
    high_result = test_func(high)

    batchsize_content = {}
    while True:
        next_test = int((low + high) / 2.)
        if next_test==low or next_test==high:
            logger.info(f'{PREFIX}Binary Search Complete: Max Batch Size = {next_test}')
            return low if low_result==want_to_get else high

        judge = test_func(next_test)
        if judge==low_result:
            low = next_test
            low_result = judge
            if DEBUG:
                low_str = str(colorstr("underline", str(low)))
                logger.info(f'{PREFIX} {low_str:>4s} ⭕ | {str(high):>4s} ❌')
        elif judge == high_result:
            high = next_test
            high_result = judge
            if DEBUG:
                high_str = str(colorstr("underline", str(high)))
                logger.info(f'{PREFIX}{str(low):>4s} ⭕ |  {high_str:>4s} ❌')

        batchsize_content['low'] = low
        batchsize_content['high'] = high
        status_update(uid, pid,
                      update_id="batchsize",
                      update_content=batchsize_content)

def get_batch_size_for_gpu(uid, pid, model, ch, imgsz, bs_factor=0.8, amp_enabled=True, max_search=True):
    with torch.cuda.amp.autocast(enabled=amp_enabled):
        return autobatch(uid, pid, model, ch, imgsz, bs_factor, max_search=max_search)

def autobatch(uid, pid, model, ch, imgsz, bs_factor=0.8, batch_size=16, max_search=True):
    # Check device
    # device = torch.device(f'cuda:0')
    device = next(model.parameters()).device
    if device != 'cpu' and torch.cuda.is_available():
       num_dev = torch.cuda.device_count() 
    else:
       logger.info(f'{PREFIX}CUDA not detected, using default CPU batch size {batch_size}')
       return batch_size
    if torch.backends.cudnn.benchmark:
        logger.info(f'{PREFIX}requires cudnn.benchmark=False, using default batch size {batch_size}')
        return batch_size

    # Inspect CUDA memory
    gb = 1 << 30 # bytes in GiB (1024**3)
    d = str(device).upper() # 'CUDA:0'
    properties = torch.cuda.get_device_properties(device)
    t = properties.total_memory / gb            # total
    r = torch.cuda.memory_reserved(device) / gb # reserved
    a = torch.cuda.memory_allocated(device)/ gb # allocated
    f = t - (r + a)                             # free = total - (reserved + allocated)
    logger.info(f'\n{PREFIX}{d} ({properties.name}) {t:.2f}G total, {r:.2f}G reserved, {a:.2f}G allocated, {f:.2f}G free')


    # model.to(device)
    model.train()
    batchsize_content = {}
    batch_size = 2 # 2, 4, 8, 16, 32, 64, 128, 256, ...
    while True:
        img = torch.zeros(batch_size, ch, imgsz, imgsz).float()
        img = img.to(device)
        try:
            y = model(img)
            # y = y[1] if isinstance(y, list) else y
            # y = y[-1] if isinstance(y, list) else y # one more time (just in case dual or triple heads: yolov9)
            ''' in v9 case,
                training :list   =   [d1, d2]
                inference:tuple  = ( [c1, c2],
                                     [d2, d2] )
                export   :tensor =   c1 ⨁ c2
                where, dx:list = [ (64,144,80,80),
                                   (64,144,40,40),
                                   (64,144,20,20) ]
                       cx:list = [ (64,144, 8400),
                                   (64,144, 8400) ]

                in v7 case,
                training :list  = x
                inference:tuple = ( z, x )
                export   :list  = x

                where, x :list = [ (64,3,80,80,85),
                                   (64,3,40,40,85),
                                   (64,3,20,20,85) ]
                       z :tensor = (64, 25200, 85)
            '''
            # loss = y.mean()
            loss = (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum() # from yolov9 code
            loss.backward() # need to free the variables of the graph
            if DEBUG: logger.info(f'{PREFIX}{batch_size:>4.0f} ⭕ success')
            batchsize_content['low'] = batch_size
            status_update(uid, pid,
                          update_id="batchsize",
                          update_content=batchsize_content)
            batch_size = batch_size * 2
            del img
        except RuntimeError as e:
            if DEBUG: logger.info(f'{PREFIX}{batch_size:>4.0f} ❌ fail')
            final_batch_size = int(batch_size / 2.)
            batchsize_content['low'] = final_batch_size
            batchsize_content['high'] = batch_size
            status_update(uid, pid,
                          update_id="batchsize",
                          update_content=batchsize_content)
            del img
            break
    torch.cuda.empty_cache()

    if max_search: # search maximum batch size (allow size other than multiple of 2)
        test_func = TestFuncGen(model, ch, imgsz)
        final_batch_size = binary_search(uid, pid, final_batch_size, batch_size, test_func, want_to_get=True)
        logger.info(f'{PREFIX}max {final_batch_size} x margin factor {bs_factor}')
        final_batch_size *= bs_factor # need some spare
        torch.cuda.empty_cache()

    gc.collect()
    logger.info(f'{PREFIX}= {int(final_batch_size * num_dev)}')
    return final_batch_size * num_dev


