'''
main.py
'''

import os
from pathlib import Path

from .search import arch_search

def run_nas(
        data_path,
        batch_size = -1,
        max_latency = 25,
        pop_size = 4,
        niter = 5,
        device=0,
):
    '''
    main func
    '''
    base_model_weights = 'bnas/media/temp_files/model/yolov5s.pt'
    best_det_model = arch_search(
        data_path, 
        base_model_weights, 
        batch_size,
        max_latency,
        pop_size,
        niter,
        device)


    return best_det_model
