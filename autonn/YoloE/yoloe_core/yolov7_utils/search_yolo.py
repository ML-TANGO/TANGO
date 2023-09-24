import argparse
import os
import time
import yaml
from pathlib import Path
import sys
from copy import deepcopy
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # to run '$ python *.py' files in subdirectories
from models.experimental import attempt_load
from utils.torch_utils import ModelEMA, select_device, is_parallel

# from .models.experimental import attempt_load
# from utils.torch_utils import select_device

from nas.search_algorithm import EvolutionFinder
from nas.predictors.efficiency_predictor import LatencyPredictor
from nas.predictors.accuracy_predictor import AccuracyCalculator
# from nas.supernet.supernet_yolov7 import YOLOSuperNet

def run_search(opt):
    search_yml = Path(os.path.dirname(__file__)) / 'cfg' / 'supernet' / 'search.yml'
    with open(search_yml, 'r') as f:
        search_opt = yaml.safe_load(f)

    opt = vars(opt)
    opt = {**opt, **search_opt}
    opt = argparse.Namespace(**opt)

    constraint_type, efficiency_constraint, accuracy_predictor = \
        opt.constraint_type, opt.efficiency_constraint, opt.accuracy_predictor
    
    device = select_device(opt.device)

    efficiency_predictor = LatencyPredictor(target="galaxy10", device=device)

    # Create model
    # /shared/models/yolov7_supernet.pt
    opt.weights = str(Path('/shared/models') / opt.weights)
    supernet = attempt_load(opt.weights, map_location=device)

    # opt parameters for fintuning
    accuracy_predictor = AccuracyCalculator(opt, supernet)

    # build the evolution finder
    finder = EvolutionFinder(
        constraint_type=constraint_type, 
        efficiency_constraint=efficiency_constraint, 
        efficiency_predictor=efficiency_predictor, 
        accuracy_predictor=accuracy_predictor
    )

    # start searching
    result_list = []
    for flops in [efficiency_constraint]:   # iterate 1
        st = time.time()
        finder.set_efficiency_constraint(flops)
        best_valids, best_info = finder.run_evolution_search()
        ed = time.time()
        # print('Found best architecture at flops <= %.2f M in %.2f seconds! It achieves %.2f%s predicted accuracy with %.2f MFLOPs.' % (flops, ed-st, best_info[0] * 100, '%', best_info[-1]))
        result_list.append(best_info)
        
    # save model into yaml
    # for i, result in enumerate(result_list):
    #     best_depth = result[1]['d']
    #     # Active subet
    #     supernet.set_active_subnet(best_depth)
    #     sample_config = supernet.get_active_net_config()
    #     # Create yaml file
    #     yaml_file = './yaml/yolov7_searched_%d.yml' % i
    #     if not os.path.exists(os.path.dirname(yaml_file)):
    #         os.makedirs(os.path.dirname(yaml_file))
    #     with open(yaml_file, 'w') as f:
    #         yaml.dump(sample_config, f, sort_keys=False)
    #     print(f'# Depth: {best_depth} | Saved best {i} model\'s config in {yaml_file}')

    final = result_list[0][3]

    return final