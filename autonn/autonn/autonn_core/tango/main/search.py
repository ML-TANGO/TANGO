import argparse
import os
import time
import yaml

from pathlib import Path
COMMON_ROOT = Path("/shared/common")
DATASET_ROOT = Path("/shared/datasets")
CORE_DIR = Path(__file__).resolve().parent.parent.parent # /source/autonn_core
CFG_PATH = CORE_DIR / 'tango' / 'common' / 'cfg'

import sys
from copy import deepcopy
import logging
import torch

from tango.common.models.experimental import attempt_load
from tango.utils.torch_utils import select_device
from tango.nas.search_algorithm import EvolutionFinder
from tango.nas.predictors.efficiency_predictor import LatencyPredictor
from tango.nas.predictors.accuracy_predictor import AccuracyCalculator


logger = logging.getLogger(__name__)

def search(proj_info, hyp, opt, data_dict, model):
    # project information ------------------------------------------------------
    userid = proj_info['userid']
    project_id = proj_info['project_id']
    target = proj_info['target_info'] # PC, Galaxy_S22, etc.
    acc = proj_info['acc'] # cuda, opencl, cpu

    if not 'Galaxy' in target:
        logger.warn(f'not supported target: {target}')
        return model

    # options ------------------------------------------------------------------
    search_yml = str(CFG_PATH / 'args-search.yaml')
    with open(search_yml, 'r') as f:
        search_opt = yaml.safe_load(f)

    constraint_type, efficiency_constraint, accuracy_predictor = \
        search_opt['constraint_type'], search_opt['efficiency_constraint'], search_opt['accuracy_predictor']
    del search_opt['constraint_type'], search_opt['efficiency_constraint'], \
        search_opt['accuracy_predictor'], search_opt['efficiency_predictor']

    opt = vars(opt)
    opt = {**opt, **search_opt}
    opt = argparse.Namespace(**opt)

    # device -------------------------------------------------------------------
    device = select_device(opt.device)

    # create model -------------------------------------------------------------
    if model:
        opt.weights = str(model)
    elif opt.weights:
        opt.weights = CFG_PATH / opt.weights
    supernet = attempt_load(opt.weights, map_location=device, fused=False) # make sure it is not a fused model

    # build latency predictor --------------------------------------------------
    efficiency_predictor = LatencyPredictor(target=target, target_acc=acc, device=device)

    # build accuracy predictor -------------------------------------------------
    accuracy_predictor = AccuracyCalculator(proj_info, hyp, opt, data_dict, supernet)

    # build the evolution finder -----------------------------------------------
    finder = EvolutionFinder(
        constraint_type=constraint_type,
        efficiency_constraint=efficiency_constraint,
        efficiency_predictor=efficiency_predictor,
        accuracy_predictor=accuracy_predictor,
        **vars(opt)
    )

    # network architecture searching -------------------------------------------
    logger.info(f"Start NAS process...")
    result_list = []
    efficiency_constraint = efficiency_constraint \
        if isinstance(efficiency_constraint, list) else [efficiency_constraint]
    for flops in efficiency_constraint:   # iterate 1
        st = time.time()
        finder.set_efficiency_constraint(flops)
        best_valids, best_info = finder.run_evolution_search(verbose=True)
        # best_valids = [ best_acc_history ... ]
        # best_info = (acc, d, flops, subnet_pt)
        ed = time.time()
        logger.info(f"found best architecture at flops <= {flops:.2f} MFLOPS in {ed-st:.2f} seconds! "
                    f"{best_info[0]*100:.2f}% predicted accuracy with {best_info[2]:.2f} MFLOPS.")
        result_list.append(best_info)
    logger.info(f"Complete NAS process")

    # save weights and configs -------------------------------------------------
    for i, (ec, result) in enumerate(zip(efficiency_constraint, result_list)):
        # load weights & config
        best_subnet_pt = result[3]
        best_subnet = attempt_load(best_subnet_pt, map_location=device, fused=True)
        best_subnet_config = best_subnet.yaml
        best_depth = best_subnet_config['depth_list']

        # save weights & config
        PROJ_PATH = COMMON_ROOT / userid / project_id
        yaml_file = str(PROJ_PATH / 'best_search.yaml')
        with open(yaml_file, 'w') as f:
            yaml.dump(best_subnet_config, f, sort_keys=False)
        logger.info(f'# Depth: {best_depth} | Saved best {i} model\'s config in {yaml_file}')


    final = result_list[0][3] # str: path/to/best/subnet/pt
    return final, yaml_file
