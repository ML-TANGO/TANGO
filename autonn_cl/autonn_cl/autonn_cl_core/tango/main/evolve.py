from pathlib import Path
COMMON_ROOT = Path("/shared/common")
DATASET_ROOT = Path("/shared/datasets")
CORE_DIR = Path(__file__).resolve().parent.parent.parent # /source/autonn_cl_core
CFG_PATH = CORE_DIR / 'tango' / 'common' / 'cfg'

import os
import time
import yaml
import random
import numpy as np
import logging

logger = logging.getLogger(__name__)

from . import status_update
from .finetune import finetune_hyp
from tango.utils.general import (   fitness,
                                    print_mutation
                                )

import argparse


def evolve(proj_info, hyp, opt, data):
    # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
    meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
            'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
            'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
            'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
            'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
            'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
            # 'box': (1, 0.02, 0.2),  # box loss gain
            # 'cls': (1, 0.2, 4.0),  # cls loss gain
            # 'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
            # 'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
            # 'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
            # 'iou_t': (0, 0.1, 0.7),  # IoU training threshold
            # 'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
            # 'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
            # 'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
            # 'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            # 'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            # 'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            # 'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
            # 'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
            # 'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
            # 'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
            # 'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            # 'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
            # 'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
            # 'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
            # 'mixup': (1, 0.0, 1.0),   # image mixup (probability)
            # 'copy_paste': (1, 0.0, 1.0),  # segment copy-paste (probability)
            # 'paste_in': (1, 0.0, 1.0)    # segment copy-paste (probability)
            }

    logger.info(f'\nHPO: Start searching optimal hyperparameters')
    # project information ------------------------------------------------------
    userid = proj_info['userid']
    project_id = proj_info['project_id']
    target = proj_info['target_info'] # PC, Galaxy_S22, etc.
    acc = proj_info['acc'] # cuda, opencl, cpu
    lt = proj_info['learning_type'].lower()
    
    # options ------------------------------------------------------------------
    hpo_yaml = str(CFG_PATH / 'args-hpo.yaml')
    with open(hpo_yaml, 'r') as f:
        hpo_opt = yaml.safe_load(f)
    logger.info(f'HPO: With pretrained model {opt.weights}')
    assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
    
    opt.notest, opt.nosave = True, True  # only test/save final epoch
    opt = vars(opt)
    opt = {**opt, **hpo_opt}
    opt = argparse.Namespace(**opt)

    total_gen = vars(opt).get('num_generation', 3)
    opt.finetune_epochs = vars(opt).get('finetune_epochs', 1)
    basemodel = vars(opt).get('weights', None)

    # hyperparameters ----------------------------------------------------------
    # with open(opt.hyp, errors='ignore') as f:
    #     hyp = yaml.safe_load(f)  # load hyps dict
    #     if 'anchors' not in hyp:  # anchors commented in hyp.yaml
    #         hyp['anchors'] = 3
    hyp_ev = {}
    hyp_ev['lr0'] = hyp['lr0']
    hyp_ev['lrf'] = hyp['lrf']
    hyp_ev['momentum'] = hyp['momentum']
    hyp_ev['weight_decay'] = hyp['weight_decay']
    hyp_ev['warmup_epochs'] = hyp['warmup_epochs']
    hyp_ev['warmup_momentum'] = hyp['warmup_momentum']
    hyp_ev['warmup_bias_lr'] = hyp['warmup_bias_lr']

    # set files to be saved  ---------------------------------------------------
    # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
    yml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # save best result here
    txt_file = Path(opt.save_dir) / 'evolve.txt'
    if opt.bucket:
        os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

    # optimal hyperparameters searching ----------------------------------------
    for gen in range(total_gen):  # generations to evolve
        if txt_file.exists():  # if evolve.txt exists: select best hyps and mutate
            # Select parent(s)
            parent = 'single'  # parent selection method: 'single' or 'weighted'
            x = np.loadtxt(str(txt_file), ndmin=2)
            n = min(5, len(x))  # number of previous results to consider
            x = x[np.argsort(-fitness(x))][:n]  # top n mutations
            w = fitness(x) - fitness(x).min()  # weights
            if parent == 'single' or len(x) == 1:
                # x = x[random.randint(0, n - 1)]  # random selection
                x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
            elif parent == 'weighted':
                x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

            # Mutate
            mp, s = 0.8, 0.2  # mutation probability, sigma
            npr = np.random
            npr.seed(int(time.time()))
            g = np.array([x[0] for x in meta.values()])  # gains 0-1
            ng = len(meta)
            v = np.ones(ng)
            while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
            for i, k in enumerate(hyp_ev.keys()):  # plt.hist(v.ravel(), 300)
                hyp_ev[k] = float(x[i + 7] * v[i])  # mutate
        
        # Constrain to limits
        for k, v in meta.items():
            hyp_ev[k] = max(hyp_ev[k], v[1])  # lower limit
            hyp_ev[k] = min(hyp_ev[k], v[2])  # upper limit
            hyp_ev[k] = round(hyp_ev[k], 5)  # significant digits

        # Report
        hyp['lr0'] = hyp_ev['lr0']
        hyp['lrf'] = hyp_ev['lrf']
        hyp['momentum'] = hyp_ev['momentum']
        hyp['weight_decay'] = hyp_ev['weight_decay']
        hyp['warmup_epochs'] = hyp_ev['warmup_epochs']
        hyp['warmup_momentum'] = hyp_ev['warmup_momentum']
        hyp['warmup_bias_lr'] = hyp_ev['warmup_bias_lr']

        status_update(userid, project_id,
                      update_id='hyperparameter',
                      update_content=hyp.copy())

        # Train mutation
        opt.gen = gen
        results = finetune_hyp(proj_info, basemodel, hyp.copy(), opt, data, tb_writer=None)

        # Write mutation results
        logger.info(f'\nHPO: Generation #{gen+1}/{total_gen}')
        logger.info('_'*150)
        print_mutation(hyp_ev.copy(), results, str(yml_file), str(txt_file)) #, opt.bucket)
        logger.info('_'*150)

    return hyp_ev.copy(), yml_file, txt_file