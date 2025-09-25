import os
import shutil
from pathlib import Path
COMMON_ROOT = Path("/shared/common")
DATASET_ROOT = Path("/shared/datasets")
CORE_DIR = Path(__file__).resolve().parent.parent.parent # /source/autonn_cl_core
CFG_PATH = CORE_DIR / 'tango' / 'common' / 'cfg'

import logging
logger = logging.getLogger(__name__)

import nni
import torch
from torch import nn

from . import status_update, Info

from tango.utils.torch_utils import (   ModelEMA,
                                        EarlyStopping,
                                        select_device_and_info,
                                        intersect_dicts,
                                        torch_distributed_zero_first,
                                        is_parallel,
                                        de_parallel,
                                        time_synchronized
                                    )

def train_with_hparams(proj_info, hyp, opt, data_dict):
    # Options ------------------------------------------------------------------
    save_dir, epochs, batch_size, weights, rank, local_rank, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.global_rank, opt.local_rank, opt.freeze

    userid, project_id, task, nas, target, target_acc = \
        proj_info['userid'], proj_info['project_id'], proj_info['task_type'], \
        proj_info['nas'], proj_info['target_info'], proj_info['acc']

    # save internally
    info = Info.objects.get(userid=userid, project_id=project_id)
    info.status = "running"
    info.progress = "hpo"
    info.save()

    # Directories --------------------------------------------------------------
    wdir = save_dir / 'hparams'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    results_file = save_dir / 'hpo_results.txt'

    # CUDA device --------------------------------------------------------------
    device_str = ''
    for i in range(torch.cuda.device_count()):
        device_str = device_str + str(i) + ','
    opt.device = device_str[:-1]
    device, device_info = select_device_and_info(opt.device)
    cuda = device.type != 'cpu'
    nc = int(data_dict.get('nc', 80))
    ch = int(data_dict['ch'])
    name = data_dict['names']

    # Model --------------------------------------------------------------------
    