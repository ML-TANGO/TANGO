import os
import glob
import shutil
import gc
import shutil
import argparse
import yaml
import logging
import subprocess

from pathlib import Path
COMMON_ROOT = Path("/shared/common")
DATASET_ROOT = Path("/shared/datasets")
CORE_DIR = Path(__file__).resolve().parent.parent.parent # /source/autonn_core
CFG_PATH = CORE_DIR / 'tango' / 'common' / 'cfg'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

from . import status_update, Info
from .train import train
from .search import search
from .evolve import evolve
from .visualize import BasemodelViewer
from .export import export_weight, export_config
from tango.utils.general import (   increment_path,
                                    get_latest_run,
                                    check_file,
                                    set_logging,
                                    strip_optimizer,
                                    colorstr        )
from tango.utils.plots import plot_evolution
from tango.utils.wandb_logging.wandb_utils import check_wandb_resume


TASK_TO_MODEL_TABLE = {
    "detection7": "yolov7",
    "detection": "yolov9",
    "classification": "resnet",
    "classification-c": "resnetc",
}

MODEL_TO_SIZE_TABLE = {
    # "yolov7": {
    #     "cloud": "-d6",
    #     "k8s": "-e6",
    #     "k8sjetsonnano": "-e6",
    #     "pcweb": "-w6",
    #     "pc": "-w6",
    #     "jetsonagxorin": "x",
    #     "jetsonagxxavier": "x",
    #     "jetsonnano": "x",
    #     "galaxys22": "-supernet",
    #     "odroidn2": "-tiny",
    # },
    # "yolov9": {
    #     "cloud": "-e",
    #     "k8s": "-c",
    #     "k8sjetsonnano": "-c",
    #     "pcweb": "-m",
    #     "pc": "-m",
    #     "jetsonagxorin": "-s",
    #     "jetsonagxxavier": "-s",
    #     "jetsonnano": "-s",
    #     "galaxys22": "-supernet",
    #     "odroidn2": "-t",
    # },
    # "resnet": {
    #     "cloud": "152",
    #     "k8s": "101",
    #     "k8sjetsonnano": "101",
    #     "pcweb": "50",
    #     "pc": "50",
    #     "jetsonagxorin": "34",
    #     "jetsonagxxavier": "34",
    #     "jetsonnano": "34",
    #     "galaxys22": "18",
    #     "odroidn2": "18",
    # },
    # "resnetc": {
    #     "cloud": "200",
    #     "k8s": "152",
    #     "k8sjetsonnano": "110",
    #     "pcweb": "56",
    #     "pc": "56",
    #     "jetsonagxorin": "44",
    #     "jetsonagxxavier": "44",
    #     "jetsonnano": "44",
    #     "galaxys22": "32",
    #     "odroidn2": "20",
    # },
    "yolov7": {
        "XXL": "-e6e",      # >  20.0G
        "XL" : "-d6",       # <= 20.0G
        "L"  : "-e6",       # <= 16.0G
        "M"  : "-w6",       # <= 12.0G
        "MS" : "x",         # <=  8.0G
        "S"  : "",          # <=  6.0G
        "T"  : "-tiny",     # <=  4.0G
        "NAS": "-supernet"
    },
    "yolov9": {
        "XXL": "-e",        # >  20.0G
        "XL" : "-e",        # <= 20.0G
        "L"  : "-c",        # <= 16.0G
        "M"  : "-m",        # <= 12.0G
        "MS" : "-m",        # <=  8.0G
        "S"  : "-s",        # <=  6.0G
        "T"  : "-t",        # <=  4.0G
        "NAS": "-supernet"
    },
    "resnet": {
        "XXL": "152",
        "XL" : "152",
        "L"  : "101",
        "M"  : "50",
        "MS" : "50",
        "S"  : "34",
        "T"  : "18",
    },
    "resnetc": {
        "XXL": "200",
        "XL" : "152",
        "L"  : "110",
        "M"  : "56",
        "MS" : "44",
        "S"  : "32",
        "T"  : "20",
    }
}

logger = logging.getLogger(__name__)


def get_user_requirements(userid, projid, resume=False):
    """
    Get user requirements(dataset, project, hyperparameters, arguments)
    """
    # ----------------------------- project_info -------------------------------
    PROJ_PATH = COMMON_ROOT / userid / projid
    proj_yaml_path = PROJ_PATH / "project_info.yaml"
    with open(proj_yaml_path, "r") as f:
        proj_info_dict = yaml.safe_load(f)
    status_update(userid, projid,
                  update_id="project_info",
                  update_content=proj_info_dict)

    target = proj_info_dict['target_info'].replace('-', '').replace('_', '').lower()
    device = proj_info_dict['acc']
    task = proj_info_dict['task_type'].lower()

    info = Info.objects.get(userid=userid, project_id=projid)
    print('DB checking...')
    info.print()
    info.target = target
    info.device = device
    info.dataset = proj_info_dict['dataset']
    info.task = task
    info.status = "running"
    info.progress = "setting"
    info.model_viz = "not ready"
    info.save()
    # print('read yaml file and store project infomation')
    # info.print()

    lt = proj_info_dict['learning_type'].lower()
    skip_bms = False
    if os.path.isfile(info.best_net):
        if lt == 'incremental' or lt == 'transfer' or lt == 'finetune' or lt == 'hpo':
            logger.info(f'Project Info: Pretrained model {info.best_net} exists.\n'
                        f'              BMS will be skipped.')
            skip_bms = True
        elif lt == 'normal':
            bak_dir = backup_previous_work(info.best_net)
            logger.info(f'Project Info: Pretrained model {info.best_net}\n'
                        f'              moved to {bak_dir}/...')
            info.best_net = ''
            info.save()
        else:
            logger.warn(f'Project Info: Pretrained model {info.best_net} exists.\n'
                        f'              But learning type {lt} is unknown, '
                        f'so it will be overwritten by the end of training.')

    # ----------------------------- TANGO+CHAT ---------------------------------
    if task == 'chat':
        logger.info(f"Project Info: Run LLM model loader")
        logger.info(f"Project Info: http://localhost:8101 to see Tango+Chat browser")
        run_chat_browser = str(CORE_DIR / 'tangochat' / 'browser.py')
        cmd = ["streamlit", "run", run_chat_browser, "--server.port", "8101", "--browser.gatherUsageStats", "false"]
        # subprocess.check_call(["streamlit", "run", run_chat_browser])
        p = subprocess.Popen(cmd, 
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.STDOUT, 
                             cwd=CORE_DIR,
                             universal_newlines=False)
        while p.poll() == None:
            out = p.stdout.readline()
            out_str = str(out, encoding='utf-8')
            out_str = out_str.split('\n')[0]
            if len(out_str) > 0:
                logger.info(f"Chat: {out_str}")
            if out_str == 'Completed':
                p.terminate()
                break
        return proj_info_dict, None, None, None, None

    # ----------------------------- dataset ------------------------------------
    dataset_on_proj = proj_info_dict["dataset"]
    if os.path.isdir(str(DATASET_ROOT / dataset_on_proj)):
        dataset_yaml_path = DATASET_ROOT / dataset_on_proj / "dataset.yaml"
    else:
        logger.warn(f"There is no {DATASET_ROOT}/{dataset_on_proj}. "
                    f"Instead embedded COCO128 dataset will be used.")
        dataset_on_proj = 'coco128'
        dataset_yaml_path = CORE_DIR / 'datasets' / 'coco128' / 'dataset.yaml'
    with open(dataset_yaml_path) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    data_dict['dataset_name'] = dataset_on_proj

    # ---------------------------- basemodel -----------------------------------
    if skip_bms:
        basemodel_yaml_path = str(PROJ_PATH / 'basemodel.yaml')
        basemodel = {
            "model_name": info.model_type,
            "model_size": info.model_size,
        }
        # save internally
        info = Info.objects.get(userid=userid, project_id=projid)
        info.model_viz = "ready"
        info.save()
    else:
        basemodel_yaml_path, basemodel = base_model_select( userid,
                                                            projid,
                                                            proj_info_dict,
                                                            data_dict)
    with open(basemodel_yaml_path, "r") as f:
        basemodel_dict = yaml.load(f, Loader=yaml.SafeLoader)
    basemodel_dict['hyp'] = 'p5' if basemodel_dict['hyp'] == 'tiny' \
                                 else basemodel_dict['hyp']
    proj_info_dict['nas'] = True if basemodel['model_size'] == '-supernet' else False

    # --------------------------- hyperparameter -------------------------------
    hyp_yaml_path = CFG_PATH / f"hyp.scratch.{basemodel_dict['hyp']}.yaml"
    with open(hyp_yaml_path) as f:
        hyp_dict = yaml.safe_load(f)
    # hyp_dict['lrc'] = hyp_dict['lr0']
    hyp_dict['anchor_t'] = 5.0 # from yolov9
    status_update(userid, projid,
                  update_id="hyperparameter",
                  update_content=hyp_dict)

    # ---------------------------- arguments -----------------------------------
    # task = proj_info_dict['task_type']
    # trick for nas (temp.)
    _task =  'detection7' if basemodel['model_name'] == 'YOLOV7' or target == 'galaxys22' else task
    opt_yaml_path = CFG_PATH / f'args-{_task}.yaml'

    if skip_bms:
        opt_yaml_path = str(PROJ_PATH / 'autonn' / 'opt.yaml')
        logger.info(f'Porject Info: previous opt.yaml loading..')
        assert os.path.isfile(opt_yaml_path)

    with open(opt_yaml_path, encoding='utf-8') as f:
        opt = argparse.Namespace(**yaml.safe_load(f))

    weights = vars(opt).get('weights', None)
    if weights:
        logger.info(f'Project Info: pretrained model = {weights}')

    # check resume
    logger.info(f'Project Info: CUDA OOM: resume? {resume}')
    if resume:
        opt.resume = True # str(PROJ_PATH / 'autonn' / 'weights' / 'last.pt')
        opt.bs_factor = info.batch_multiplier - 0.1

    # check incremental
    if lt == 'incremental':
        best_acc = vars(opt).get('best_acc', None)
        logger.info(f'Project Info: Incremental: best accuracy? {best_acc}')
        if best_acc == None:
            opt.best_acc = 0.0

    '''
    WORLD_SIZE: the total number of processes participating in the traininig 
                (typically the number of GPUs across all nodes)
    RANK:       global rank, the unique ID for each process involved in the training
                (ranging from 0 to WORLD_SIZE-1)
    LOCAL_RANK: the ID under the same node
                (often GPU is specified with LOCAL_RANK, but not necessarily 1-to-1)  
    '''
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

    opt.project = str(PROJ_PATH)
    opt.data = str(dataset_yaml_path)
    opt.cfg = str(basemodel_yaml_path)
    opt.hyp = str(hyp_yaml_path)
    opt.img_size = [basemodel_dict['imgsz'], basemodel_dict['imgsz']]
    opt_content = vars(opt)
    status_update(userid, projid,
                  update_id="arguments",
                  update_content=opt_content)

    status_update(userid, projid,
                  update_id="basemodel",
                  update_content=basemodel)

    return proj_info_dict, opt, hyp_dict, basemodel_dict, data_dict


def base_model_select(userid, project_id, proj_info, data, manual_select=False):
    # read condition
    task = proj_info['task_type']
    target = proj_info['target_info'].replace('-', '').replace('_', '').lower()

    target_acc = proj_info['acc']
    target_mem = proj_info['memory']
    target_mem = float(target_mem)
    if target_acc == 'cpu':
        model_size = "T"
    else:
        if target_mem <= 4.0:
            model_size = "T"
        elif target_mem <= 6.0:
            model_size = "S"
        elif target_mem <= 8.0:
            model_size = "MS"
        elif target_mem <= 12.0:
            model_size = "M"
        elif target_mem <= 16.0:
            model_size = "L"
        elif target_mem <= 20.0:
            model_size = "XL"
        else: # target_mem > 20G
            model_size = "XXL"

    logger.info(f"\nBMS: Based on memory = {target_mem}G, acc = {target_acc}")
    # for supporting both v7 and v9, resent-c and resnet
    task_ = 'classification-c' if data['nc'] <= 10 and task == 'classification' else task
    task_ = 'detection7' if data['nc'] <= 50 and task == 'detection' else task

    # for supporting nas
    if target == 'galaxys22' and task == 'detection':
        task_ = 'detection7'
        model_size = 'NAS'

    # look up table
    if not manual_select:
        model = TASK_TO_MODEL_TABLE[task_]
        # size = MODEL_TO_SIZE_TABLE[model][target]
        size = MODEL_TO_SIZE_TABLE[model][model_size]
    else:
        model = manual_select['model']
        size = manual_select['size']

    # resnet and resnet-c is in the same directory
    dirname = model if task_ != 'classification-c' else model[:-1] # remove 'c' from 'resnetc'
    filename = f'{model}{size}.yaml'

    # store basemodel.yaml
    PROJ_PATH = COMMON_ROOT / userid / project_id
    source_path = f'{CFG_PATH}/{dirname}/{filename}'
    target_path = f'{PROJ_PATH}/basemodel.yaml'
    shutil.copy(source_path, target_path)
    logger.info(f'BMS: Selected model is [{model.upper()}{size.upper()}]\n')

    # construct nodes and edges
    viewer = BasemodelViewer(userid, project_id)
    viewer.parse_yaml(target_path, data)
    viewer.update()

    # save internally
    info = Info.objects.get(userid=userid, project_id=project_id)
    info.model_type = model
    info.model_size = size
    info.status = "running"
    info.progress = "bms"
    info.model_viz = "ready"
    info.save()
    # print('basemodel selected')
    # info.print()

    # for updating status (P.M)
    model_p = model.upper()
    size_p = size.replace('-', '').replace('_', '').upper()
    basemodel = {
        "model_name": model_p,
        "model_size": size_p,
    }

    return target_path, basemodel


def backup_previous_work(model):
    m = Path(model)
    cur_dir = m.parent
    model_name = m.stem

    all_files = [x for x in glob.glob(str(cur_dir/'*')) if os.path.isfile(x)]
    bestmodel_files = [f for f in all_files if model_name in f] # i.e, bestmodel.pt, bestmodel.onnx, ...
    meta_file = cur_dir / 'neural_net_info.yaml'
    bestmodel_files.append(meta_file)

    bak_dir = increment_path(cur_dir / 'bak', exist_ok=False) # i.e bak, bak2, bak3, ...
    Path(bak_dir).mkdir(parents=True, exist_ok=True)
    for f in bestmodel_files:
        shutil.move(str(cur_dir/f), bak_dir)
    return bak_dir


def run_autonn(userid, project_id, resume=False, viz2code=False, nas=False, hpo=False):
    # Set Logging --------------------------------------------------------------
    set_logging(int(os.environ['RANK']) if 'RANK' in os.environ else -1)

    # Load settings ------------------------------------------------------------
    proj_info, opt, hyp, basemodel, data = get_user_requirements(userid, project_id, resume)

    target = proj_info['target_info'] # PC, Galaxy_S22, etc.
    target_acc = proj_info['acc'] # cuda, opencl, cpu
    target_engine = proj_info['engine'].replace('-', '').replace('_', '').lower() # tensorrt, pytorch, tvm, etc.
    task = proj_info['task_type'].lower() # detection, classification, chat
    if task == "chat":
        return

    opt.lt = proj_info.get('learning_type', 'normal').lower() # normal / incremental / transfer / hpo
    logger.info(f"Project Info: Learning Type: {opt.lt}")
    if opt.lt == 'hpo' or opt.evolve:
        hpo = True
    elif opt.lt == 'transfer':
        opt.weights = proj_info.get('weights')

    proj_info['userid'] = userid
    proj_info['project_id'] = project_id
    proj_info['viz'] = viz2code
    proj_info['hpo'] = hpo
    if target == 'Galaxy_S22' and task == 'detection':
        nas = True
    proj_info['nas'] = nas

    # Clear CUDA memory --------------------------------------------------------
    torch.cuda.empty_cache()
    gc.collect()

    if basemodel['imgsz']==1280:
        opt.loss_name = 'AuxOTA'


    # check options ------------------------------------------------------------
    # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    opt.name = 'evolve' if opt.evolve else opt.name
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

    if opt.resume: # resume an interrupted run ---------------------------------
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        # assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        if os.path.isfile(ckpt):
            # apriori = opt.global_rank, opt.local_rank
            # with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            #     opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))  # replace
            # opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = '', ckpt, True, opt.total_batch_size, *apriori  # reinstate
            opt.weights = ckpt
            logger.info('Resuming training from %s' % ckpt)


    # Pre-train --------------------------------------------------------------------
    tb_writer = None  # init loggers
    # if opt.global_rank in [-1, 0]:
    #     prefix = colorstr('tensorboard: ')
    #     try:
    #         tb_writer = SummaryWriter(log_dir=str(opt.save_dir))  # Tensorboard
    #         logger.info(f"{prefix}Start with 'tensorboard --logdir {str(opt.save_dir)}',"
    #                     f" view at http://localhost:6006/")
    #     except Exception as e:
    #         logger.warn(f'{prefix}Fail to load tensorbord because {e}')
    #===========================================================================
    results, train_final = train(proj_info, hyp, opt, data, tb_writer)
    #===========================================================================
    # tb_writer.flush()
    best_acc = results[3] if task == 'detection' else results[0]
    logger.info(f'Train: Training complete. Best results: {best_acc:.2f},'
                f' Best model saved as: {train_final}\n')

    # NAS ----------------------------------------------------------------------
    if nas:
        #=======================================================================
        train_final, yaml_file = search(proj_info, hyp, opt, data, train_final)
        #=======================================================================

        opt.resume = True
        opt.weights = str(train_final)
        # results, train_final = train(proj_info, hyp, opt, data, tb_writer=None)


    # HPO ----------------------------------------------------------------------
    if hpo:
        # opt.resume = False
        opt.weights = str(train_final)
        #=======================================================================
        hyp, yaml_file, txt_file = evolve(proj_info, hyp, opt, data)
        #=======================================================================

        # plot results
        plot_evolution(yaml_file, txt_file)
        logger.info(f'HPO: Hyperparameter optimization complete. '
                    f'Best results saved as: {yaml_file}\n')

    # Train --------------------------------------------------------------------
    # re-train a model with the best architecture & hyperparameters
    # if nas or hpo:
    #     #=======================================================================
    #     results, train_final = train(proj_info, hyp, opt, data, tb_writer)
    #     #=======================================================================
    #     best_acc = results[3] if task == 'detection' else results[0]
    #     logger.info(f'\nTrain: Final training complete. Best results: {best_acc:.2f},'
    #                 f' Best model saved as: {train_final}\n')

    # Incremental Learning -----------------------------------------------------
    if opt.lt == 'incremental':
        if opt.best_acc > best_acc:
            logger.info(f"\nIncremental: Training complete but got nothing better")
            return
        else:
            logger.info(f"\nIncremental: Training complete and got a better model")


    # Model Export -------------------------------------------------------------
    '''
    cloud           : pytorch (torchscript)
    k8s             : pytorch (torchscript)
    k8sjetsonnano   : tensorrt
    pcweb           : pytorch (torchscript)
    pc              : pytorch (torchscript)
    jetsonagxorin   : tensorrt
    jetsonagxxavier : tensorrt
    jetsonnano      : tensorrt
    galaxys22       : tflite (-> opencl)
    odroidn2        : onnx ; (tvm -> acl -> opencl)
    '''
    # Strip optimizers ---------------------------------------------------------
    # [tenace's note] what strip_optimizer does is ...
    # 1. load cpu model
    # 2. get attribute 'ema' if it exists
    # 3. replace attribute 'model' with 'ema'
    # 4. reset attributes 'optimizer', 'training_results', 'ema', and 'updates' to None
    # 5. reset attribute 'epoch' to -1
    # 6. convert model to FP16 precision (not anymore by tenace)
    # 7. set [model].[parameters].requires_grad = False
    # 8. save model to original file path
    if not train_final.exists():
        logger.warn("\nModel Exporter: Training complete but no trained weights")
        return

    stripped_train_final = COMMON_ROOT / userid / project_id / 'bestmodel.pt'
    shutil.copyfile(str(train_final), str(stripped_train_final))
    strip_optimizer(stripped_train_final)  # strip optimizers
    train_final = stripped_train_final

    # save externally ----------------------------------------------------------
    opt.best_acc = float(best_acc)  # numpy.float64 to float
    # print(type(opt.best_acc))
    opt.weights = str(train_final)
    with open(Path(opt.save_dir) / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)
    # print('='*30)
    # for k,v in vars(opt).items():
    #     print(f"{k}: {v}")
    # print('='*30)
    # save internally ----------------------------------------------------------
    info = Info.objects.get(userid=userid, project_id=project_id)
    info.status = "running"
    info.progress = "model export"
    info.best_acc = opt.best_acc
    info.best_net = str(train_final)
    info.save()

    logger.info(f'Model Exporter: Converting models for target [{target}({target_acc}):{target_engine}]...')
    logger.info('='*100)

    # export weights -----------------------------------------------------------
    target_engine = proj_info['engine']
    channel = data.get('ch')
    convert = ['torchscript', 'onnx']
    if target_engine == 'tensorrt':
        convert.append('engine')
    if task == 'detection':
        convert.append('onnx_end2end')
    export_weight(train_final, target_acc, convert, task=task, ch=channel, imgsz=opt.img_size)

    # export meta file ---------------------------------------------------------
    src_nninfo_path = CFG_PATH / 'neural_net_info.yaml'
    dst_nninfo_path = COMMON_ROOT / userid / project_id / 'neural_net_info.yaml'
    export_config(src_nninfo_path, dst_nninfo_path, data, basemodel, target_acc, target_engine, task=task)

    # print model export summary -----------------------------------------------
    logger.info(f'\nModel Exporer: Export complete')
    mb = os.path.getsize(train_final) / 1E6  # filesize
    logger.info(f'Source Model = {train_final}({mb:.1f} MB), {results[3]} mAP')
    logger.info('='*100)
    for model_type in convert:
        if model_type == 'onnx_end2end':
            dir = Path(train_final).parent / Path(train_final).stem
            suffix = '-end2end.onnx'
            exported_bestmodel_path = f"{str(dir)}{suffix}"
        else:
            suffix = f'.{model_type}'
            exported_bestmodel_path = Path(train_final).with_suffix(suffix)
            model_type = 'tensor-rt' if model_type == 'engine' else model_type

        if not os.path.isfile(exported_bestmodel_path):
            exported_bestmodel_path = 'not exist'
            mb = 0.0
        else:
            mb = os.path.getsize(exported_bestmodel_path) / 1E6  # filesize
        logger.info(f'{model_type.upper():>20s} Model: {exported_bestmodel_path}({mb:.1f} MB)')
    logger.info('='*100)
    logger.info(f'Meta data = {dst_nninfo_path}\n')


    # Inference ----------------------------------------------------------------
    # version 1. call a function from main codes
    # from .detect import detect
    # weights = dst_bestmodel_path
    # cfg = COMMON_ROOT / userid / project_id / 'basemodel.yaml'
    # detect(weights, cfg, save_img=False)
    # INF_DIR = CORE_DIR / 'tango' / 'inference'

    # version 2. run an inference app directly
    # from tango.inference.detection_app import run
    # weights = './yolov9-m.torchscript'
    # source = str(INF_DIR / 'horses.jpg')
    # run(weights, source, view_img=False, save_img=True, save_txt=True, save_dir=str(INF_DIR))

    # return train_final  # bestmodel.pt, bestmodel.torchscript, bestmodel.onnx
    return

