"""
AutoNN Core Module - Automated Neural Network training and optimization.

This moduls provides functionality for automated model selection, trining,
optimization, and export for various machine learning tasks.
"""

import os
import glob
import shutil
import gc
import argparse
import logging
import subprocess
from pathlib import Path

# Standard library imports
import yaml

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# Local application imports
from . import status_update, Info
from .train import train
from .continual import train_continual
from .search import search
from .evolve import evolve
from .visualize import BasemodelViewer
from .export import export_weight, export_config, convert_yolov9
from tango.utils.general import (
    increment_path,
    get_latest_run,
    check_file,
    set_logging,
    strip_optimizer,
    fuse_layers,
    colorstr
)
from tango.utils.plots import plot_evolution

# Constants
COMMON_ROOT = Path("/shared/common")
DATASET_ROOT = Path("/shared/datasets")
COCO128_SEG_DATASET_ROOT = Path("/shared/datasets/coco128_seg")
MODEL_ROOT = Path("/shared/models")
CORE_DIR = Path(__file__).resolve().parent.parent.parent # /source/autonn_cl_core
CFG_PATH = CORE_DIR / 'tango' / 'common' / 'cfg'
os.environ['OLLAMA_MODELS'] = str(MODEL_ROOT)

# Tasks to model mapping
TASK_TO_MODEL_TABLE = {
    "detection7": "yolov7",
    "detection": "yolov9",
    "classification": "resnet",
    "classification-c": "resnetc",
    "segmentation": "yolov9"
}

# Model size mapping by target
MODEL_TO_SIZE_TABLE = {
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
    Get user requirements (dataset, project, hyperparameters, arguments).
    
    Args:
        userid: User identifier
        projid: Project identifier
        resume: Whether to resume previous training
        
    Returns:
        Tuple containing project info, options, hyperparameters, basemodel, and data dictionaries
    """
    # Read project information
    PROJ_PATH = COMMON_ROOT / userid / projid
    proj_yaml_path = PROJ_PATH / "project_info.yaml"

    with open(proj_yaml_path, "r") as f:
        proj_info_dict = yaml.safe_load(f)

    status_update(
        userid, 
        projid,
        update_id="project_info",
        update_content=proj_info_dict
    )

    target = proj_info_dict['target_info'].replace('-', '').replace('_', '').lower()
    device = proj_info_dict['acc']
    task = proj_info_dict['task_type'].lower()

    # Update database info
    info = Info.objects.get(userid=userid, project_id=projid)
    info.target = target
    info.device = device
    info.dataset = proj_info_dict['dataset']
    info.task = task
    info.status = "running"
    info.progress = "setting"
    info.model_viz = "not ready"
    info.save()

    # Handle learning type
    lt = proj_info_dict['learning_type'].lower()
    skip_bms = False
    
    if os.path.isfile(info.best_net):
        if lt in ['incremental', 'transfer', 'finetune', 'hpo']:
            logger.info(
                f'{colorstr("Project Info: ")}Pretrained model {info.best_net} exists.\n'
                f'              BMS will be skipped.'
            )
            skip_bms = True
        elif lt == 'normal':
            bak_dir = backup_previous_work(info.best_net)
            logger.info(
                f'{colorstr("Project Info: ")}Pretrained model {info.best_net}\n'
                f'              moved to {bak_dir}/...'
            )
            info.best_net = ''
            info.save()
        else:
            logger.warning(
                f'{colorstr("Project Info: ")}Pretrained model {info.best_net} exists.\n'
                f'              But learning type {lt} is unknown, '
                f'so it will be overwritten by the end of training.'
            )

    # Handle chat task
    if task == 'chat':
        return handle_chat_task()

    # Process dataset information
    dataset_on_proj = proj_info_dict["dataset"]
    data_dict, dataset_yaml_path = get_dataset_info(dataset_on_proj, task)
    data_dict['dataset_name'] = dataset_on_proj

    # Select base model
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
        basemodel_yaml_path, basemodel = base_model_select(
            userid,
            projid,
            proj_info_dict,
            data_dict
        )
    
    with open(basemodel_yaml_path, "r") as f:
        basemodel_dict = yaml.load(f, Loader=yaml.SafeLoader)

    # Adjust hyperparameter based on task
    if task == 'detection':
        basemodel_dict['hyp'] = 'p5' if basemodel_dict['hyp'] == 'tiny' else basemodel_dict['hyp']
    elif task == 'segmentation':
        basemodel_dict['hyp'] = 'seg'
    else:  # classification
        basemodel_dict['hyp'] = 'cls'

    proj_info_dict['nas'] = True if basemodel['model_size'] == '-supernet' else False

    # Load hyperparameters
    hyp_yaml_path = PROJ_PATH / f"hyp.scratch.{basemodel_dict['hyp']}.yaml"
    logger.info(f'{colorstr("hyp: ")}hyperparameters from {hyp_yaml_path}')

    with open(hyp_yaml_path) as f:
        hyp_dict = yaml.safe_load(f)

    hyp_dict['anchor_t'] = 5.0 # from yolov9
    
    status_update(
        userid, 
        projid,
        update_id="hyperparameter",
        update_content=hyp_dict
    )

    # Load arguments
    if (basemodel['model_name'] == 'YOLOV7' or target == 'galaxys22') and task == 'detection':
        _task = 'detection7'
    else:
        _task = task
    
    opt_yaml_path = PROJ_PATH / f'args-{_task}.yaml'

    if skip_bms:
        opt_yaml_path = str(PROJ_PATH / 'autonn' / 'opt.yaml')
        logger.info(f'{colorstr("Project Info: ")}previous opt.yaml loading..')
        assert os.path.isfile(opt_yaml_path)

    with open(opt_yaml_path, encoding='utf-8') as f:
        opt = argparse.Namespace(**yaml.safe_load(f))

    # Check for weights and resume state
    weights = vars(opt).get('weights', None)
    if weights:
        logger.info(f'{colorstr("Project Info: ")}pretrained model = {weights}')

    logger.info(f'{colorstr("Project Info: ")}CUDA OOM: resume? {resume}')
    if resume:
        opt.resume = True
        opt.bs_factor = info.batch_multiplier - 0.1

    # Check for incremental learning
    if lt == 'incremental':
        best_acc = vars(opt).get('best_acc', None)
        logger.info(f'{colorstr("Project Info: ")}Incremental: best accuracy? {best_acc}')
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
    # Set distributed training parameters
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

    # Set paths and configurations
    opt.project = str(PROJ_PATH)
    opt.data = str(dataset_yaml_path)
    opt.cfg = str(basemodel_yaml_path)
    opt.hyp = str(hyp_yaml_path)
    opt.img_size = [basemodel_dict['imgsz'], basemodel_dict['imgsz']]

    # Update status
    opt_content = vars(opt)
    status_update(userid, projid, update_id="arguments", update_content=opt_content)
    status_update(userid, projid, update_id="basemodel", update_content=basemodel)

    return proj_info_dict, opt, hyp_dict, basemodel_dict, data_dict


def handle_chat_task():
    """
    Handle the chat task by running Ollama server and Streamlit browser.
    
    Returns:
        Tuple containing project info and None values for remaining requirements
    """
    logger.info(f'{colorstr("Project Info: ")}Run LLM model loader')
    logger.info(f'{colorstr("Project Info: ")}http://localhost:11434 for Ollama server')
    
    cmd2 = ["ollama", "serve"]
    p2 = subprocess.Popen(
        cmd2,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=CORE_DIR,
        text=True,
        encoding='utf-8'
    )
    
    logger.info(f'{colorstr("Project Info: ")}http://localhost:8101 to see Tango+Chat browser')
    run_chat_browser = str(CORE_DIR / 'tangochat' / 'browser.py')
    cmd = [
        "streamlit", "run", run_chat_browser, 
        "--server.port", "8101", 
        "--browser.gatherUsageStats", "false"
    ]
    
    p = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        cwd=CORE_DIR,
        text=True,
        encoding='utf-8'
    )

    while p.poll() is None:
        out = p.stdout.readline()
        out = out.split('\n')[0]
        out2 = p2.stdout.readline()
        out2 = out2.split('\n')[0]
        
        if len(out) > 0:
            if ("streamlit" not in out) and ("Network URL" not in out):
                logger.info(f"Chat: {out}")
        
        if len(out2) > 0:
            logger.info(f"Chat: Ollama: {out2}")
        
        if out == 'Completed' or out2 == "Completed":
            p2.terminate()
            p.terminate()
            break

    return None, None, None, None, None


def get_dataset_info(dataset_name, task):
    """
    Get dataset information from dataset.yaml file.
    """
    if task == 'segmentation':
        root_path = COCO128_SEG_DATASET_ROOT
        if not root_path.exists():
            raise FileNotFoundError('Segmentation dataset root not found: {}'.format(root_path))
        dataset_yaml_path = root_path / 'dataset.yaml'
        data_dict = create_segmentation_dataset_yaml(root_path, dataset_yaml_path)
        return data_dict, dataset_yaml_path

    dataset_yaml_path = None
    data_dict = None

    if os.path.isdir(str(DATASET_ROOT / dataset_name)):
        dataset_yaml_path = DATASET_ROOT / dataset_name / 'dataset.yaml'
    else:
        logger.warning('There is no {}/{}.'.format(DATASET_ROOT, dataset_name))
        if task == 'detection':
            logger.info('Instead embedded COCO128 dataset will be used.')
            dataset_name = 'coco128'
            dataset_yaml_path = CORE_DIR / 'datasets' / 'coco128' / 'dataset.yaml'

    if not dataset_yaml_path or not os.path.isfile(dataset_yaml_path):
        logger.warning('Not found dataset.yaml at {}'.format(dataset_yaml_path))
        if task == 'classficiation':
            logger.info('Try to make dataset.yaml from {}/{}...'.format(DATASET_ROOT, dataset_name))
            data_dict = create_classification_dataset_yaml(dataset_name, dataset_yaml_path)

    if data_dict is None:
        with open(dataset_yaml_path) as f:
            data_dict = yaml.load(f, Loader=yaml.SafeLoader)

    if 'dataset_name' not in data_dict:
        data_dict['dataset_name'] = dataset_name

    if task == 'segmentation' and data_dict.get('mask_format') is None:
        logger.info("Setting default mask_format='poly' for segmentation dataset")
        data_dict['mask_format'] = 'poly'

    return data_dict, dataset_yaml_path

def create_classification_dataset_yaml(dataset_name, yaml_path):
    """
    Create a dataset.yaml file for classification tasks.
    """
    dataset_dict = {
        'train': str(DATASET_ROOT / dataset_name / 'train'),
        'val': str(DATASET_ROOT / dataset_name / 'val'),
        'nc': 2,
        'ch': 1,
        'names': [],
        'dataset_name': dataset_name
    }

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_dict, f, sort_keys=False)

    return dataset_dict


def create_segmentation_dataset_yaml(root_path, yaml_path):
    """
    Create a dataset.yaml file for segmentation tasks when none is provided.
    """
    existing = {}
    if yaml_path.exists():
        try:
            with open(yaml_path, 'r') as f:
                existing = yaml.safe_load(f) or {}
        except Exception as err:
            logger.warning(f'Failed to read existing dataset yaml {yaml_path}: {err}')
            existing = {}

    train_dir = root_path / 'images' / 'train'
    val_dir = root_path / 'images' / 'val'
    label_dirs = [root_path / 'labels' / 'train', root_path / 'labels' / 'val']

    class_ids = set()
    for label_dir in label_dirs:
        if not label_dir.exists():
            continue
        for label_file in label_dir.rglob('*.txt'):
            try:
                with open(label_file) as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        class_ids.add(int(float(parts[0])))
            except Exception as err:
                logger.warning(f'Failed to parse {label_file}: {err}')

    nc = max(class_ids) + 1 if class_ids else 1
    names = [f'class_{i}' for i in range(nc)]

    dataset_dict = dict(existing) if isinstance(existing, dict) else {}
    dataset_dict.update({
        'path': str(root_path),
        'train': str(train_dir),
        'val': str(val_dir),
        'mask_format': 'poly',
        'dataset_name': root_path.name,
    })
    prev_nc = dataset_dict.get('nc', 0) or 0
    dataset_dict['nc'] = max(nc, int(prev_nc))
    if not dataset_dict.get('names'):
        dataset_dict['names'] = names

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_dict, f, sort_keys=False)

    return dataset_dict

def base_model_select(userid, project_id, proj_info, data, manual_select=False):
    """
    Select an appropriate base model based on task and target hardware.
    
    Args:
        userid: User identifier
        project_id: Project identifier
        proj_info: Project information dictionary
        data: Dataset information dictionary
        manual_select: Optional manual model selection dictionary
        
    Returns:
        Tuple containing target path and basemodel information
        tuple
            - Path: Path object pointing to basemodel.yaml file
            - dict: Dictionary containing basemodel information
    """
    # Read conditions
    task = proj_info['task_type']
    target = proj_info['target_info'].replace('-', '').replace('_', '').lower()
    target_acc = proj_info['acc']
    target_mem = float(proj_info['memory'])

    # Select model size based on memory contraints
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

    logger.info(f'\n{colorstr("BMS: ")}Based on memory = {target_mem}G, acc = {target_acc}')

    # Determine task variant
    if task == 'classification':
        task_ = 'classification-c' if data['nc'] <= 10 else task
    elif task == 'detection':
        task_ = 'detection7' if data['nc'] <= 50 else task
    elif task == 'segmentation':
        task_ = 'segmentation'
    else:
        logger.warning(f"\nBMS: Not supported task: {task}")

    # Special case for Galaxy S22 with detection task
    if target == 'galaxys22' and task == 'detection':
        task_ = 'detection7'
        model_size = 'NAS'

    # Look up appropriate model
    if not manual_select:
        model = TASK_TO_MODEL_TABLE[task_]
        size = MODEL_TO_SIZE_TABLE[model][model_size]
    else:
        model = manual_select['model']
        size = manual_select['size']

    # Handle resnet naming convention
    dirname = model if task_ != 'classification-c' else model[:-1] # remove 'c' from 'resnetc'
    if task == 'segmentation':
        filename = f'{model}{size}-seg.yaml'
    else:
        filename = f'{model}{size}.yaml'

    # Store basemodel.yaml
    PROJ_PATH = COMMON_ROOT / userid / project_id
    source_path = f'{CFG_PATH}/{dirname}/{filename}'
    target_path = f'{PROJ_PATH}/basemodel.yaml'
    shutil.copy(source_path, target_path)
    logger.info(f'{colorstr("BMS: ")}Selected model is [{model.upper()}{size.upper()}]\n')

    # construct visualization
    viewer = BasemodelViewer(userid, project_id)
    if model.upper() == 'YOLOV9': # and size.upper() == '-M':
        viewer.update_detection()
    else:
        viewer.parse_yaml(target_path, data)
        viewer.update()

    # Save internally
    info = Info.objects.get(userid=userid, project_id=project_id)
    info.model_type = model
    info.model_size = size
    info.status = "running"
    info.progress = "bms"
    info.model_viz = "ready"
    info.save()

    # Prepare return values
    model_p = model.upper()
    size_p = size.replace('-', '').replace('_', '').upper()
    basemodel = {
        "model_name": model_p,
        "model_size": size_p,
    }

    return target_path, basemodel


def backup_previous_work(model):
    """
    Backup previous work to a new directory.
    
    Args:
        model: Path to the model file
        
    Returns:
        Path to the backup directory
    """
    logger.info(f'!!! backup previous works !!!')
    m = Path(model)
    cur_dir = m.parent
    model_name = m.stem

    # Find all related files
    all_files = [x for x in glob.glob(str(cur_dir/'*')) if os.path.isfile(x)]
    bestmodel_files = [f for f in all_files if model_name in f] # i.e, bestmodel.pt, bestmodel.onnx, ...

    # Add meta file if exists
    meta_file = cur_dir / 'neural_net_info.yaml'
    if os.path.isfile(meta_file):
        bestmodel_files.append(meta_file)

    # Add saved model directory if exists
    saved_model_dir = cur_dir / 'bestmodel_saved_model'
    if os.path.isdir(saved_model_dir):
        bestmodel_files.append(saved_model_dir)

    # Create backup directory and move files
    bak_dir = increment_path(cur_dir / 'bak', exist_ok=False) # i.e bak, bak2, bak3, ...
    Path(bak_dir).mkdir(parents=True, exist_ok=True)

    for f in bestmodel_files:
        shutil.move(str(cur_dir/f), bak_dir)

    return bak_dir


def run_autonn_cl(userid, project_id, resume=False, viz2code=False, nas=False, hpo=False):
    """
    Main function to run the AutoNN workflow.
    
    Args:
        userid: User identifier
        project_id: Project identifier
        resume: Whether to resume previous training
        viz2code: Whether to visualize to code
        nas: Whether to run neural architecture search
        hpo: Whether to run hyperparameter optimization
        
    Returns:
        None
    """
    # Set logging
    set_logging(int(os.environ['RANK']) if 'RANK' in os.environ else -1)

    # Load settings
    proj_info, opt, hyp, basemodel, data = get_user_requirements(userid, project_id, resume)

    # Handle missing project information
    if proj_info is None:
        return

    # Extract project information
    target = proj_info['target_info'] # PC, Galaxy_S22, etc.
    target_acc = proj_info['acc'] # cuda, opencl, cpu
    target_engine = proj_info['engine'].replace('-', '').replace('_', '').lower() # tensorrt, pytorch, tvm, etc.
    task = proj_info['task_type'].lower() # detection, classification, chat

    # Handle chat task early return
    if task == "chat":
        return

    # Configure learning process
    opt.lt = proj_info.get('learning_type', 'normal').lower() # normal / incremental / transfer / hpo
    logger.info(f'{colorstr("Project Info: ")}Learning Type: {opt.lt}')

    if opt.lt == 'hpo' or opt.evolve:
        hpo = True
    elif opt.lt == 'transfer':
        opt.weights = proj_info.get('weights')

    # Store additional project information
    proj_info['userid'] = userid
    proj_info['project_id'] = project_id
    proj_info['viz'] = viz2code
    proj_info['hpo'] = hpo

    if target == 'Galaxy_S22' and task == 'detection':
        nas = True

    proj_info['nas'] = nas

    # Clear CUDA memory
    torch.cuda.empty_cache()
    gc.collect()

    # Handle large image size models
    if basemodel['imgsz']==1280:
        opt.loss_name = 'AuxOTA'

    # Check options
    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'

    # Extend image size if necessay
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)

    # Configure output directories
    opt.name = 'evolve' if opt.evolve else opt.name
    opt.save_dir = increment_path(
        Path(opt.project) / opt.name, 
        exist_ok=opt.exist_ok | opt.evolve
    )

    # Handle resume option
    if opt.resume:
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path

        if os.path.isfile(ckpt):
            opt.weights = ckpt
            logger.info(f'{colorstr("Project Info: ")}Resuming training from %s' % ckpt)

    # Initialize tensorboard writer 
    tb_writer = None  # init loggers
    # if opt.global_rank in [-1, 0]:
    #     prefix = colorstr('tensorboard: ')
    #     try:
    #         tb_writer = SummaryWriter(log_dir=str(opt.save_dir))  # Tensorboard
    #         logger.info(f"{prefix}Start with 'tensorboard --logdir {str(opt.save_dir)}',"
    #                     f" view at http://localhost:6006/")
    #     except Exception as e:
    #         logger.warn(f'{prefix}Fail to load tensorbord because {e}')

    # Train model
    results, train_final = train_continual(proj_info, hyp, opt, data, tb_writer)

    # Log training results
    best_acc = results[3] if task in ['detection', 'segmentation'] else results[0]
    logger.info(
        f'{colorstr("Train: ")}Training complete. Best results: {best_acc:.2f},'
        f' Best model saved as: {train_final}\n'
    )

    # Run neural architecture search if enabled
    if nas:
        train_final, yaml_file = search(proj_info, hyp, opt, data, train_final)
        opt.resume = True
        opt.weights = str(train_final)
        # results, train_final = train(proj_info, hyp, opt, data, tb_writer=None)

    # Run hyperparameter optimization if enabled
    if hpo:
        opt.weights = str(train_final)
        hyp, yaml_file, txt_file = evolve(proj_info, hyp, opt, data)

        # Plot evolution results
        plot_evolution(yaml_file, txt_file)
        logger.info(
            f'{colorstr("HPO: ")}Hyperparameter optimization complete. '
            f'Best results saved as: {yaml_file}\n'
        )

    # Train --------------------------------------------------------------------
    # re-train a model with the best architecture & hyperparameters
    # if nas or hpo:
    #     #=======================================================================
    #     results, train_final = train(proj_info, hyp, opt, data, tb_writer)
    #     #=======================================================================
    #     best_acc = results[3] if task == 'detection' else results[0]
    #     logger.info(f'\nTrain: Final training complete. Best results: {best_acc:.2f},'
    #                 f' Best model saved as: {train_final}\n')

    # Handle incremental learning
    if opt.lt == 'incremental':
        if opt.best_acc > best_acc:
            logger.info(f'\n{colorstr("Incremental: ")}Training complete but got nothing better')
            return
        else:
            logger.info(f'\n{colorstr("Incremental: ")}Training complete and got a better model')

    # Convert Model ------------------------------------------------------------
    # Dual or Triple heads for training -> Single head for inference
    # Reparameterization process
    #     (1) Load model configuration(e.g. yolov9-s-converted.yaml) with single head [model]
    #     (2) Load model weights(e.g. best.pt) with daul/triple heads from training result [ckpt]
    #     (3) Cherry-pick weight params from [ckpt] to [model] using dedeicated indices
    #     (4) Other params in [model] like epoch, training_results, etc set to None
    #     (4) Save [model] (e.g. best_converted.pt)
    # --------------------------------------------------------------------------

    # Convert YOLOv9 model for inference
    if 'yolov9' in basemodel['name']:
        inf_cfg = str(CFG_PATH / 'yolov9' / f"{basemodel['name']}-converted.yaml")
        train_final = convert_yolov9(train_final, inf_cfg)

    # Model Export -------------------------------------------------------------
    # cloud           : pytorch (torchscript, onnx)
    # k8s             : pytorch (torchscript, onnx)
    # k8sjetsonnano   : onnx_end2end -> tensorrt at target
    # pcweb           : pytorch (torchscript, onnx)
    # pc              : pytorch (torchscript, onnx)
    # jetsonagxorin   : onnx_end2end -> tensorrt at target
    # jetsonagxxavier : onnx_end2end -> tensorrt at target
    # jetsonnano      : onnx_end2end -> tensorrt at target
    # galaxys22       : tflite
    # raspberrypi     : edge-tpu tflite
    # --------------------------------------------------------------------------

    # Strip optimizer ----------------------------------------------------------
    # [tenace's note] what strip_optimizer does is ...
    # 1. load cpu model
    # 2. get attribute 'ema' if it exists
    # 3. replace attribute 'model' with attribute 'ema'
    # 4. reset attributes 'optimizer', 'training_results', 'ema', and 'updates' to None
    # 5. reset attribute 'epoch' to -1
    # 6. convert model to FP16 precision (not anymore by tenace)
    # 7. set [model].[parameters].requires_grad = False
    # 8. save model to original file path
    # --------------------------------------------------------------------------

    # Check if trained model exists
    if not Path(train_final).exists():
        logger.warning(f'\n{colorstr("Model Exporter: ")}Training complete but no trained weights')
        return

    # Strip optimizer and save model
    stripped_train_final = COMMON_ROOT / userid / project_id / 'bestmodel.pt'
    # shutil.copyfile(str(train_final), str(stripped_train_final))
    strip_optimizer(
        f=train_final,
        s=stripped_train_final,
        prefix=colorstr("Model Exporter: ")
    )  # strip optimizers
    train_final = stripped_train_final

    # Fuse layers for inference efficiency
    # [tenace's note] we need to save fused model to file as final inference model
    # 1. conv+bn combination to conv with bias
    # 2. implicit+conv to conv with bias
    fuse_layers(train_final, prefix=colorstr("Model Exporter: "))

    # Save training configuration
    opt.best_acc = float(best_acc)  # numpy.float64 to float
    opt.weights = str(train_final)

    with open(Path(opt.save_dir) / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Update internal status
    info = Info.objects.get(userid=userid, project_id=project_id)
    info.status = "running"
    info.progress = "model export"
    info.best_acc = opt.best_acc
    info.best_net = str(train_final)
    info.save()

    # Export model for target device
    export_model(userid, project_id, train_final, opt, data, basemodel, 
                 target, target_acc, target_engine, task, results)


def export_model(userid, project_id, train_final, opt, data, basemodel, 
                target, target_acc, target_engine, task, results):
    """
    Export the trained model to various formats for deployment.
    
    Args:
        userid: User identifier
        project_id: Project identifier
        train_final: Path to the trained model
        opt: Augments(options) dictionary
        data: Dataset information dictionary
        basemodel: Base model information dictionary
        target: Target hardware platform
        target_acc: Target accelerator
        target_engine: Target inference engine
        task: Type of task
        results: Training results
        
    Returns:
        None
    """
    logger.info(
        f'{colorstr("Model Exporter: ")}'
        f'Converting models for target '
        f'[{target}({target_acc}):{target_engine}]...')
    logger.info('='*100)

    # Determine export formats
    channel = data.get('ch')
    convert = ['torchscript', 'onnx']

    if task in ['detection', 'segmentation']:
        if target_engine == 'tensorrt':
            convert.append('onnx_end2end')
        if target_engine == 'tflite':
            tfmodels = ['pb', 'tflite']
            convert.extend(tfmodels)
            if target_acc == 'tpu':
                convert.append('edgetpu')

    # Export weights to different formats
    export_weight(
        train_final, 
        target_acc, 
        convert, 
        task=task, 
        ch=channel, 
        imgsz=opt.img_size
    )

    # Export meta configuration
    src_nninfo_path = CFG_PATH / 'neural_net_info.yaml'
    dst_nninfo_path = COMMON_ROOT / userid / project_id / 'neural_net_info.yaml'
    export_config(
        src_nninfo_path, 
        dst_nninfo_path, 
        data, 
        basemodel, 
        target_acc, 
        target_engine, 
        task=task
    )

    # Print model export summary
    print_export_summary(train_final, results, convert, task, dst_nninfo_path)


def print_export_summary(train_final, results, convert, task, dst_nninfo_path):
    """
    Print a summary of exported models.
    
    Args:
        train_final: Path to the trained model
        results: Training results
        convert: List of export formats
        task: Type of task
        dst_nninfo_path: Path to neural_net_info.yaml file
        
    Returns:
        None
    """
    # Print source model information
    mb = os.path.getsize(train_final) / 1E6  # filesize
    if task in ['detection', 'segmentation']:
        metric_value = results[3] if isinstance(results, (list, tuple)) and len(results) > 3 else None
        metric_label = 'IoU' if task == 'segmentation' else 'mAP'
        if metric_value is not None:
            logger.info(f'Source Model = {train_final}({mb:.1f} MB), {metric_value} {metric_label}')
        else:
            logger.info(f'Source Model = {train_final}({mb:.1f} MB)')
    elif task == 'classification':
        logger.info(f'Source Model = {train_final}({mb:.1f} MB), val-accuracy = {results[0]}')

    logger.info('='*100)

    # Print information for each exported model
    for model_type in convert:
        exported_bestmodel_path, model_size = get_exported_model_info(train_final, model_type)
        model_type = get_display_model_type(model_type)
        logger.info(f'{model_type.upper():>20s} Model: {exported_bestmodel_path}({model_size:.1f} MB)')

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

def get_exported_model_info(train_final, model_type):
    """
    Get information about an exported model.
    
    Args:
        train_final: Path to the trained model
        model_type: Type of export format
        
    Returns:
        Tuple containing path and size of the exported model
    """
    if model_type == 'onnx_end2end':
        dir_path = Path(train_final).parent / Path(train_final).stem
        suffix = '-end2end.onnx'
        exported_path = f"{str(dir_path)}{suffix}"
    elif model_type == 'edgetpu':
        dir_path = Path(train_final).parent / Path(train_final).stem
        suffix = f'_edgetpu.tflite'
        exported_path = f"{str(dir_path)}{suffix}"
    else:
        suffix = f'.{model_type}'
        exported_path = Path(train_final).with_suffix(suffix)

    if not os.path.isfile(exported_path):
        exported_path = 'not exist'
        mb = 0.0
    else:
        mb = os.path.getsize(exported_path) / 1E6  # filesize

    return exported_path, mb


def get_display_model_type(model_type):
    """
    Get a model format about an exported model for display.
    
    Args:
        model_type: Type of export format
        
    Returns:
        String for displaying model format
    """
    if model_type == 'edgetpu':
        return 'edge-tpu'
    elif model_type == 'engine':
        return 'tensor-rt'
    else:
        return model_type
