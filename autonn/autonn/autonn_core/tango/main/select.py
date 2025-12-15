"""
AutoNN Core Module - Automated Neural Network training and optimization.

This moduls provides functionality for automated model selection, trining,
optimization, and export for various machine learning tasks.
"""

import os
import glob
import shutil
import tempfile
import gc
import argparse
import logging
import signal
import subprocess
import threading
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
from . import status_update #, Info
from .train import train
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
from tango.utils.django_utils import safe_update_info, safe_get_info_values, safe_get_info_field
from tango.utils.torch_utils import select_device_and_info

# Constants
COMMON_ROOT = Path("/shared/common")
DATASET_ROOT = Path("/shared/datasets")
MODEL_ROOT = Path("/shared/models")
CORE_DIR = Path(__file__).resolve().parent.parent.parent # /source/autonn_core
CFG_PATH = CORE_DIR / 'tango' / 'common' / 'cfg'
REPO_ROOT = CORE_DIR.parent
PROJECT_DATA_ROOT = REPO_ROOT / 'project_manager' / 'data' / 'datasets'
os.environ['OLLAMA_MODELS'] = str(MODEL_ROOT)

# Tasks to model mapping
TASK_TO_MODEL_TABLE = {
    "detection7": "yolov7",
    "detection": "yolov9",
    "classification": "resnet",
    "classification-c": "resnetc",
    "classification-v": "vgg",
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
    },
    "vgg": {
        "XXL": "19",
        "XL" : "19",
        "L"  : "19",
        "M"  : "16",
        "MS" : "16",
        "S"  : "11",
        "T"  : "11",
    }
}

logger = logging.getLogger(__name__)

# --- get config: hyperparameters, options(args)
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

    status_update(userid, projid, update_id="project_info", update_content=proj_info_dict)

    target = proj_info_dict['target_info'].replace('-', '').replace('_', '').lower()
    device = proj_info_dict['acc']
    task = proj_info_dict['task_type'].lower()

    # ORM update (1/4)
    safe_update_info(userid, projid,
            targeti=target, device=device, dataset=proj_info_dict['dataset'], task=task,
            status="running", progress="setting", model_viz="not ready",
    )

    # Handle learning type
    lt = proj_info_dict['learning_type'].lower()
    skip_bms = False
   
    best_ckpt = safe_get_info_field(userid, projid, "best_net")
    last_ckpt = get_latest_run(COMMON_ROOT / userid / projid)
    last_epoch = safe_get_info_field(userid, projid, "epoch")
    if isinstance(best_ckpt, str) and os.path.isfile(best_ckpt):
        if lt in ['incremental', 'transfer', 'finetune', 'hpo']:
            logger.info(
                f'{colorstr("Project Info: ")}Pretrained model {best_ckpt} exists.\n'
                f'              BMS will be skipped.'
            )
            skip_bms = True
        elif lt == 'normal':
            if not resume:
                bak_dir = backup_previous_work(best_ckpt)
                logger.info(
                    f'{colorstr("Project Info: ")}Pretrained model {best_ckpt}\n'
                    f'              moved to {bak_dir}/...'
                )
                # ORM update (2/4)
                safe_update_info(userid, projid, best_net='')
            else:
                logger.info(
                    f'{colorstr("Project Info: ")}Last epoch = {last_epoch}\n'
                    f'              Last checkpoint = {last_ckpt}'
                )
                skip_bms = True
        else:
            logger.warning(
                f'{colorstr("Project Info: ")}Pretrained model {best_ckpt} exists.\n'
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
        model_value = safe_get_info_values(userid, projid, fields=["model_type", "model_size"])
        basemodel = {
            "model_name": model_value['model_type'], #info.model_type,
            "model_size": model_value['model_size'], #info.model_size,
        }
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
        basemodel_dict['hyp'] = 'p5'
    elif task == 'classification':
        basemodel_dict['hyp'] = 'cls'
    elif task == 'segmentation':
        basemodel_dict['hyp'] = 'seg'
    else:
        logger.warning(f'{colorstr("hyp: ")} unsupprted task: {task}')
        basemodel_dict['hyp'] = 'p5'

    proj_info_dict['nas'] = True if basemodel['model_size'] == '-supernet' else False

    # Load hyperparameters
    hyp_yaml_path = PROJ_PATH / "hyp.yaml"

    seleted_hyp_src = None
    hyp_yaml_path = PROJ_PATH / f"hyp.scratch.{basemodel_dict['hyp']}.yaml"
    
    if skip_bms:
        seleted_hyp_src = PROJ_PATH / 'autonn' / 'hyp.yaml'
    else:
        seleted_hyp_src = PROJ_PATH / f"hyp.scratch.{basemodel_dict['hyp']}.yaml"

    if seleted_hyp_src.is_file():
        _atomic_copy(seleted_hyp_src, hyp_yaml_path)

    logger.info(f'{colorstr("Project Info: ")}hyperparameters from {hyp_yaml_path}')

    with open(hyp_yaml_path) as f:
        hyp_dict = yaml.safe_load(f)
    
    # Load arguments
    opt_yaml_path = PROJ_PATH / 'opt.yaml'

    selected_opt_src = None
    if skip_bms:
        selected_opt_src = PROJ_PATH / 'autonn' / 'opt.yaml'
    else:
        selected_opt_src = PROJ_PATH / f'args-{task}.yaml'

    if selected_opt_src.is_file():
        _atomic_copy(selected_opt_src, opt_yaml_path)

    logger.info(f'{colorstr("Project Info: ")}arguments from {opt_yaml_path}')    

    with open(opt_yaml_path, encoding='utf-8') as f:
        opt = argparse.Namespace(**yaml.safe_load(f))

    # Check for weights and resume state
    weights = vars(opt).get('weights', None)
    if weights:
        logger.info(f'{colorstr("Project Info: ")}transfer learning/fine-tuning from pretrained model[{weights}]')
    
    if resume:
        opt.oom = True
        opt.resume = str(last_ckpt) #True
        opt.last_epoch = last_epoch
        prev_bs_factor = safe_get_info_field(userid, projid, "batch_multiplier")
        opt.bs_factor = prev_bs_factor - 0.1 #info.batch_multiplier - 0.1
        opt.batch_size = -1 # ensure autobatch
        logger.info(f'{colorstr("Project Info: ")}resuming from last ckpt[{opt.resume}] @ epoch #{last_epoch:>3}')

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
    (주의) torchrun으로 DDP 실행하면 torchrun이 환경변수로 WORLD_SIZE, RANK, LOCAL_RANK 설정
    지금은 DDP 실행여부를 판단하기 전이므로 환경변수 설정이 안되어 있음
    '''
    # Set distributed training parameters
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

    # Set image size
    if opt.img_size == -1:
        opt.img_size = [basemodel_dict['imgsz'], basemodel_dict['imgsz']]
    else:
        if isinstance(opt.img_size, int):
            opt.img_size = [opt.img_size, opt.img_size]
        elif isinstance(opt.img_size, list):
            pass
        else:
            logger.warning(f"{colorstr('arguments: ')}Unexpected argument type for img_size: {type(opt.img_size)}")
            opt.img_size = [640, 640]
        basemodel_dict['imgsz'] = opt.img_size[0] # 파싱할 때 여기에 기재된 입력 해상도를 사용함
        with open(basemodel_yaml_path, "w") as f:
            yaml.safe_dump(basemodel_dict, f, sort_keys=False)

    # Set paths and configurations
    opt.project = str(PROJ_PATH)
    opt.data = str(dataset_yaml_path)
    opt.cfg = str(basemodel_yaml_path)
    opt.hyp = str(hyp_yaml_path)

    # Update status
    status_update(userid, projid, update_id="hyperparameter", update_content=hyp_dict)
    status_update(userid, projid, update_id="arguments", update_content=vars(opt))
    status_update(userid, projid, update_id="basemodel", update_content=basemodel)

    return proj_info_dict, opt, hyp_dict, basemodel_dict, data_dict

def _atomic_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(dst.parent), delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        shutil.copy2(src, tmp_path)
        os.replace(tmp_path, dst)
    finally:
        if tmp_path.exists():
            try: tmp_path.unlink()
            except Exception:
                pass

def _atomic_move(src: Path, dst: Path):
    """같은 파일시스템 내에서 rename(원자적). src가 없으면 False 반환."""
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dst)
    return True

def _safe_unlink(paths):
    for p in paths:
        try:
            if p.is_file():
                p.unlink()
        except Exception as e:
            logger.warning(f'{colorstr("Project Info: ")}skip deleting {p}: {e}')


# --- get datasets
def get_dataset_info(dataset_name, task):
    """
    Get dataset information from dataset.yaml file.
    
    Args:
        dataset_name: Name of the dataset
        task: Type of task (detection, classification, etc.)
        
    Returns:
        tuple
            - dict: Dictionary containing dataset information
            - Path: Path object pointing to dataset.yaml file
    """
    if os.path.isdir(str(DATASET_ROOT / dataset_name)):
        dataset_yaml_path = DATASET_ROOT / dataset_name / "dataset.yaml"
    else:
        logger.warning(f"There is no {DATASET_ROOT}/{dataset_name}.")
        if task == 'detection':
            logger.info("Instead embedded COCO128 dataset will be used.")
            dataset_name = 'coco128'
            dataset_yaml_path = CORE_DIR / 'datasets' / 'coco128' / 'dataset.yaml'
    
    if not os.path.isfile(dataset_yaml_path):
        logger.warning(f"Not found dataset.yaml")
        if task == "classficiation":
            logger.info(f"Try to make dataset.yaml from {DATASET_ROOT}/{dataset_name}...")
            dataset_dict = create_classification_dataset_yaml(dataset_name, dataset_yaml_path)
    
    with open(dataset_yaml_path) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)

    _ensure_dataset_links(dataset_name)

    return data_dict, dataset_yaml_path

def _ensure_dataset_links(dataset_name: str) -> None:
    embedded_root = CORE_DIR / 'datasets' / dataset_name
    target_root = DATASET_ROOT / dataset_name
    if embedded_root.exists():
        _symlink_directory(target_root, embedded_root)

    # Provide COCO annotation fallback for evaluation metrics
    if 'coco' in dataset_name.lower():
        _ensure_coco_annotations()

def _symlink_directory(target: Path, source: Path) -> None:
    try:
        if target.exists() or target.is_symlink():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(source, target)
        logger.info(f"Linked embedded dataset {source} -> {target}")
    except OSError as err:
        logger.debug(f"Failed to create symlink {target} -> {source}: {err}")

def _ensure_coco_annotations() -> None:
    src_ann = PROJECT_DATA_ROOT / 'MS-COCO' / 'annotations' / 'instances_val2017.json'
    if not src_ann.exists():
        return

    for dataset_name in ('coco', 'coco128'):
        dest = DATASET_ROOT / dataset_name / 'annotations' / src_ann.name
        try:
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(src_ann, dest)
            logger.info(f"Linked COCO annotation {src_ann} -> {dest}")
        except OSError as err:
            logger.debug(f"Failed to link annotation {src_ann} -> {dest}: {err}")

def create_classification_dataset_yaml(dataset_name, yaml_path):
    """
    Create a dataset.yaml file for classification tasks.
    
    Args:
        dataset_name: Name of the dataset
        yaml_path: Path to save the dataset.yaml file
        
    Returns:
        Dictionary containing dataset information
    """
    dataset_dict = {
        'train': f'{str(DATASET_ROOT / dataset_name / "train")}',
        'val': f'{str(DATASET_ROOT / dataset_name / "val")}',
        'nc': 2,
        'ch': 1,
        'names': []
    }
    
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_dict, f)
    
    return dataset_dict


# --- select base model
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
    else:
        logger.warning(f"\nBMS: Not supported task: {task}")

    # Special case for Galaxy S22 + Detection / Rasberry Pi5 + Classification
    if target == 'galaxys22' and task == 'detection':
        # task_ = 'detection7'
        model_size = 'NAS'
    elif target == 'rasberrypi5' and task == 'classification':
        task_ = 'classification-v'
        model_size = 'M' # = vgg16

    # Look up appropriate model
    if not manual_select:
        model = TASK_TO_MODEL_TABLE[task_]
        size = MODEL_TO_SIZE_TABLE[model][model_size]
    else:
        model = manual_select['model']
        size = manual_select['size']

    # Handle resnet naming convention
    dirname = model if task_ != 'classification-c' else model[:-1] # remove 'c' from 'resnetc'
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
        viewer.update_yolov9m()
    elif model.upper() == 'VGG':
        viewer.update_vgg16()
    else:
        viewer.parse_yaml(target_path, data)
        viewer.update()

    # ORM update (3/4)
    safe_update_info(userid, project_id,
            model_type=model, model_size=size, progress="bms", model_viz="ready")

    # Prepare return values
    model_p = model.upper()
    size_p = size.replace('-', '').replace('_', '').upper()
    basemodel = {
        "model_name": model_p,
        "model_size": size_p,
    }

    return target_path, basemodel


# --- main entry point
def run_autonn(userid, project_id, resume=False, viz2code=False, nas=False, hpo=False, stop_event=None):
    """
    Main function to run the AutoNN workflow.
    
    Args:
        userid: User identifier
        project_id: Project identifier
        resume: Whether to resume previous training
        viz2code: Whether to visualize to code
        nas: Whether to run neural architecture search
        hpo: Whether to run hyperparameter optimization
    """
    stop_flag_path = None

    def stop_requested():
        file_flag = bool(stop_flag_path and Path(stop_flag_path).exists())
        return bool(stop_event and stop_event.is_set()) or file_flag

    def start_stop_flag_watcher(path, proc=None):
        if not stop_event or not path:
            return

        def _watch():
            stop_event.wait()
            try:
                Path(path).write_text("stop", encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to write stop flag: %s", exc)
            if proc:
                try:
                    if hasattr(os, "killpg"):
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    else:
                        proc.terminate()
                    proc.wait(timeout=30)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to terminate ddp subprocess: %s", exc)
                    try:
                        proc.kill()
                    except Exception:
                        pass

        t = threading.Thread(target=_watch, daemon=True)
        t.start()

    # Set logging
    set_logging(int(os.environ['RANK']) if 'RANK' in os.environ else -1)

    # Load settings
    proj_info, opt, hyp, basemodel, data = get_user_requirements(userid, project_id, resume)

    # Handle missing project information
    if proj_info is None:
        return

    if stop_requested():
        safe_update_info(userid, project_id, status="stopped", progress="stopped")
        return True

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
    if basemodel['imgsz']==1280 and ("YOLOv7" in basemodel['name'].upper()):
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
            logger.info(f'{colorstr("Project Info: ")}Resuming training {ckpt} from epoch {opt.last_epoch}')

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

    # CUDA device
    # Respect pre-set CUDA_VISIBLE_DEVICES (e.g., "3,2,1,0")
    # so logical cuda:0 maps to intended GPU
    env_dev = os.environ.get('CUDA_VISIBLE_DEVICES')
    user_opt_device = getattr(opt, "device", None)
    if user_opt_device is not None:
        user_opt_device = str(user_opt_device).strip()
        if user_opt_device == "":
            user_opt_device = None
    gpu_num = 0
    if env_dev:
        opt.device = str(env_dev).strip()
    elif user_opt_device:
        opt.device = user_opt_device
        parsed_devices = [
            d.strip() for d in user_opt_device.split(",")
            if d.strip().isdigit()
        ]
        gpu_num = len(parsed_devices)
    else:
        gpu_num = torch.cuda.device_count()
        opt.device = ",".join(str(i) for i in range(gpu_num))
    logger.info(f'{colorstr("DEVICE:")} {opt.device}')
    device, device_info = select_device_and_info(opt.device)

    system = {}
    system['torch'] = torch.__version__
    system['cuda'] = torch.version.cuda
    system['cudnn'] = torch.backends.cudnn.version() / 1000.0
    for i, d in enumerate(device_info):
        system_info = {}
        system_info['devices'] = d[0]
        system_info['gpu_model'] =  d[1]
        if d[0] == 'CPU':
            d[2] = "0.0"
        system_info['memory'] = d[2]
        system[f'{i}'] = system_info
    status_update(userid, project_id, update_id="system", update_content=system)

    # Train model
    already_ddp = "WORLD_SIZE" in os.environ
    ddp =  (torch.cuda.is_available() and gpu_num > 1) and not already_ddp
    nproc = gpu_num if ddp else 1
    env = os.environ.copy()
    if ddp:
        env["AUTONN_ALLOW_ORM"] = "1"
        import tempfile, json
        cfg = {
            "proj_info": proj_info,
            "hyp": hyp,
            "opt": vars(opt) if hasattr(opt, "__dict__") else opt,
            "data_dict": data,
            "device": device,
        }
        
        fd_cfg, cfg_path = tempfile.mkstemp(prefix=f"train_cfg_", suffix=".json")
        os.close(fd_cfg)
        json_cfg = to_jsonable(cfg)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(json_cfg, f, ensure_ascii=False, indent=2)
        
        fd_res, res_path = tempfile.mkstemp(prefix="train_result_", suffix=".json")
        os.close(fd_res)
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump({}, f)
        
        fd_stop, stop_flag_path = tempfile.mkstemp(prefix="autonn_stop_", suffix=".flag")
        os.close(fd_stop)

        env["TRAIN_CONFIG"] = cfg_path
        env["RESULT_PATH"] = res_path
        env["STOP_FLAG_PATH"] = stop_flag_path
        
        MODULE = "tango.main.train"
        cmd = [
            shutil.which("torchrun") or "torchrun",
            f"--nproc_per_node={nproc}",
            "-m", MODULE
        ]
        preexec_fn = os.setsid if hasattr(os, "setsid") else None
        ddp_proc = subprocess.Popen(cmd, env=env, cwd=str(CORE_DIR), preexec_fn=preexec_fn)
        start_stop_flag_watcher(stop_flag_path, ddp_proc)
        ret = ddp_proc.wait()
    
        if stop_requested():
            logger.info("[AutoNN] Stop requested during DDP run; skipping export")
            safe_update_info(userid, project_id, status="stopped", progress="stopped")
            # 결과가 있다면 읽어 두되, 이후 단계는 중단
            if os.path.exists(res_path):
                with open(res_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                results = payload.get("results")
                train_final = payload.get("train_final")
            else:
                results = None
                train_final = None
        else:
            if ret != 0:
                raise subprocess.CalledProcessError(ret, cmd)
            with open(res_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            results = payload.get("results")
            train_final = payload.get("train_final")

        os.remove(cfg_path)
        os.remove(res_path)
        if stop_flag_path and Path(stop_flag_path).exists():
            Path(stop_flag_path).unlink(missing_ok=True)
    else:
        results, train_final = train(
            proj_info, hyp, opt, data, device, tb_writer, stop_event=stop_event, stop_flag_path=stop_flag_path
        )

    if stop_requested() or results is None or train_final is None:
        logger.info("[AutoNN] Stop requested after training; skipping export")
        safe_update_info(userid, project_id, status="stopped", progress="stopped")
        return True

    # Log training results
    best_acc = results[3] if task == 'detection' else results[0]
    logger.info(
        f'{colorstr("Train: ")}Training complete. Best results: {best_acc:.2f},'
        f' Best model saved as: {train_final}\n'
    )

    yaml_file = None

    # Run neural architecture search if enabled
    if nas:
        train_final, yaml_file = search(
            proj_info, hyp, opt, data, device, train_final
        )

        if stop_requested():
            logger.info("[AutoNN] Stop requested after NAS; skipping further steps")
            safe_update_info(userid, project_id, status="stopped", progress="stopped")
            return True
        # opt.resume = True
        opt.weights = str(train_final)

    # Run hyperparameter optimization if enabled
    if hpo:
        # opt.resume = True
        opt.weights = str(train_final)
        hyp, yaml_file, txt_file = evolve(
            proj_info, hyp, opt, data, device, train_final
        )

        if stop_requested():
            logger.info("[AutoNN] Stop requested after HPO; skipping further steps")
            safe_update_info(userid, project_id, status="stopped", progress="stopped")
            return True

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
    # 8. save model to target file path(e.g. best_stripped.pt)
    # --------------------------------------------------------------------------

    # Check if trained model exists
    if not Path(train_final).exists():
        logger.warning(f'\n{colorstr("Model Exporter: ")}Training complete but no trained weights')
        return

    # Strip optimizer, ema, etc.
    stripped_train_final = str(train_final).replace('.pt', '_stripped.pt')
    if not Path(stripped_train_final).exists():
        strip_optimizer(
            f=train_final,
            s=stripped_train_final,
            prefix=colorstr("Model Exporter: ")
        )
    train_final = stripped_train_final

    # Convert Model ------------------------------------------------------------
    # Dual or Triple heads for training -> Single head for inference
    # Reparameterization process
    #     (1) Load model configuration(e.g. yolov9-s-converted.yaml) with single head [model]
    #     (2) Load model weights(e.g. best.pt) with daul/triple heads from training result [ckpt]
    #     (3) Cherry-pick weight params from [ckpt] to [model] using dedeicated indices
    #     (4) Other params in [model] like epoch, training_results, etc set to None
    #     (4) Save [model] (e.g. best_stripped_converted.pt)
    # --------------------------------------------------------------------------

    # Convert YOLOv9 model for inference
    if 'yolov9' in basemodel['name']:
        inf_cfg = CFG_PATH / 'yolov9' / f"{basemodel['name']}-converted.yaml"
        if 'supernet' in basemodel['name']:
            if yaml_file and Path(yaml_file).exists():
                inf_cfg = Path(opt.save_dir) / 'best_search_converted.yaml'
                shutil.copy2(yaml_file, inf_cfg)
                logger.info(f"{colorstr('Model Exporter:')} Using NAS-derived config {inf_cfg}")
            else:
                logger.warning(
                    f"{colorstr('Model Exporter:')}No NAS config found; falling back to default converted template"
                )
        inf_cfg = str(inf_cfg)
        train_final = convert_yolov9(train_final, inf_cfg)

    # Fuse layers for inference efficiency -------------------------------------
    # [tenace's note] we need to save fused model to file as final inference model
    # 1. conv+bn combination to conv with bias
    # 2. implicit+conv to conv with bias
    # --------------------------------------------------------------------------
    final_out = COMMON_ROOT / userid / project_id / 'bestmodel.pt'
    shutil.copyfile(train_final, final_out)
    fuse_layers(final_out, prefix=colorstr("Model Exporter: "))

    # Save training configuration
    opt.best_acc = float(best_acc)  # numpy.float64 to float
    opt.weights = str(final_out)

    with open(Path(opt.save_dir) / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # ORM update (4/4)
    safe_update_info(userid, project_id,
        status="running", progress="model_export",
        best_acc=opt.best_acc, best_net=str(final_out),
    )

    # Export model for target device
    export_model(userid, project_id, final_out, opt, data, basemodel, 
                 target, target_acc, target_engine, task, results)

    return bool(stop_event and stop_event.is_set())

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

def to_jsonable(obj):
    import enum
    from datetime import datetime, date

    # 기본 타입은 그대로
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # torch.device -> 문자열 (예: "cuda:0" 또는 "cpu")
    if isinstance(obj, torch.device):
        return str(obj)  # f"{obj.type}:{obj.index}" 도 가능

    # torch.dtype -> 문자열 (예: "torch.float32")
    if isinstance(obj, torch.dtype):
        return str(obj)

    # torch.Tensor -> 리스트 또는 스칼라
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return obj.tolist()

    # numpy 스칼라/배열
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # 경로, 날짜/시간, Enum
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, enum.Enum):
        return obj.value

    # 시퀀스/매핑
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    # 그 외는 표현 문자열로 대체
    return repr(obj)

# --- export model
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

    if task == 'detection':
        if target_engine == 'tensorrt':
            convert.append('onnx_end2end')
        # if target_engine == 'tflite':
        #     tfmodels = ['pb', 'tflite']
        #     convert.extend(tfmodels)
        #     if target_acc == 'tpu':
        #         convert.append('edgetpu')

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
    if task == 'detection':
        logger.info(f'Source Model = {train_final}({mb:.1f} MB), {results[3]} mAP')
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



# --- task: chat
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
