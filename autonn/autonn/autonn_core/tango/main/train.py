import logging
import math
import os
import gc
import random
import time
import datetime
import shutil
from copy import deepcopy
from pathlib import Path
from threading import Thread
import json

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
# import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from . import status_update, Info
from . import test  # import test.py to get mAP or val_accuracy after each epoch
from tango.common.models.experimental import attempt_load
from tango.common.models.yolo               import Model, DetectionModel
from tango.common.models.resnet_cifar10     import ClassifyModel
from tango.common.models.supernet_yolov7    import NASModel
# from tango.common.models import *
from tango.utils.autoanchor import check_anchors
from tango.utils.autobatch import get_batch_size_for_gpu
from tango.utils.datasets import (  create_dataloader,
                                    create_dataloader_v9,
                                    AlbumentationDatasetImageFolder
                                 )
from tango.utils.general import (   DEBUG,
                                    labels_to_class_weights,
                                    labels_to_class_weights_v9,
                                    labels_to_image_weights,
                                    init_seeds,
                                    init_seeds_v9,
                                    fitness,
                                    strip_optimizer,
                                    check_dataset,
                                    check_img_size,
                                    check_amp,
                                    one_cycle,
                                    colorstr,
                                    TQDM_BAR_FORMAT,
                                    TqdmLogger,
                                    save_csv,
                                    save_npy,
                                )
from tango.utils.google_utils import attempt_download
from tango.utils.loss import    (   ComputeLoss,
                                    ComputeLossOTA,
                                    ComputeLossAuxOTA,
                                    ComputeLossTAL,
                                    ComputeLoss_v9,
                                    FocalLossCE
                                )
from tango.utils.plots import   (   plot_images,
                                    plot_labels,
                                    plot_results,
                                    plot_cls_results,
                                    plot_lr_scheduler
                                )
from tango.utils.torch_utils import (   ModelEMA,
                                        EarlyStopping,
                                        select_device_and_info,
                                        intersect_dicts,
                                        torch_distributed_zero_first,
                                        is_parallel,
                                        de_parallel,
                                        time_synchronized
                                    )
# from tango.utils.wandb_logging.wandb_utils import WandbLogger


logger = logging.getLogger(__name__)


def get_optimizer(model, is_adam=False, lr=0.001, momentum=0.9, decay=1e-5):
    # YOLOv5 3-param group optimizer: 0) weights with decay, 1) weights no decay, 2) biases no decay
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()

    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
            g[2].append(v.bias)
        if isinstance(v, bn):  # weight (no decay)
            g[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g[0].append(v.weight)

        if hasattr(v, 'im'):
            if hasattr(v.im, 'implicit'):
                g[1].append(v.im.implicit)
            else:
                for iv in v.im:
                    g[1].append(iv.implicit)
        if hasattr(v, 'ia'):
            if hasattr(v.ia, 'implicit'):
                g[1].append(v.ia.implicit)
            else:
                for iv in v.ia:
                    g[1].append(iv.implicit)
        if hasattr(v, 'im2'):
            if hasattr(v.im2, 'implicit'):
                g[1].append(v.im2.implicit)
            else:
                for iv in v.im2:
                    g[1].append(iv.implicit)
        if hasattr(v, 'ia2'):
            if hasattr(v.ia2, 'implicit'):
                g[1].append(v.ia2.implicit)
            else:
                for iv in v.ia2:
                    g[1].append(iv.implicit)
        if hasattr(v, 'im3'):
            if hasattr(v.im3, 'implicit'):
                g[1].append(v.im3.implicit)
            else:
                for iv in v.im3:
                    g[1].append(iv.implicit)
        if hasattr(v, 'ia3'):
            if hasattr(v.ia3, 'implicit'):
                g[1].append(v.ia3.implicit)
            else:
                for iv in v.ia3:
                    g[1].append(iv.implicit)
        if hasattr(v, 'im4'):
            if hasattr(v.im4, 'implicit'):
                g[1].append(v.im4.implicit)
            else:
                for iv in v.im4:
                    g[1].append(iv.implicit)
        if hasattr(v, 'ia4'):
            if hasattr(v.ia4, 'implicit'):
                g[1].append(v.ia4.implicit)
            else:
                for iv in v.ia4:
                    g[1].append(iv.implicit)
        if hasattr(v, 'im5'):
            if hasattr(v.im5, 'implicit'):
                g[1].append(v.im5.implicit)
            else:
                for iv in v.im5:
                    g[1].append(iv.implicit)
        if hasattr(v, 'ia5'):
            if hasattr(v.ia5, 'implicit'):
                g[1].append(v.ia5.implicit)
            else:
                for iv in v.ia5:
                    g[1].append(iv.implicit)
        if hasattr(v, 'im6'):
            if hasattr(v.im6, 'implicit'):
                g[1].append(v.im6.implicit)
            else:
                for iv in v.im6:
                    g[1].append(iv.implicit)
        if hasattr(v, 'ia6'):
            if hasattr(v.ia6, 'implicit'):
                g[1].append(v.ia6.implicit)
            else:
                for iv in v.ia6:
                    g[1].append(iv.implicit)
        if hasattr(v, 'im7'):
            if hasattr(v.im7, 'implicit'):
                g[1].append(v.im7.implicit)
            else:
                for iv in v.im7:
                    g[1].append(iv.implicit)
        if hasattr(v, 'ia7'):
            if hasattr(v.ia7, 'implicit'):
                g[1].append(v.ia7.implicit)
            else:
                for iv in v.ia7:
                    g[1].append(iv.implicit)

    if is_adam:
        optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
    else:
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)

    optimizer.add_param_group({'params': g[0], 'weight_decay': decay})  # add g0 with weight_decay
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
    logger.info(f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups "
                f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias")
    return optimizer


def train(proj_info, hyp, opt, data_dict, tb_writer=None):
    # Options ------------------------------------------------------------------
    save_dir, epochs, batch_size, weights, rank, local_rank, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.global_rank, opt.local_rank, opt.freeze

    userid, project_id, task, nas, target, target_acc = \
        proj_info['userid'], proj_info['project_id'], proj_info['task_type'], \
        proj_info['nas'], proj_info['target_info'], proj_info['acc']

    # save internally
    info = Info.objects.get(userid=userid, project_id=project_id)
    info.status = "running"
    info.progress = "train"
    info.save()

    # Directories --------------------------------------------------------------
    # if not opt.resume and opt.lt != 'incremental' and opt.lt != 'transfer':
    #     logger.warn(f'FileSystem: {save_dir} already exists. It will deleted and remade')
    #     shutil.rmtree(opt.save_dir)
    if os.path.isdir(str(save_dir)):
        shutil.rmtree(save_dir) # remove 'autonn' dir
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # CUDA device --------------------------------------------------------------
    device_str = ''
    for i in range(torch.cuda.device_count()):
        device_str = device_str + str(i) + ','
    opt.device = device_str[:-1]
    # opt.total_batch_size = opt.batch_size
    device, device_info = select_device_and_info(opt.device)

    system = {}
    system['torch'] = torch.__version__
    system['cuda'] = torch.version.cuda
    system['cudnn'] = torch.backends.cudnn.version() / 1000.0
    server_gpu_mem = 0
    for i, d in enumerate(device_info):
        system_info = {}
        system_info['devices'] = d[0]
        system_info['gpu_model'] =  d[1]
        if d[0] == 'CPU':
            d[2] = "0.0"
        system_info['memory'] = d[2]
        server_gpu_mem = round(float(d[2]))
        system[f'{i}'] = system_info
    # system_content = json.dumps(system)
    status_update(userid, project_id,
                  update_id="system",
                  update_content=system)

    # Configure ----------------------------------------------------------------
    plots = not opt.evolve # create plots
    cuda = device.type != 'cpu'
    seed = opt.seed if opt.seed else 0
    if opt.loss_name == 'TAL':
        init_seeds_v9(seed + 1 + rank, deterministic=True) # from yolov9
    else:
        init_seeds(2 + rank)
    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    ch = int(data_dict.get('ch', 3))
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Model --------------------------------------------------------------------
    pretrained = weights.endswith('.pt')
    logger.info(f'\n{colorstr("Models: ")}Pretrained model exists? {pretrained}')
    if pretrained:
        ''' transfer learning / fine tuning / resume '''
        with torch_distributed_zero_first(local_rank): # rank):
            attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        exclude = []

        if task == 'classification':
            model = ClassifyModel(
                opt.cfg or ckpt['model'].yaml, 
                ch=ch, 
                nc=nc
            ).to(device)
        elif task == 'detection':
            if nas or target == 'Galaxy_S22':
                model = NASModel(
                    opt.cfg or ckpt['model'].yaml, 
                    ch=ch, 
                    nc=nc, 
                    anchors=hyp.get('anchors')
                ).to(device)  # create
            elif opt.loss_name == 'TAL':
                model = DetectionModel(
                    opt.cfg or ckpt['model'].yaml,
                    ch=ch,
                    nc=nc,
                    anchors=hyp.get('anchors')
                ).to(device)  # create
            else:
                model = Model(
                    opt.cfg or ckpt['model'].yaml,
                    ch=ch,
                    nc=nc,
                    anchors=hyp.get('anchors')
                ).to(device)  # create
            exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys

        logger.info(f'{colorstr("Models: ")}'
                    f'Loading and overwrite weights from the pretrained model...')
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(
            state_dict, 
            model.state_dict(), 
            exclude=exclude
        )  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info(f'{colorstr("Models: ")}'
                    f'Transferred {len(state_dict)}/{len(model.state_dict())} items from {weights}')  # report
    else:
        ''' learning from scratch '''
        if task == 'classification':
            model = ClassifyModel(
                opt.cfg, 
                ch=ch, 
                nc=nc
            ).to(device)
        elif task == 'detection':
            if nas or target == 'Galaxy_S22':
                model = NASModel(
                    opt.cfg, 
                    ch=ch, 
                    nc=nc, 
                    anchors=hyp.get('anchors')
                ).to(device)  # create
            elif opt.loss_name == 'TAL':
                model = DetectionModel(
                    opt.cfg or ckpt['model'].yaml,
                    ch=ch,
                    nc=nc,
                    anchors=hyp.get('anchors')
                ).to(device)  # create
            else:
                model = Model(
                    opt.cfg,
                    ch=ch,
                    nc=nc,
                    anchors=hyp.get('anchors')
                ).to(device)  # create
    amp_enable = check_amp(model)

    # status_update(userid, project_id,
    #               update_id="model",
    #               update_content=model.nodes_info)
    status_update(userid, project_id,
                  update_id="model_summary",
                  update_content=model.briefs)

    # Freeze -------------------------------------------------------------------
    # parameter names to freeze (full or partial)
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            logger.info('freezing %s' % k)
            v.requires_grad = False

    # Image sizes --------------------------------------------------------------
    imgsz, imgsz_test = [x for x in opt.img_size]
    if task == 'detection':
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        # verify imgsz are gs-multiples (gs=grid stride)
        imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]

    # Batch size ---------------------------------------------------------------
    if opt.resume:
        logger.info('='*100)
        bs_factor = opt.bs_factor # it would be 0.1 less than previous one
        logger.info(f"bs_factor = {bs_factor}")
        logger.info('='*100)
    else:
        bs_factor = 0.8
        opt.bs_factor = bs_factor

    if opt.lt == 'incremental' and opt.weights:
        batch_size = opt.batch_size
    else:
        autobatch_rst = get_batch_size_for_gpu( 
            userid,
            project_id,
            model,
            ch,
            imgsz,
            bs_factor,
            amp_enabled=amp_enable,
            max_search=True 
        )
        batch_size = int(autobatch_rst) # autobatch_rst = result * bs_factor * gpu_number
    
    batch_size = min(batch_size, server_gpu_mem*2)
    logger.info(f'{colorstr("AutoBatch: ")}'
                f'Limit batch size {batch_size} (up to 2 x GPU memory {server_gpu_mem}G)')

    if opt.local_rank != -1: # DDP mode
        logger.info(f'LOCAL RANK is not -1; Multi-GPU training')
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        # assert batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        total_batch_size = opt.world_size * batch_size # assume all gpus are identical
    else: # Single-GPU
        total_batch_size = batch_size

    opt.batch_size = batch_size
    opt.total_batch_size = total_batch_size
    info.batch_size = batch_size
    info.batch_multiplier = bs_factor
    info.save()
    # print('batch size determined')
    # info.print()
    status_update(userid, project_id,
                  update_id="arguments",
                  update_content=vars(opt))

    # Dataset ------------------------------------------------------------------
    with torch_distributed_zero_first(local_rank): #rank):
        check_dataset(data_dict)  # check
    train_path = data_dict['train']
    test_path = data_dict['val']
    is_coco = True if data_dict['dataset_name'] == 'coco' and 'coco' in train_path else False

    # Optimizer ----------------------------------------------------------------
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / opt.total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= opt.total_batch_size * accumulate / nbs  # scale weight_decay
    weight_decay_, momentum_, lr0_ = hyp['weight_decay'], hyp['momentum'], hyp['lr0']
    optimizer = get_optimizer(model, opt.adam, lr0_, momentum_, weight_decay_)

    # Scheduler ----------------------------------------------------------------
    # https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    if opt.linear_lr:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # if plots: # pytorch warning! scheduler update before optimizer update
    #     plot_lr_scheduler(optimizer, scheduler, epochs, save_dir)

    # EMA (required) -----------------------------------------------------------
    ema = ModelEMA(model) if rank in [-1, 0] else None
    if ema != None:
        logger.info(f'\n{colorstr("EMA: ")}Using ModelEMA()')

    # Resume (option) ----------------------------------------------------------
    # see line 133 - 151, it is connected to the model loading procedure
    # we should delete ckpt, state_dict to avoid cuda memory leak 
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']
        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']
        # Results
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt
        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs <= start_epoch:
            finetune_epoch = vars(opt).get('finetune_epoch', 1)
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch']+1, finetune_epoch))
            epochs += finetune_epoch # ckpt['epoch']  # finetune additional epochs
        del ckpt, state_dict

    # DP mode (option) ---------------------------------------------------------
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        logger.warning('Using DataParallel(): not recommened, use DDP instead.')

    # SyncBatchNorm (option) ---------------------------------------------------
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # TrainDataloader ----------------------------------------------------------
    if task == 'detection':
        if opt.loss_name == 'TAL': # v9
            dataloader, dataset = create_dataloader_v9(
                train_path,
                imgsz,
                batch_size // opt.world_size,
                gs,
                opt.single_cls,
                hyp=hyp,
                augment=True,
                cache=opt.cache_images,
                rect=opt.rect,
                rank=local_rank,
                workers=opt.workers,
                image_weights=opt.image_weights,
                close_mosaic=opt.close_mosaic != 0,
                quad=opt.quad,
                prefix='train',
                shuffle=True,
                min_items=0, #opt.min_items,
                uid=userid,
                pid=project_id,
            )
        else: # v7
            dataloader, dataset = create_dataloader(
                userid,
                project_id,
                train_path,
                imgsz,
                batch_size // opt.world_size,
                gs, # stride
                opt, # single_cls
                hyp=hyp,
                augment=True,
                cache=opt.cache_images,
                rect=opt.rect,
                rank=rank,
                world_size=opt.world_size,
                workers=opt.workers,
                image_weights=opt.image_weights,
                quad=opt.quad,
                prefix='train'
            )
        mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    elif task == 'classification':
        try:
            # from torchvision import datasets, transforms
            # train_transform = transforms.Compose(
            #     [
            #         transforms.Grayscale(num_output_channels=1),
            #         transforms.Resize(imgsz),
            #         transforms.CenterCrop(imgsz),
            #         transforms.ToTensor(),
            #         transforms.Normalize(mean=hyp['mean'], std=hyp['std']),
            #     ]
            # )
            # dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            train_transform = A.Compose(
                [
                    # A.ToGray(p=1.0),
                    A.Resize(height=imgsz, width=imgsz),
                    A.Affine(rotate=hyp['rotate'], 
                             scale=hyp['scale'],
                             shear=hyp['shear'],
                    ),
                    A.OneOf(
                        [
                            A.ElasticTransform(
                                p=hyp['elastic_tr']['p'],
                                alpha=hyp['elastic_tr']['alpha'],
                                sigma=hyp['elastic_tr']['sigma'],
                            ),
                            A.GridDistortion(
                                p=hyp['grid_distr']['p'],
                                num_steps=hyp['grid_distr']['step'],
                                distort_limit=hyp['grid_distr']['distort_limit'],
                            ),
                            A.OpticalDistortion(
                                p=hyp['optical_distr']['p'],
                                distort_limit=hyp['optical_distr']['distort_limit'], 
                                shift_limit=hyp['optical_distr']['shift_limit'], 
                            ),
                        ],
                        p=hyp['one_of_tr_prob'],
                    ),
                    A.HorizontalFlip(p=hyp['hflip']),
                    A.VerticalFlip(p=hyp['vflip']),
                    A.Normalize(mean=hyp['mean'], std=hyp['std']),
                    ToTensorV2(),
                ]
            )
            dataset = AlbumentationDatasetImageFolder(
                root=train_path, 
                transform=train_transform, 
                ch=ch
            )
            dataset_info = {}
            total_files_cnt = sum([len(f) for r,d,f in os.walk(train_path)])
            dataset_info['total'] = len(dataset.imgs)
            dataset_info['current'] = len(dataset.imgs)
            dataset_info['found'] = total_files_cnt
            dataset_info['missing'] = total_files_cnt - len(dataset.imgs)
            # dataset_info['empty'] = ne
            # dataset_info['corrupted'] = nc

            status_update(userid, project_id,
                          update_id=f"train_dataset",
                          update_content=dataset_info)
            mlc = len( list(set(dataset.classes)) ) - 1 # label indices start from 0, 1, ..
        except Exception as e:
            logger.warning(f'Failed to load dataset {train_path}: {e}')

        try:
            from torch.utils.data import DataLoader
            dataloader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=opt.workers,
                                    drop_last=True)
        except Exception as e:
            logger.warning(f'Failed to get dataloder for training: {e}')
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Process 0 (TestDataLoader) -----------------------------------------------
    if rank in [-1, 0]:
        if task == 'detection':
            if opt.loss_name == 'TAL': # v9
                testloader = create_dataloader_v9(
                    test_path,
                    imgsz,
                    batch_size // opt.world_size * 2,
                    gs,
                    opt.single_cls,
                    hyp=hyp,
                    cache=opt.cache_images and not opt.notest,
                    rect=True,
                    rank=-1,
                    workers=opt.workers * 2,
                    pad=0.5,
                    prefix='val',
                    uid=userid,
                    pid=project_id,
                )[0]
            else:
                testloader = create_dataloader(
                    userid,
                    project_id,
                    test_path,
                    imgsz_test,
                    batch_size // opt.world_size, # * 2 may lead out-of-memory
                    gs, # stride
                    opt, # single_cls
                    hyp=hyp,
                    cache=opt.cache_images and not opt.notest,
                    rect=True,
                    rank=-1,
                    world_size=opt.world_size,
                    workers=opt.workers * 2,
                    pad=0.5,
                    prefix='val'
                )[0]

            if not opt.resume:
                labels = np.concatenate(dataset.labels, 0)
                c = torch.tensor(labels[:, 0])  # classes
                # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
                # model._initialize_biases(cf.to(device))

                # Anchors
                if not opt.noautoanchor:
                    check_anchors(userid, project_id, dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
                model.half().float()  # pre-reduce anchor precision

                # if plots:
                #     plot_labels(labels, names, save_dir, loggers=None)
                #     if tb_writer:
                #         tb_writer.add_histogram('classes', c, 0)
        elif task == 'classification':
            try:
                # val_transform = transforms.Compose(
                #     [
                #         transforms.Grayscale(num_output_channels=1),
                #         transforms.Resize(imgsz),
                #         transforms.CenterCrop(imgsz),
                #         transforms.ToTensor(),
                #         transforms.Normalize(mean=hyp['mean'], std=hyp['std']),
                #     ]
                # )
                # val_dataset = datasets.ImageFolder(root=test_path, transform=val_transform)
                val_transform = A.Compose(
                    [
                        # A.ToGray(p=1.0),
                        A.Resize(height=imgsz, width=imgsz),
                        A.Normalize(mean=hyp['mean'], std=hyp['std']),
                        ToTensorV2(),
                    ]
                )
                val_dataset = AlbumentationDatasetImageFolder(
                    root=test_path, 
                    transform=val_transform, 
                    ch=ch
                )
                dataset_info = {}
                total_files_cnt = sum([len(f) for r,d,f in os.walk(test_path)])
                dataset_info['total'] = len(val_dataset.imgs)
                dataset_info['current'] = len(val_dataset.imgs)
                dataset_info['found'] = total_files_cnt
                dataset_info['missing'] = total_files_cnt - len(val_dataset.imgs)
                # dataset_info['empty'] = ne
                # dataset_info['corrupted'] = nc

                status_update(userid, project_id,
                              update_id=f"val_dataset",
                              update_content=dataset_info)
            except Exception as e:
                logger.warning(f'Failed to load dataset {train_path}: {e}')
            try:
                from torch.utils.data import DataLoader
                testloader = DataLoader(val_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=opt.workers,
                                        drop_last=False)
            except Exception as e:
                logger.warning(f'Failed to get dataloder for training: {e}')
            # a bit trick
            anchor_summary = {}
            aat, bpr = 0., 0.
            anchor_summary['anchor2target_ratio'] = f"{aat:.2f}"
            anchor_summary['best_possible_recall'] = f"{bpr:.4f}"
            status_update(userid, project_id,
                          update_id="anchors",
                          update_content=anchor_summary)

    # DDP mode -----------------------------------------------------------------
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
                    # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
                    find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules()))
        logger.info('Using DDP()')

    # Model parameters ---------------------------------------------------------
    if task == 'detection':
        # nl = model.module.model[-1].nl if is_parallel(model) else model.model[-1].nl
        nl = de_parallel(model).model[-1].nl # number of detection layers (to scale hyps)
        if opt.loss_name != 'TAL': # v7
            hyp['box'] *= 3. / nl  # scale to layers
            hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
            hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
            model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
            model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
        else:
            model.class_weights = labels_to_class_weights_v9(dataset.labels, nc).to(device) * nc  # attach class weights
        hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.names = names

    # Save training settings ---------------------------------------------------
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)
    status_update(userid, project_id,
              update_id="hyperparameter",
              update_content=hyp)

    # Loss function ------------------------------------------------------------
    if task == 'classification':
        if opt.loss_name == 'CE':
            if opt.label_smoothing > 0.0:
                ls_alpha = hyp['label_smoothing'] = opt.label_smoothing
            else:
                ls_alpha = 0.1
            compute_loss = nn.CrossEntropyLoss(label_smoothing=ls_alpha) # 0.1
        elif opt.loss_name == 'FL':
            compute_loss = FocalLossCE()
        else:
            logger.warning(f'not supported loss function {opt.loss_name}')
    else: # if task == 'detection':
        if opt.loss_name == 'TAL':
            compute_loss = ComputeLoss_v9(model) #ComputeLossTAL(model)
        else:
            if opt.loss_name == 'OTA':
                compute_loss_ota = ComputeLossOTA(model)  # init loss class
            elif opt.loss_name == 'AuxOTA':
                compute_loss_ota = ComputeLossAuxOTA(model)  # init loss class
            else:
                compute_loss_ota = None
            compute_loss = ComputeLoss(model)  # init loss class
    
    # Ealry Stopper ------------------------------------------------------------
    # how many epochs could you be waiting for patiently 
    # although accuracy was not better?
    patience_epochs = opt.patience if opt.patience else ''
    stopper, stop = EarlyStopping(patience=patience_epochs), False

    # Start training -----------------------------------------------------------

    t0 = time.time() # time_synchronized()
    nb = len(dataloader)  # number of batches
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    if task == 'detection':
        maps = np.zeros(nc)  # mAP per class
        results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    elif task == 'classification':
        results = (0, 0) # val_accuracy, val_loss
    last_opt_step = -1
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=amp_enable)

    logger.info(f'\n{colorstr("Train: ")}Image sizes {imgsz}, {imgsz_test}\n'
                f'       Using {dataloader.num_workers} dataloader workers\n'
                f'       Logging results to {save_dir}\n'
                f'       Starting training for {epochs} epochs...')
    train_start = {}
    train_start['status'] = 'start'
    train_start['epochs'] = epochs
    status_update(userid, project_id,
              update_id="train_start",
              update_content=train_start)
    info = Info.objects.get(userid=userid, project_id=project_id)
    info.progress = "train"
    info.save()
    logger.info(f'{colorstr("Train: ")}start epoch = {start_epoch}, final epoch = {epochs}')

    # torch.save(model, wdir / 'init.pt')
    for epoch in range(start_epoch, epochs):
        t_epoch = time.time() #time_synchronized()
        model.train()

        # Update image weights (optional)
        if opt.image_weights:
            if task == 'detection':
                # Generate indices
                if rank in [-1, 0]:
                    cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                    iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                    dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
                # Broadcast if DDP
                if rank != -1:
                    indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                    dist.broadcast(indices, 0)
                    if rank != 0:
                        dataset.indices = indices.cpu().numpy()
            else:
                logger.warning(f"taks = {task}: only detection task supports image weight")

        # Stop mosaic augmentation
        if epoch == (epochs - opt.close_mosaic):
            logger.info("Closing dataloader mosaic")
            dataset.mosaic = False

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        # mean losses
        if task == 'detection':
            if opt.loss_name == 'TAL':
                mloss = torch.zeros(3, device=device)
            else:
                mloss = torch.zeros(4, device=device)
        elif task == 'classification':
            mloss = torch.zeros(1, device=device)
            macc = torch.zeros(1, device=device)

        # distribute data
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)

        # progress bar
        pbar = enumerate(dataloader)
        # pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)

        if opt.loss_name == 'TAL': # v9
            title_s = ('\n' + '%11s' * 7) % (
                'Epoch', 'GPU_Mem', 'box', 'cls', 'dfl', 'Labels', 'Img_Size'
            )
        else: # v7
            title_s = ('\n' + '%11s' * 8) % (
                'Epoch', 'GPU_Mem', 'Box', 'Obj', 'Cls', 'Total', 'Labels', 'Img_Size'
            )

        # if rank in [-1, 0]:
        #     pbar = tqdm(
        #         pbar,
        #         desc=title_s,
        #         total=nb,
        #         miniters=1,
        #         bar_format=TQDM_BAR_FORMAT,
        #     )  # progress bar

        train_loss = {}
        optimizer.zero_grad()
        # training batches start ===============================================
        if task == 'detection':
            for i, (imgs, targets, paths, _) in pbar:
                # logger.info(f'step-{i} start')
                t_batch = time.time() #time_synchronized()
                ni = i + nb * epoch  # number integrated batches (since train start)
                imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
                
                # Warmup
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                    accumulate = max(1, np.interp(ni, xi, [1, nbs / opt.total_batch_size]).round())
                    for j, x in enumerate(optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        # x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                        x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])
                    # hyp['current_momentum'] = x['momentum']
                    # hyp[f'current_bias_lr'] = optimizer.param_groups[0]['lr']
                    # hyp[f'current_lr'] = x['lr']
                    # status_update(userid, project_id,
                    #           update_id="hyperparameter",
                    #           update_content=hyp)
                    logger.info(f'step-{i} warming up... '
                                f'bias lr = {optimizer.param_groups[0]["lr"]}, '
                                f'lr = {x["lr"]}, '
                                f'momentum={x["momentum"]}')

                # Multi-scale
                if opt.multi_scale:
                    sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                    sf = sz / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                        imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
                    # logger.info(f'step-{i} multi-scaling.. size={ns}')

                # Forward
                # optimizer.zero_grad()
                with amp.autocast(enabled=amp_enable):
                    pred = model(imgs)  # forward
                    # logger.info(f'step-{i} forward done')
                    # if 'loss_ota' not in hyp or hyp['loss_ota'] == 1:
                    if 'OTA' in opt.loss_name: #opt.loss_name == 'OTA':
                        loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)  # loss scaled by batch_size
                    else:
                        loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                    # logger.info(f'step-{i} compute loss done {loss}')
                    if rank != -1:
                        loss *= opt.world_size  # gradient averaged between devices in DDP mode
                    if opt.quad:
                        loss *= 4.

                # Backward
                loss_items_na = loss_items.cpu().numpy()
                is_missing_value = False
                for l in loss_items_na:
                    if np.isinf(l) or np.isnan(l):
                        is_missing_value = True
                        break
                if not is_missing_value:
                    scaler.scale(loss).backward()
                    # logger.info(f'step-{i} backward done')
                else:
                    logger.warning(f'step-{i} can not backward')

                # Optimize
                # if ni % accumulate == 0:
                if ni - last_opt_step >= accumulate:
                    # https://pytorch.org/docs/master/notes/amp_examples.html
                    scaler.unscale_(optimizer)  # unscale gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients                    
                    scaler.step(optimizer)  # optimizer.step
                    scaler.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)
                    last_opt_step = ni
                    # logger.info(f'step-{i} optimizer update done')

                # Report & Plot(option)
                if rank in [-1, 0]:
                    mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)


                    if not is_missing_value:
                        mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses

                    if opt.loss_name == 'TAL':
                        s = ('%11s' * 2 + '%11.4g' * 5) % (
                            '%g/%g' % (epoch, epochs - 1), mem,         # epoch/total, gpu_mem
                            *mloss, targets.shape[0], imgs.shape[-1]    # box, dfl, cls, labels, imgsz
                        )
                    else:
                        s = ('%10s' * 2 + '%11.4f' * 4 + '%11.0f' * 2) % (
                            '%g/%g' % (epoch, epochs - 1), mem,         # epoch/total, gpu_mem
                            *mloss, targets.shape[0], imgs.shape[-1]    # box, obj, cls, total, labels, imgsz
                        )

                    # pbar.set_description(s)
                    mloss_np = mloss.to('cpu').numpy()
                    mloss_list = mloss_np.tolist()
                    train_loss['epoch'] = epoch + 1
                    train_loss['total_epoch'] = epochs
                    train_loss['gpu_mem'] = mem
                    train_loss['box'] = mloss_list[0]
                    train_loss['obj'] = mloss_list[1]
                    train_loss['cls'] = mloss_list[2]
                    if opt.loss_name == 'TAL':
                        train_loss['total'] = train_loss['box'] + train_loss['obj'] + train_loss['cls'] # sum(mloss_list)
                    else:
                        train_loss['total'] = mloss_list[3]
                    train_loss['label'] = targets.shape[0]
                    train_loss['step'] = i + 1
                    train_loss['total_step'] = nb
                    train_loss['time'] = f"{(time.time() - t_batch):.1f} s"

                    status_update(userid, project_id,
                                  update_id="train_loss",
                                  update_content=train_loss)
                    ten_percent_cnt = int((i+1)/nb*10+0.5)
                    bar = '|'+ '#'*ten_percent_cnt + ' '*(10-ten_percent_cnt)+'|'
                    sec = time.time()-t_epoch
                    elapsed = str(datetime.timedelta(seconds=sec)).split('.')[0]
                    content_s = s + (f'{bar}{(i+1)/nb*100:3.0f}% {i+1:4.0f}/{nb:4.0f} {elapsed}')

                    if (i % 50) == 0:
                        logger.info(title_s)
                    logger.info(content_s)

                    # if plots and ni < 10:
                    #     f = save_dir / f'train_batch{ni}.jpg'  # filename
                    #     Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
                    #     if tb_writer:
                    #         tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    #         tb_writer.add_graph(torch.jit.trace(model, imgs, strict=False), [])  # add model graph
        elif task == 'classification':
            logger.info(('\n' + '%10s' * 6) % ('TrainEpoch', 'GPU_Mem', 'Batch', 'mLoss', 'mAcc', '=curr/all'))
            accumulated_imgs_cnt = 0
            tacc = 0
            for i, (imgs, targets) in pbar:
                t_batch = time.time() #time_synchronized()
                ni = i + nb * epoch  # number integrated batches (since train start)
                imgs = imgs.float().to(device, non_blocking=True)
                targets = targets.to(device)

                # Warmup
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    accumulate = max(1, np.interp(ni, xi, [1, nbs / opt.total_batch_size]).round())
                    for j, x in enumerate(optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])
                            hyp[f'momentum_group{j}'] = x['momentum']
                        else:
                            hyp[f'momentum_group{j}'] = None
                        hyp[f'lr_group{j}'] = x['lr']
                    # status_update(userid, project_id,
                    #           update_id="hyperparameter",
                    #           update_content=hyp)

                # Forward
                optimizer.zero_grad()
                with amp.autocast(enabled=amp_enable):
                    pred = model(imgs)  # forward
                    loss = compute_loss(pred, targets)
                    _, out = pred.max(1)  # top-1_pred_value, top-1_pred_cls_idx
                    if rank != -1:
                        loss *= opt.world_size  # gradient averaged between devices in DDP mode
                    if opt.quad:
                        loss *= 4.
                    acc = torch.eq(out, targets).sum()
                    # print(acc.shape, acc)
                    # acc = out.eq(targets.view_as(out)).sum() #.item()


                # Backward
                scaler.scale(loss).backward()

                # Optimize
                if ni % accumulate == 0:
                    scaler.step(optimizer)  # optimizer.step
                    scaler.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)

                # Print
                if rank in [-1, 0]:
                    mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                    mloss = (mloss * i + loss) / (i + 1)  # mean train loss
                    macc = (macc * accumulated_imgs_cnt + acc) / (accumulated_imgs_cnt + len(imgs)) # mean train accuracy
                    tacc += acc.item()
                    accumulated_imgs_cnt += len(imgs)

                    s = ('%10s' * 2 + '%10.4g' * 3 + '%10s') % (
                        '%g/%g' % (epoch, epochs - 1), mem,     # epoch/total, gpu_mem
                        targets.shape[0], mloss, macc,          # labels, loss, acc
                        '%g/%g' % (tacc, accumulated_imgs_cnt)  # correct/all
                    ) #imgs.shape[-1])

                    mloss_item = mloss.item()
                    macc_item = macc.item()
                    train_loss['epoch'] = epoch + 1
                    train_loss['total_epoch'] = epochs
                    train_loss['gpu_mem'] = mem
                    train_loss['box'] = accumulated_imgs_cnt # TODO: box -> images
                    train_loss['obj'] = tacc # TODO: obj -> correct
                    train_loss['cls'] = macc_item # TODO: cls -> acc
                    train_loss['total'] = mloss_item # TODO: total -> loss
                    train_loss['label'] = targets.shape[0]
                    train_loss['step'] = i + 1
                    train_loss['total_step'] = nb
                    train_loss['time'] = f"{(time.time() - t_batch):.1f} s"

                    status_update(userid, project_id,
                                  update_id="train_loss",
                                  update_content=train_loss)
                    if len(dataloader) -1 == i:
                        logger.info(f'{s}')
        # training batches end =================================================

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        for i, lr_element in enumerate(lr):
            hyp[f'lr_group{i}'] = lr_element
        # status_update(userid, project_id,
        #           update_id="hyperparameter",
        #           update_content=hyp)
        scheduler.step()

        # (DDP process 0 or single-GPU) test ===================================
        if rank in [-1, 0]:
            # mAP
            if task == 'detection':
                if opt.loss_name == 'TAL':
                    ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
                else:
                    ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            elif task == 'classification':
                ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not opt.notest or final_epoch:  # Calculate mAP
                # wandb_logger.current_epoch = epoch + 1
                if task == 'detection':
                    # results: tuple    (mp, mr, map50, map, box, obj, cls)
                    # maps: numpy array (ap0, ap1, ..., ap79)  : mAP per cls
                    # times: tuple      (inf, nms, total, imgsz, imgsz, batchsz)
                    results, maps, times = test.test(
                        proj_info,
                        data_dict,
                        batch_size=batch_size // opt.world_size, # multiplying by 2 may cause out of gpu memory
                        imgsz=imgsz_test,
                        model=ema.ema,
                        single_cls=opt.single_cls,
                        dataloader=testloader,
                        save_dir=save_dir,
                        verbose=nc < 50 and final_epoch,
                        plots=plots and final_epoch,
                        # wandb_logger=wandb_logger,
                        half_precision=True,
                        compute_loss=compute_loss,
                        is_coco=is_coco,
                        metric=opt.metric
                    )
                elif task == 'classification':
                    # results: tuple - (val_accuracy, val_loss)
                    # times:   float - total
                    results, times = test.test_cls(
                        proj_info,
                        data_dict,
                        batch_size=batch_size,
                        imgsz=imgsz_test,
                        model=ema.ema,
                        dataloader=testloader,
                        save_dir=save_dir,
                        verbose=nc < 50 and final_epoch,
                        plots=False, #plots and final_epoch,
                        compute_loss=compute_loss,
                        half_precision=True
                    )

            # Write
            with open(results_file, 'a') as f:
                if task == 'detection':
                    f.write(s + '%10.4f' * 7 % results + '\n')  # append metrics, val_loss
                elif task == 'classification':
                    f.write(s + '%10.4f' * 2 % results + '\n') # append val_acc, val_loss
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

            # Log
            if task == 'detection':
                if opt.loss_name == 'TAL': # v9
                    tags = ['train/box_loss', 'train/dfl_loss', 'train/cls_loss',  # train loss
                            'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5_0.95',
                            'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                            'x/lr0', 'x/lr1', 'x/lr2']  # params
                else: # v7
                    tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                            'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5_0.95',
                            'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                            'x/lr0', 'x/lr1', 'x/lr2']  # params
                zip_list = list(mloss[:-1]) + list(results) + lr
            elif task == 'classification':
                tags = ['train/acc', 'train/loss', # train accurarcy and loss
                        'val/acc', 'val/loss', # val accurarcy and loss
                        'x/lr0', 'x/lr1', 'x/lr2']  # params
                zip_list = [macc, mloss] + list(results) + lr

            # for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
            for x, tag in zip(zip_list, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # tensorboard

            epoch_summary = {}
            epoch_summary['total_epoch'] = epochs
            epoch_summary['current_epoch'] = epoch + 1
            if task == 'detection':
                epoch_summary['train_loss_box'] = mloss_list[0]
                epoch_summary['train_loss_obj'] = mloss_list[1]
                epoch_summary['train_loss_cls'] = mloss_list[2]
                if opt.loss_name == 'TAL':
                    epoch_summary['train_loss_total'] = sum(mloss_list)
                else:
                    epoch_summary['train_loss_total'] = mloss_list[3]
                epoch_summary['val_acc_P'] = results[0]
                epoch_summary['val_acc_R'] = results[1]
                epoch_summary['val_acc_map50'] = results[2]
                epoch_summary['val_acc_map'] = results[3]
            elif task == 'classification':
                epoch_summary['train_loss_total'] = mloss_item
                epoch_summary['val_acc_map50'] = macc_item # results[1]
                epoch_summary['val_acc_map'] = results[0]
            epoch_summary['epoch_time'] = (time.time() - t_epoch) # unit: sec
            epoch_summary['total_time'] = (time.time() - t0) / 3600 # unit: hour
            status_update(userid, project_id,
                          update_id="epoch_summary",
                          update_content=epoch_summary)

            # Update best mAP
            if task == 'detection':
                fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            elif task == 'classification':
                fi = results[0] # validation accuracy itself
            stop = stopper(epoch=epoch, fitness=fi)
            if fi > best_fitness:
                best_fitness = fi

            # Save model
            if (not opt.nosave) or (final_epoch and not opt.evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': results_file.read_text(),
                        'model': deepcopy(model.module if is_parallel(model) else model).half(),
                        'ema': deepcopy(ema.ema).half(),
                        # 'model': deepcopy(model.module if is_parallel(model) else model),
                        # 'ema': deepcopy(ema.ema),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),}

                # Save last, best and delete
                if best_fitness == fi:
                    torch.save(ckpt, best)
                    mb = os.path.getsize(best) / 1E6  # filesize
                    logger.info(f'epoch {epoch} : {best} {mb:.1f} MB')
                if not opt.nosave:
                    torch.save(ckpt, last)
                del ckpt

                # Save epoch internally
                info = Info.objects.get(userid=userid, project_id=project_id)
                info.epoch = epoch
                info.save()
        
            # EarlyStopping
            if rank != -1: # if DDP training
                broadcast_list = [stop if rank == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)
                if rank != 0:
                    stop = broadcast_list[0]
            if stop:
                logger.info(f"early stopping...")
                break
        # end validation =======================================================

    # end training -------------------------------------------------------------

    if rank in [-1, 0]:
        # Plots ----------------------------------------------------------------
        if plots:
            if task == 'detection':
                plot_results(
                    save_dir=save_dir,
                    use_dfl=True if opt.loss_name == 'TAL' else False
                )
            elif task == 'classification':
                plot_cls_results(save_dir=save_dir)

        logger.info(f'\n{colorstr("Train: ")}{epoch-start_epoch+1} epochs completed({(time.time() - t0) / 60:.3f} min).')

        # Test best.pt after fusing layers -------------------------------------
        # [tenace's note] argument of type 'PosixPath' is not iterable (??)
        if best.exists():
            if task == 'detection':
                m = os.path.splitext(best)[0] + "_stripped.pt"
                strip_optimizer(deepcopy(best), m, prefix=colorstr("Test: "))
                results, _, _ = test.test(
                    proj_info,
                    data_dict,
                    batch_size=1, # batch_size * 2, # default: 32
                    imgsz=imgsz_test,  # default: 640
                    conf_thres=0.001, # default: 0.001
                    iou_thres=0.7, # default: 0.7
                    single_cls=opt.single_cls, # default: False
                    augment=False, # default: False
                    verbose=False, # default: False
                    model=attempt_load(m, map_location=device, fused=True).half(),
                    dataloader=testloader,
                    save_dir=save_dir,
                    save_txt=False, # default: False
                    save_hybrid=False, # default: False
                    save_conf=False, # default: False
                    save_json=True, #  # default: False # pycocotool
                    plots=False, # default: True
                    compute_loss=None,  # default: None
                    half_precision=False, # default: True
                    trace=False, # default: False
                    is_coco=is_coco, # default: False
                    metric=opt.metric # default: 'v5', possible: 'v7', 'v9'
                )
        #     elif task == 'classification':
        #         results, _ = test.test_cls(proj_info,
        #                                    data_dict,
        #                                    batch_size=batch_size * 2,
        #                                    imgsz=imgsz_test,
        #                                    model=attempt_load(m, device),
        #                                    dataloader=testloader,
        #                                    save_dir=save_dir,
        #                                    plots=False,
        #                                    half_precision=True)

        final_model = best if best.exists() else last  # final model file

    else:
        dist.destroy_process_group()

    # remove Model from cuda ---------------------------------------------------
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # report to PM -------------------------------------------------------------
    mb = os.path.getsize(final_model) / 1E6  # filesize
    train_end = {}
    train_end['status'] = 'end'
    train_end['epochs'] = epochs
    train_end['bestmodel'] = str(final_model)
    train_end['bestmodel_size'] = f'{mb:.1f} MB'
    train_end['time'] = f'{(time.time() - t0) / 3600:.3f} hours'
    status_update(userid, project_id,
                  update_id="train_end",
                  update_content=train_end)
    
    return results, final_model