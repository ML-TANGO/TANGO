# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset.

Models and datasets download automatically from the latest YOLOv5 release.
Models: https://github.com/ultralytics/yolov5/tree/master/models
Datasets: https://github.com/ultralytics/yolov5/tree/master/data
Tutorial: https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data

Usage:
    $ python path/to/train.py --data coco128.yaml \
        -weights yolov5s.pt --img 640  # from pretrained (RECOMMENDED)
    $ python path/to/train.py --data coco128.yaml \
        --weights '' --cfg yolov5s.yaml --img 640  # from scratch
"""

import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import csv

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from tqdm import tqdm

import val  # for end-of-epoch mAP
# from models.experimental import attempt_load
# from models.yolo import Model
from yolov5_utils.autoanchor import check_anchors
from yolov5_utils.autobatch import check_train_batch_size
# from yolov5_utils.callbacks import Callbacks
from datasets import create_dataloader
from yolov5_utils.downloads import attempt_download
from yolov5_utils.general import (
    LOGGER, check_dataset, check_file, check_img_size,
    check_suffix, check_yaml, colorstr, get_latest_run,
    increment_path, init_seeds, intersect_dicts, is_ascii,
    labels_to_class_weights, labels_to_image_weights, methods, one_cycle,
    print_args, print_mutation, strip_optimizer, linear)
from yolov5_utils.loss import ComputeLoss
from yolov5_utils.metrics import fitness
from yolov5_utils.plots import plot_evolve, plot_labels
from yolov5_utils.torch_utils import (
    EarlyStopping, ModelEMA, de_parallel, select_device,
    torch_distributed_zero_first)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# https://pytorch.org/docs/stable/elastic/run.html
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


# hyp is path/to/hyp.yaml or hyp dictionary
def train(hyp, opt, device):
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg,\
        resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights,\
        opt.single_cls, opt.evolve, opt.dataset, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
    opt_dict = vars(opt)
    model = opt_dict.pop('model')
    # _compute_loss = opt_dict.pop('compute_loss')
    # _optimizer = opt_dict.pop('optimizer')
    # _val_data = opt_dict.pop('val_data')
    # _dataset = opt_dict.pop('data')
    # _train_loader = opt_dict.pop('train_data')
    # _lf = opt_dict.pop('lf')
    # _scheduler = opt_dict.pop('scheduler')
    # _device = opt_dict.pop('device')
    # _device = torch.device(
    #     'cuda' if torch.cuda.is_available() else 'cpu') if device is None \
    #     else device
    # callbacks.run('on_pretrain_routine_start')

    # Directories -------------------------------------------------------------
    w = save_dir / 'weights'  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters ---------------------------------------------------------
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ')
                + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Save run settings -------------------------------------------------------
    if not evolve:
        with open(save_dir / 'hyp.yaml', 'w') as f:
            yaml.safe_dump(hyp, f, sort_keys=False)
        with open(save_dir / 'opt.yaml', 'w') as f:
            yaml.safe_dump(opt_dict, f, sort_keys=False)

    # Loggers -----------------------------------------------------------------
    data_dict = None
    # if RANK in [-1, 0]:
    #     # loggers instance
    #     loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)
    #     if loggers.wandb:
    #         data_dict = loggers.wandb.data_dict
    #         if resume:
    #             weights, epochs, hyp, batch_size = opt.weights, opt.epochs,\
    #                   opt.hyp, opt.batch_size

    #     # Register actions
    #     for k in methods(loggers):
    #         callbacks.register_action(k, callback=getattr(loggers, k))

    # Config ------------------------------------------------------------------
    plots = not evolve and not opt.noplots  # create plots
    cuda = device.type != 'cpu'
    init_seeds(1 + RANK)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']
    # number of classes
    # nc = model.nc
    nc = 1 if single_cls else int(data_dict['num_classes'])
    # class names
    names = ['item'] if single_cls and len(data_dict['names']) != 1 \
        else data_dict['names']
    assert len(names) == nc,\
        f'{len(names)} names found for nc={nc} dataset in {data}'  # check
    # is_coco = isinstance(val_path, str) \
    #     and val_path.endswith('coco/val2017.txt')  # COCO dataset

    # Model -------------------------------------------------------------------
    # check_suffix(weights, '.pt')  # check weights
    # pretrained = weights.endswith('.pt')
    pretrained = False
    # if pretrained:
    #     with torch_distributed_zero_first(LOCAL_RANK):
    #         # download if not found locally
    #         weights = attempt_download(weights)
    #     # load checkpoint to CPU to avoid CUDA memory leak
    #     ckpt = torch.load(weights, map_location='cpu')
    #     model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc,
    #                   anchors=hyp.get('anchors')).to(device)  # create
    #     exclude = ['anchor'] if (cfg or hyp.get('anchors')) \
    #          and not resume else []  # exclude keys
    #     # checkpoint state_dict as FP32
    #     csd = ckpt['model'].float().state_dict()
    #     # intersect
    #     csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)
    #     model.load_state_dict(csd, strict=False)  # load
    #     LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items'
    #                 f' from {weights}')  # report
    # else:
    #     model = Model(cfg, ch=3, nc=nc,
    #                   anchors=hyp.get('anchors')).to(device)  # create

    # Freeze ------------------------------------------------------------------
    # layers to freeze
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1
                                      else range(freeze[0]))]
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False

    # Image size --------------------------------------------------------------
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    # verify imgsz is gs-multiple
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)
    # imgsz = opt.imgsz

    # Batch size --------------------------------------------------------------
    # single-GPU only, estimate best batch size
    if RANK == -1 and batch_size == -1:
        batch_size = check_train_batch_size(model, imgsz)
        LOGGER.info(f"on_params_update: batch_size: {batch_size}")

    # Optimizer ---------------------------------------------------------------
    nbs = 64  # nominal batch size
    # accumulate loss before optimizing
    accumulate = max(round(nbs / batch_size), 1)
    # scale weight_decay
    hyp['weight_decay'] *= batch_size * accumulate / nbs
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    g = [], [], []  # optimizer parameter groups
    # normalization layers, i.e. BatchNorm2d()
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g[2].append(v.bias)
        if isinstance(v, bn):  # weight (no decay)
            g[1].append(v.weight)
        # weight (with decay)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            g[0].append(v.weight)

    if opt.optimizer == 'Adam':
        # adjust beta1 to momentum
        optimizer = Adam(g[2], lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))
    elif opt.optimizer == 'AdamW':
        # adjust beta1 to momentum
        optimizer = AdamW(g[2], lr=hyp['lr0'], betas=(hyp['momentum'],
                          0.999))
    else:
        optimizer = SGD(g[2], lr=hyp['lr0'], momentum=hyp['momentum'],
                        nesterov=True)

    optimizer.add_param_group(
        {'params': g[0],
         'weight_decay': hyp['weight_decay']})  # add g0 with weight_decay
    # add g1 (BatchNorm2d weights)
    optimizer.add_param_group({'params': g[1]})
    LOGGER.info(f"{colorstr('optimizer:')}"
                f" {type(optimizer).__name__} with parameter groups "
                f"{len(g[1])} weight (no decay), {len(g[0])} weight,"
                f" {len(g[2])} bias")
    del g

    # Scheduler ---------------------------------------------------------------
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    else:
        # linear
        lf = linear(hyp['lrf'], epochs)
        # lf = lambda \
        #     x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']
    # plot_lr_scheduler(optimizer, scheduler, epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA
    ema = ModelEMA(model) if RANK in [-1, 0] else None

    # Resume
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

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if resume:
            assert start_epoch > 0,\
                f'{weights} training to {epochs} epochs is finished,' \
                f' nothing to resume.'
        if epochs < start_epoch:
            LOGGER.info(f"{weights} has been trained"
                        f" for {ckpt['epoch']} epochs."
                        f" Fine-tuning for {epochs} more epochs.")
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, csd

    # DP mode -----------------------------------------------------------------
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING: DP not recommended,'
                       ' use torch.distributed.run.\n')
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm -----------------------------------------------------------
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)\
              .to(device)
        LOGGER.info('Using SyncBatchNorm()')

    # Trainloader -------------------------------------------------------------
    train_loader, dataset = create_dataloader(
        train_path,
        imgsz,
        batch_size // WORLD_SIZE,
        gs,
        single_cls,
        hyp=hyp,
        augment=True,
        cache=None if opt.cache == 'val' else opt.cache,
        rect=opt.rect,
        rank=LOCAL_RANK,
        workers=workers,
        image_weights=opt.image_weights,
        quad=opt.quad,
        prefix=colorstr('train: '),
        shuffle=True)
    # max label class
    mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())
    nb = len(train_loader)  # number of batches
    assert mlc < nc,\
        f'Label class {mlc} exceeds nc={nc} in {data}.'\
        f' Possible class labels are 0-{nc - 1}'

    # Process 0 ---------------------------------------------------------------
    if RANK in [-1, 0]:
        val_loader = create_dataloader(val_path,
                                       imgsz,
                                       batch_size // WORLD_SIZE * 2,
                                       gs,
                                       single_cls,
                                       hyp=hyp,
                                       cache=None if noval else opt.cache,
                                       rect=True,
                                       rank=-1,
                                       workers=workers * 2,
                                       pad=0.5,
                                       prefix=colorstr('val: '))[0]
        # val_loader = val_data

        if not resume:
            labels = np.concatenate(dataset.labels, 0)
            # c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                plot_labels(labels, names, save_dir)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'],
                              imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

        # callbacks.run('on_pretrain_routine_end')
        LOGGER.info('on_pretrain_routine_end')

    # DDP mode ----------------------------------------------------------------
    if cuda and RANK != -1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    # Model attributes --------------------------------------------------------
    # number of detection layers (to scale hyps)
    nl = de_parallel(model).model[-1].nl
    hyp['box'] *= 3 / nl  # scale to layers
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    # scale to image size and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    # attach class weights
    model.class_weights = \
        labels_to_class_weights(dataset.labels, nc).to(device) * nc
    model.names = names

    # Start training ----------------------------------------------------------
    t0 = time.time()
    # number of warmup iterations, max(3 epochs, 100 iterations)
    nw = max(round(hyp['warmup_epochs'] * nb), 100)
    # limit warmup to < 1/2 of training
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    results = (0, 0, 0, 0, 0, 0, 0)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    stopper = EarlyStopping(patience=opt.patience)
    compute_loss = ComputeLoss(model)  # init loss class
    # callbacks.run('on_train_start')
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')

    # log ---------------------------------------------------------------------
    # fieldnames = ['epoch', 'Precision', 'Recall', 'mAP@0.5',
    #               'mAP@.5:.95', 'lBox', 'lObj', 'lCls',
    #               'neck1-path1', 'neck1-path2', 'neck2-path1', 'neck2-path2',
    #               'neck2-path3', 'neck3-path1', 'neck3-path2', 'neck3-path3',
    #               'neck4-path1', 'neck4-path2', 'neck5-path1', 'neck5-path2',
    #               'neck5-path3']
    fieldnames = ['epoch', 'Precision', 'Recall', 'mAP@0.5',
                  'mAP@.5:.95', 'lBox', 'lObj', 'lCls']
    csvfile = str(save_dir / 'log.csv')
    with open(csvfile, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(start_epoch, epochs):  # epoch -------------------------
        # callbacks.run('on_train_epoch_start')
        LOGGER.info('on_train_epoch_start')
        model.train()

        # Update image weights (optional, single-GPU only)
        if opt.image_weights:
            # class weights
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc
            # image weights
            iw = labels_to_image_weights(dataset.labels,
                                         nc=nc,
                                         class_weights=cw)
            # rand weighted idx
            dataset.indices = random.choices(range(dataset.n),
                                             weights=iw,
                                             k=dataset.n)

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(3, device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%10s' * 7)
                    % ('Epoch', 'gpu_mem', 'box', 'obj',
                       'cls', 'labels', 'img_size'))
        if RANK in (-1, 0):
            # progress bar
            pbar = tqdm(pbar, total=nb,
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        optimizer.zero_grad()
        log_csv = {}
        log_csv['epoch'] = epoch
        for i, (imgs, targets, paths, _) in pbar:  # batch --------------------
            # callbacks.run('on_train_batch_start')
            # number integrated batches (since train start)
            ni = i + nb * epoch
            # uint8 to float32, 0-255 to 0.0-1.0
            imgs = imgs.to(device, non_blocking=True).float() / 255

            # Warmup ----------------------------------------------------------
            if ni <= nw:
                xi = [0, nw]  # x interp
                # iou loss ratio (obj_loss = 1.0 or iou)
                # compute_loss.gr = \
                #     np.interp(ni, xi, [0.0, 1.0])
                accumulate = max(
                    1,
                    np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0,
                    # all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(
                        ni,
                        xi,
                        [hyp['warmup_bias_lr'] if j == 2
                            else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(
                            ni, xi,
                            [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale -----------------------------------------------------
            if opt.multi_scale:
                # size
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    # new shape (stretched to gs-multiple)
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]
                    imgs = nn.functional.interpolate(
                        imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward ---------------------------------------------------------
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                # loss scaled by batch_size
                loss, loss_items = compute_loss(pred, targets.to(device))
                if RANK != -1:
                    # gradient averaged between devices in DDP mode
                    loss *= WORLD_SIZE
                if opt.quad:
                    loss *= 4.

            # Backward --------------------------------------------------------
            scaler.scale(loss).backward()

            # Optimize --------------------------------------------------------
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log -------------------------------------------------------------
            if RANK in (-1, 0):
                # update mean losses
                mloss = (mloss * i + loss_items) / (i + 1)
                gb = torch.cuda.memory_reserved() / 1E9 \
                    if torch.cuda.is_available() else 0
                mem = f'{gb:.3g}G'  # cuda memory (GB)
                pbar.set_description(('%10s' * 2 + '%10.4g' * 5) %
                                     (f'{epoch}/{epochs - 1}',
                                      mem, *mloss, targets.shape[0],
                                      imgs.shape[-1]))
                # callbacks.run('on_train_batch_end',
                #               ni, model, imgs, targets, paths, plots)
                # if callbacks.stop_training:
                #     return
                log_csv['lBox'] = mloss[0].item()
                log_csv['lObj'] = mloss[1].item()
                log_csv['lCls'] = mloss[2].item()
            # end batch -------------------------------------------------------

        # Scheduler -----------------------------------------------------------
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        if RANK in (-1, 0):
            # mAP -------------------------------------------------------------
            # callbacks.run('on_train_epoch_end', epoch=epoch)
            LOGGER.info(f'on_train_epoch_end, epoch={epoch}')
            ema.update_attr(model,
                            include=['yaml', 'nc', 'hyp', 'names', 'stride',
                                     'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                results, maps, _ = val.run(
                    # data_dict,
                    batch_size=batch_size // WORLD_SIZE * 2,
                    imgsz=imgsz,
                    model=ema.ema,
                    single_cls=single_cls,
                    dataloader=val_loader,
                    save_dir=save_dir,
                    plots=False,
                    # callbacks=callbacks,
                    compute_loss=compute_loss)

            # Update best mAP -------------------------------------------------
            # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            fi = fitness(np.array(results).reshape(1, -1))
            # if fi > best_fitness:
            #     best_fitness = fi
            # log_vals = list(mloss) + list(results) + lr
            # callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness,
            #               fi)

            # log -------------------------------------------------------------
            log_csv['Precision'] = results[0]
            log_csv['Recall'] = results[1]
            log_csv['mAP@0.5'] = results[2]
            log_csv['mAP@.5:.95'] = results[3]

            # Save model ------------------------------------------------------
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'date': datetime.now().isoformat()}

                # Save last, best and delete
                # torch.save(ckpt, last)
                # neck_path = []
                # for name, module in model.neck_module.named_modules():
                #     if hasattr(module, 'get_arch_weight'):
                #         print(f'  * neck {name}', end='\t')
                #         for idx, item in enumerate(module.get_arch_weight()):
                #             if item > 0.0:
                #                 print(colorstr(
                #                     "bright_cyan",
                #                     f'[path {idx+1}]:'
                #                     f' {item:0.3f}'), end=' \t')
                #             else:
                #                 print(f'[path {idx+1}]:'
                #                       f' {item:0.3f}', end=' \t')
                #             neck_path.append(item)
                #         print('')

                # log_csv['neck1-path1'] = neck_path[0].item()
                # log_csv['neck1-path2'] = neck_path[1].item()
                # log_csv['neck2-path1'] = neck_path[2].item()
                # log_csv['neck2-path2'] = neck_path[3].item()
                # log_csv['neck2-path3'] = neck_path[4].item()
                # log_csv['neck3-path1'] = neck_path[5].item()
                # log_csv['neck3-path2'] = neck_path[6].item()
                # log_csv['neck3-path3'] = neck_path[7].item()
                # log_csv['neck4-path1'] = neck_path[8].item()
                # log_csv['neck4-path2'] = neck_path[9].item()
                # log_csv['neck5-path1'] = neck_path[10].item()
                # log_csv['neck5-path2'] = neck_path[11].item()
                # log_csv['neck5-path3'] = neck_path[12].item()

                with open(csvfile, 'a') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(log_csv)

                if epoch == 0 or best_fitness < fi:
                    bestmodel = str(w / 'bestmodel.pt')
                    torch.save(ckpt, bestmodel)
                    LOGGER.info(
                        colorstr("bright_green", f'save the best model:')
                        + f' path = {bestmodel} epoch = {epoch} '
                        f'previous best = {best_fitness} current = {fi}')
                    best_fitness = fi
                if (epoch > 0) and (opt.save_period > 0) \
                        and (epoch % opt.save_period == 0):
                    # torch.save(ckpt, w / f'epoch{epoch}.pt')
                    lastmodel = w / 'lastmodel.pt'
                    torch.save(ckpt, lastmodel)
                    LOGGER.info(
                        colorstr("bright_yellow", f'save the last model:')
                        + f' path = {lastmodel} epoch = {epoch} lr = {lr[0]}'
                        f' best = {best_fitness} current = {fi}')
                del ckpt
                # callbacks.run('on_model_save', last, epoch, final_epoch,
                #               best_fitness, fi)
            # Stop Single-GPU -------------------------------------------------
            if RANK == -1 and stopper(epoch=epoch, fitness=fi):
                break

            # Stop DDP
            # stop = stopper(epoch=epoch, fitness=fi)
            # if RANK == 0:
            #    # broadcast 'stop' to all ranks
            #    dist.broadcast_object_list([stop], 0)

        # Stop DPP
        # with torch_distributed_zero_first(RANK):
        # if stop:
        #    break  # must break all DDP ranks

        # end epoch -----------------------------------------------------------
    # end training ------------------------------------------------------------
    if RANK in (-1, 0):
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed'
                    f' in {(time.time() - t0) / 3600:.3f} hours.')
        # for f in last, best:
        #     if f.exists():
        #         strip_optimizer(f)  # strip optimizers
        #         if f is best:
        #             LOGGER.info(f'\nValidating {f}...')
        #             results, _, _ = val.run(
        #                 data_dict,
        #                 batch_size=batch_size // WORLD_SIZE * 2,
        #                 imgsz=imgsz,
        #                 model=attempt_load(f, device).half(),
        #                 # best pycocotools results at 0.65
        #                 iou_thres=0.65 if is_coco else 0.60,
        #                 single_cls=single_cls,
        #                 dataloader=val_loader,
        #                 save_dir=save_dir,
        #                 save_json=is_coco,
        #                 verbose=True,
        #                 plots=plots,
        #                 # callbacks=callbacks,
        #                 # val best model with plots
        #                 compute_loss=compute_loss)
        #             if is_coco:
        #                 callbacks.run('on_fit_epoch_end',
        #                               list(mloss) + list(results) + lr,
        #                               epoch, best_fitness, fi)

        # callbacks.run('on_train_end', last, best, plots, epoch, results)

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default=ROOT / 'yolov5s.pt',
                        help='initial weights path')
    parser.add_argument('--cfg', type=str,
                        default='', help='model.yaml path')
    # parser.add_argument('--cfg', type=str,
    #                     default=ROOT / 'models' / 'yolov5s.yaml',
    #                     help='model.yaml path')
    parser.add_argument('--dataset', type=str,
                        default=ROOT / 'data/coco128.yaml',
                        help='dataset.yaml path')
    parser.add_argument('--hyp', type=str,
                        default=ROOT / 'data/hyps/hyp.scratch-low.yaml',
                        help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16,
                        help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int,
                        default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true',
                        help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False,
                        help='resume most recent training')
    parser.add_argument('--nosave', action='store_true',
                        help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true',
                        help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true',
                        help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true',
                        help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300,
                        help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='',
                        help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram',
                        help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true',
                        help='use weighted image selection for training')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true',
                        help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true',
                        help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str,
                        choices=['SGD', 'Adam', 'AdamW'], default='SGD',
                        help='optimizer')
    parser.add_argument('--sync-bn', action='store_true',
                        help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8,
                        help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train',
                        help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true',
                        help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100,
                        help='EarlyStopping patience')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0],
                        help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1,
                        help='Save checkpoint every x epochs')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='DDP parameter, do not modify')

    # Weights & Biases arguments
    parser.add_argument('--entity', default=None, help='W&B: Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True,
                        default=False, help='W&B: Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1,
                        help='W&B: Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest',
                        help='W&B: Version of dataset artifact to use')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt):
    # Checks
    if RANK in (-1, 0):
        print_args(vars(opt))
        # check_requirements(exclude=['thop'])

    # Resume
    # resume an interrupted run
    # if opt.resume and not check_wandb_resume(opt) and not opt.evolve:
    # if opt.resume and not opt.evolve:
    #     # specified or most recent path
    #     ckpt = opt.resume if isinstance(opt.resume, str) \
    #         else get_latest_run()
    #     assert os.path.isfile(ckpt),\
    #         'ERROR: --resume checkpoint does not exist'
    #     with open(Path(ckpt).parent.parent / 'opt.yaml',
    #               errors='ignore') as f:
    #         opt = argparse.Namespace(**yaml.safe_load(f))  # replace
    #     opt.cfg, opt.weights, opt.resume = '', ckpt, True  # reinstate
    #     LOGGER.info(f'Resuming training from {ckpt}')
    # else:
    #     opt.dataset, opt.cfg, opt.hyp, opt.weights, opt.project = \
    #         check_file(opt.dataset), check_yaml(opt.cfg), \
    #         check_yaml(opt.hyp),\
    #         str(opt.weights), str(opt.project)  # checks
    #     assert len(opt.cfg) or len(opt.weights),\
    #         'either --cfg or --weights must be specified'
    #     if opt.evolve:
    #         # if default project name, rename to runs/evolve
    #         if opt.project == str(ROOT / 'runs/train'):
    #             opt.project = str(ROOT / 'runs/evolve')
    #         # pass resume to exist_ok and disable resume
    #         opt.exist_ok, opt.resume = \
    #             opt.resume, False
    #     if opt.name == 'cfg':
    #         opt.name = Path(opt.cfg).stem  # use model.yaml as name
    #     opt.save_dir = str(increment_path(Path(opt.project)
    #                        / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    # device = opt.device
    # if LOCAL_RANK != -1:
    #     msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
    #     assert not opt.image_weights, f'--image-weights {msg}'
    #     assert not opt.evolve, f'--evolve {msg}'
    #     assert opt.batch_size != -1,\
    #         f'AutoBatch with --batch-size -1 {msg},' \
    #         f' please pass a valid --batch-size'
    #     assert opt.batch_size % WORLD_SIZE == 0,\
    #         f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
    #     assert torch.cuda.device_count() > LOCAL_RANK,\
    #         'insufficient CUDA devices for DDP command'
    #     torch.cuda.set_device(LOCAL_RANK)
    #     device = torch.device('cuda', LOCAL_RANK)
    #     dist.init_process_group(backend="nccl" if dist.is_nccl_available()
    #                             else "gloo")

    # Train
    if not opt.evolve:
        train(opt.model.hyp, opt, device)
        if WORLD_SIZE > 1 and RANK == 0:
            LOGGER.info('Destroying process group... ')
            dist.destroy_process_group()

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata
        # (mutation scale 0-1, lower_limit, upper_limit)
        meta = {
            # initial learning rate (SGD=1E-2, Adam=1E-3)
            'lr0': (1, 1e-5, 1e-1),
            # final OneCycleLR learning rate (lr0 * lrf)
            'lrf': (1, 0.01, 1.0),
            'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
            'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
            'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
            'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
            'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
            'box': (1, 0.02, 0.2),  # box loss gain
            'cls': (1, 0.2, 4.0),  # cls loss gain
            'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
            'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
            'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
            'iou_t': (0, 0.1, 0.7),  # IoU training threshold
            'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
            'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
            # focal loss gamma (efficientDet default gamma=1.5)
            'fl_gamma': (0, 0.0, 2.0),
            'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            # image HSV-Saturation augmentation (fraction)
            'hsv_s': (1, 0.0, 0.9),
            'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
            'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
            'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
            'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
            # image perspective (+/- fraction), range 0-0.001
            'perspective': (0, 0.0, 0.001),
            'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
            'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
            'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
            'mixup': (1, 0.0, 1.0),  # image mixup (probability)
            'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)

        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
        opt.noval, opt.nosave, save_dir = \
            True, True, Path(opt.save_dir)  # only val/save final epoch
        # evolvable indices
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]
        evolve_yaml, evolve_csv = \
            save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            # download evolve.csv if exists
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {evolve_csv}')

        for _ in range(opt.evolve):  # generations to evolve
            # if evolve.csv exists: select best hyps and mutate
            if evolve_csv.exists():
                # Select parent(s)
                # parent selection method: 'single' or 'weighted'
                parent = 'single'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    # weighted selection
                    x = x[random.choices(range(n), weights=w)[0]]
                elif parent == 'weighted':
                    # weighted combination
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                # mutate until a change occurs (prevent duplicates)
                while all(v == 1):
                    v = (g * (npr.random(ng) < mp)
                         * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device)
            # Write mutation results
            print_mutation(results, hyp.copy(), save_dir, opt.bucket)

        # Plot results
        plot_evolve(evolve_csv)
        LOGGER.info(
            f'Hyperparameter evolution finished {opt.evolve} generations\n'
            f"Results saved to {colorstr('bold', save_dir)}\n"
            f'Usage example: $ python train.py --hyp {evolve_yaml}')


def run(**kwargs):
    # Usage:
    # import train
    # train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        if k != 'args':
            setattr(opt, k, v)
        else:  # v : dictionary of arg
            for key, value in v.items():
                setattr(opt, key, value)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
