import logging
import math
import os
import gc
import random
import time
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
from . import test  # import test.py to get mAP after each epoch
from tango.common.models.experimental import attempt_load
from tango.common.models.yolo import Model
from tango.utils.autoanchor import check_anchors
from tango.utils.autobatch import get_batch_size_for_gpu
from tango.utils.datasets import create_dataloader
from tango.utils.general import (   labels_to_class_weights,
                                    labels_to_image_weights,
                                    init_seeds,
                                    fitness,
                                    strip_optimizer,
                                    check_dataset,
                                    check_img_size,
                                    one_cycle,
                                    colorstr
                                )
from tango.utils.google_utils import attempt_download
from tango.utils.loss import ComputeLoss, ComputeLossOTA
from tango.utils.plots import   (   plot_images,
                                    plot_labels,
                                    plot_results,
                                    plot_lr_scheduler
                                )
from tango.utils.torch_utils import (   ModelEMA,
                                        select_device_and_info,
                                        intersect_dicts,
                                        torch_distributed_zero_first,
                                        is_parallel,
                                        time_synchronized
                                    )
from tango.utils.wandb_logging.wandb_utils import WandbLogger

# # YOLOv7-NAS
from tango.common.models.supernet_yolov7 import YOLOSuperNet


logger = logging.getLogger(__name__)


def train(proj_info, hyp, opt, data_dict, tb_writer=None):
    # Options ------------------------------------------------------------------
    save_dir, epochs, batch_size, weights, rank, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.global_rank, opt.freeze

    userid, project_id, task, nas, hpo, target, target_acc = \
        proj_info['userid'], proj_info['project_id'], proj_info['task_type'], \
        proj_info['nas'], proj_info['hpo'], proj_info['target_info'], proj_info['acc']

    # Directories --------------------------------------------------------------
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # Save run settings --------------------------------------------------------
    # with open(save_dir / 'hyp.yaml', 'w') as f:
    #     yaml.dump(hyp, f, sort_keys=False)
    # with open(save_dir / 'opt.yaml', 'w') as f:
    #     yaml.dump(vars(opt), f, sort_keys=False)

    # CUDA device --------------------------------------------------------------
    device_str = ''
    for i in range(torch.cuda.device_count()):
        device_str = device_str + str(i) + ','
    opt.device = device_str[:-1]
    opt.total_batch_size = opt.batch_size
    device, device_info = select_device_and_info(opt.device)

    system = {}
    system['torch'] = torch.__version__
    system['cuda'] = torch.version.cuda
    system['cudnn'] = torch.backends.cudnn.version() / 1000.0
    for i, d in enumerate(device_info):
        system_info = {}
        system_info['devices'] = d[0]
        system_info['gpu_model'] = d[1]
        system_info['memory'] = d[2]
        system[f'{i}'] = system_info
    # system_content = json.dumps(system)
    status_update(userid, project_id,
                  update_id="system",
                  update_content=system)

    # Configure ----------------------------------------------------------------
    plots = not opt.evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)

    # Logging- Doing this before checking the dataset. Might update data_dict
    # loggers = {'wandb': None}  # loggers dict
    # if rank in [-1, 0]:
    #     opt.hyp = hyp  # add hyperparameters
    #     run_id = torch.load(weights, map_location=device).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
    #     wandb_logger = WandbLogger(opt, Path(opt.save_dir).stem, run_id, data_dict)
    #     loggers['wandb'] = wandb_logger.wandb
    #     data_dict = wandb_logger.data_dict
    #     if wandb_logger.wandb:
    #         weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # WandbLogger might update weights, epochs if resuming

    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Model --------------------------------------------------------------------
    pretrained = weights.endswith('.pt')
    if pretrained:
        ''' transfer learning / fine tuning / resume '''
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        if target == 'Galaxy_S22':
            model = YOLOSuperNet(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        else:
            model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        ''' learning from scratch '''
        if target == 'Galaxy_S22':
            model = YOLOSuperNet(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        else:
            model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create

    status_update(userid, project_id,
                  update_id="model",
                  update_content=model.nodes_info)
    status_update(userid, project_id,
                  update_id="model_summary",
                  update_content=model.briefs)

    # Freeze -------------------------------------------------------------------
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            logger.info('freezing %s' % k)
            v.requires_grad = False

    # # Image sizes --------------------------------------------------------------
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # Batch size ---------------------------------------------------------------
    info = Info.objects.get(userid=userid, project_id=project_id)
    bs_factor = 0.8
    autobatch_rst = get_batch_size_for_gpu(userid, project_id, model, 3 if task=='detection' else 1, imgsz, amp_enabled=True)
    batch_size = int(autobatch_rst * bs_factor)
    if opt.local_rank != -1: # DDP mode
        logger.info(f'[ local rank check ]: opt.local_rank is not -1')
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        total_batch_size = opt.world_size * batch_size
    else: # Single-GPU
        total_batch_size = batch_size

    opt.batch_size = batch_size
    opt.total_batch_size = total_batch_size
    info.batch_size = batch_size
    info.batch_multiplier = bs_factor
    info.save()
    status_update(userid, project_id,
                  update_id="arguments",
                  update_content=vars(opt))

    # Dataset ------------------------------------------------------------------
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    train_path = data_dict['train']
    test_path = data_dict['val']
    is_coco = True if data_dict['dataset_name'] == 'coco' and 'coco' in train_path else False

    # Optimizer ----------------------------------------------------------------
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / opt.total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= opt.total_batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay
        if hasattr(v, 'im'):
            if hasattr(v.im, 'implicit'):           
                pg0.append(v.im.implicit)
            else:
                for iv in v.im:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imc'):
            if hasattr(v.imc, 'implicit'):           
                pg0.append(v.imc.implicit)
            else:
                for iv in v.imc:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imb'):
            if hasattr(v.imb, 'implicit'):           
                pg0.append(v.imb.implicit)
            else:
                for iv in v.imb:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imo'):
            if hasattr(v.imo, 'implicit'):           
                pg0.append(v.imo.implicit)
            else:
                for iv in v.imo:
                    pg0.append(iv.implicit)
        if hasattr(v, 'ia'):
            if hasattr(v.ia, 'implicit'):           
                pg0.append(v.ia.implicit)
            else:
                for iv in v.ia:
                    pg0.append(iv.implicit)
        if hasattr(v, 'attn'):
            if hasattr(v.attn, 'logit_scale'):   
                pg0.append(v.attn.logit_scale)
            if hasattr(v.attn, 'q_bias'):   
                pg0.append(v.attn.q_bias)
            if hasattr(v.attn, 'v_bias'):  
                pg0.append(v.attn.v_bias)
            if hasattr(v.attn, 'relative_position_bias_table'):  
                pg0.append(v.attn.relative_position_bias_table)
        if hasattr(v, 'rbr_dense'):
            if hasattr(v.rbr_dense, 'weight_rbr_origin'):  
                pg0.append(v.rbr_dense.weight_rbr_origin)
            if hasattr(v.rbr_dense, 'weight_rbr_avg_conv'): 
                pg0.append(v.rbr_dense.weight_rbr_avg_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_pfir_conv'):  
                pg0.append(v.rbr_dense.weight_rbr_pfir_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_idconv1'): 
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_conv2'):   
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_dw'):   
                pg0.append(v.rbr_dense.weight_rbr_gconv_dw)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_pw'):   
                pg0.append(v.rbr_dense.weight_rbr_gconv_pw)
            if hasattr(v.rbr_dense, 'vector'):   
                pg0.append(v.rbr_dense.vector)

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler ----------------------------------------------------------------
    # https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # if plots: # pytorch warning! scheduler update before optimizer update
    #     plot_lr_scheduler(optimizer, scheduler, epochs, save_dir)

    # EMA (required) -----------------------------------------------------------
    ema = ModelEMA(model) if rank in [-1, 0] else None
    logger.info('Using ModelEMA()')

    # Resume (option) ----------------------------------------------------------
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
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs
        del ckpt, state_dict

    # DP mode (option) ---------------------------------------------------------
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        logger.info('Using DataParallel()')

    # SyncBatchNorm (option) ---------------------------------------------------
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # TrainDataloader ----------------------------------------------------------
    dataloader, dataset = create_dataloader(userid, project_id, train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix='train')
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Process 0 (TestDataLoader) -----------------------------------------------
    if rank in [-1, 0]:
        testloader = create_dataloader(userid, project_id, test_path, imgsz_test, batch_size, gs, opt,  # testloader
                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
                                       world_size=opt.world_size, workers=opt.workers,
                                       pad=0.5, prefix='val')[0]

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                plot_labels(labels, names, save_dir, loggers=None)
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)
            # Anchors
            if not opt.noautoanchor:
                check_anchors(userid, project_id, dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

    # DDP mode -----------------------------------------------------------------
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
                    # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
                    find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules()))
        logger.info('Using DDP()')

    # Model parameters ---------------------------------------------------------
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Save training settings ---------------------------------------------------
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)
    status_update(userid, project_id,
              update_id="hyperparameter",
              update_content=hyp)

    # Start training -----------------------------------------------------------
    t0 = time_synchronized() # time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss_ota = ComputeLossOTA(model)  # init loss class
    compute_loss = ComputeLoss(model)  # init loss class

    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')
    train_start = {}
    train_start['status'] = 'start'
    train_start['epochs'] = epochs
    status_update(userid, project_id,
              update_id="train_start",
              update_content=train_start)

    # torch.save(model, wdir / 'init.pt')
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        t_epoch = time_synchronized()
        model.train()

        # Update image weights (optional)
        if opt.image_weights:
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

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        # logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        # if rank in [-1, 0]:
        #     pbar = tqdm(pbar, total=nb)  # progress bar

        train_loss = {}
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch ---------------------
            t_batch = time_synchronized()
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
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
                status_update(userid, project_id,
                          update_id="hyperparameter",
                          update_content=hyp)

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                if 'loss_ota' not in hyp or hyp['loss_ota'] == 1:
                    loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)  # loss scaled by batch_size
                else:
                    loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

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
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                # pbar.set_description(s)

                mloss_list = mloss.to('cpu').numpy().tolist()
                train_loss['epoch'] = epoch + 1
                train_loss['total_epoch'] = epochs
                train_loss['gpu_mem'] = mem
                train_loss['box'] = mloss_list[0]
                train_loss['obj'] = mloss_list[1]
                train_loss['cls'] = mloss_list[2]
                train_loss['total'] = mloss_list[3]
                train_loss['label'] = targets.shape[0]
                train_loss['step'] = i + 1
                train_loss['total_step'] = nb
                train_loss['time'] = f"{(time_synchronized() - t_batch):.1f} s"
                status_update(userid, project_id,
                              update_id="train_loss",
                              update_content=train_loss)
                # Plot
                # if plots and ni < 10:
                #     f = save_dir / f'train_batch{ni}.jpg'  # filename
                #     Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
                #     if tb_writer:
                #         tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                #         tb_writer.add_graph(torch.jit.trace(model, imgs, strict=False), [])  # add model graph
                # elif plots and ni == 10 and wandb_logger.wandb:
                #     wandb_logger.log({"Mosaics": [wandb_logger.wandb.Image(str(x), caption=x.name) for x in
                #                                   save_dir.glob('train*.jpg') if x.exists()]})
        # end training batches -------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        for i, lr_element in enumerate(lr):
            hyp[f'lr_group{j}'] = lr_element
        status_update(userid, project_id,
                  update_id="hyperparameter",
                  update_content=hyp)
        scheduler.step()

        if rank in [-1, 0]: # (DDP process 0 or single-GPU) test ---------------
            # mAP
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP
                # wandb_logger.current_epoch = epoch + 1
                results, maps, times = test.test(proj_info,
                                                 data_dict,
                                                 batch_size=batch_size, # multiplying by 2 may cause out of gpu memory
                                                 imgsz=imgsz_test,
                                                 model=ema.ema,
                                                 single_cls=opt.single_cls,
                                                 dataloader=testloader,
                                                 save_dir=save_dir,
                                                 verbose=nc < 50 and final_epoch,
                                                 plots=plots and final_epoch,
                                                 # wandb_logger=wandb_logger,
                                                 compute_loss=compute_loss,
                                                 is_coco=is_coco,
                                                 v5_metric=opt.v5_metric)

            # Write
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # append metrics, val_loss
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

            # Log
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5_0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # tensorboard
            #     if wandb_logger.wandb:
            #         wandb_logger.log({tag: x})  # W&B

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            # wandb_logger.end_epoch(best_result=best_fitness == fi)

            # Save model
            if (not opt.nosave) or (final_epoch and not opt.evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': results_file.read_text(),
                        'model': deepcopy(model.module if is_parallel(model) else model).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),}
                        # 'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if not opt.nosave:
                    torch.save(ckpt, last)
                # if (best_fitness == fi) and (epoch >= 200):
                #     torch.save(ckpt, wdir / 'best_{:03d}.pt'.format(epoch))
                # if epoch == 0:
                #     torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                # elif ((epoch+1) % 25) == 0:
                #     torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                # elif epoch >= (epochs-5):
                #     torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                # if wandb_logger.wandb:
                #     if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                #         wandb_logger.log_model(
                #             last.parent, opt, epoch, fi, best_model=best_fitness == fi)
                del ckpt

                # Save epoch internally
                info = Info.objects.get(userid=userid, project_id=project_id)
                info.epoch = epoch
                info.save()

        # end validation -------------------------------------------------------

    # end training -------------------------------------------------------------

    if rank in [-1, 0]:
        # Plots
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png
            # if wandb_logger.wandb:
            #     files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
            #     wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files
            #                                   if (save_dir / f).exists()]})

        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time_synchronized() - t0) / 3600))

        # Test best.pt after fusing layers
        # [tenace's note] argument of type 'PosixPath' is not iterable
        if best.exists():
            m = str(best)
        else:
            m = str(last)
        results, _, _ = test.test(proj_info,
                                  data_dict,
                                  batch_size=batch_size * 2,
                                  imgsz=imgsz_test,
                                  conf_thres=0.001,
                                  iou_thres=0.7,
                                  model=attempt_load(m, device).half(),
                                  single_cls=opt.single_cls,
                                  dataloader=testloader,
                                  save_dir=save_dir,
                                  save_json=False, #True,
                                  plots=False,
                                  is_coco=is_coco,
                                  v5_metric=opt.v5_metric)

        # Strip optimizers
        final = best if best.exists() else last  # final model
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
        if opt.bucket:
            os.system(f'gsutil cp {final} gs://{opt.bucket}/weights')  # upload
        # if wandb_logger.wandb and not opt.evolve:  # Log the stripped model
        #     wandb_logger.wandb.log_artifact(str(final), type='model',
        #                                     name='run_' + wandb_logger.wandb_run.id + '_model',
        #                                     aliases=['last', 'best', 'stripped'])
        # wandb_logger.finish_run()
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    gc.collect()

    mb = os.path.getsize(final) / 1E6  # filesize
    train_end = {}
    train_end['status'] = 'end'
    train_end['epochs'] = epochs
    train_end['bestmodel'] = str(final)
    train_end['bestmodel_size'] = f'{mb:.1f} MB'
    train_end['time'] = f'{(time_synchronized() - t0) / 3600:.3f} hours'
    status_update(userid, project_id,
                  update_id="train_end",
                  update_content=train_end)
    return results, final
