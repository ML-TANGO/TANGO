'''
[TEANCE] TODO: since this finetune.py is almost the same to trian.py,
         it should be merged by one file.
         - similarities and differences -
         =======================================================================
         procedure    |         train           |         finetune
         -----------------------------------------------------------------------
         options      |         same            |           same
         directories  |  last.pt, best.pt       |       subnet_bbbbhhhh.pt
                      |     results.txt         |             X
         device       |         same            |           same
         configure    |         same            |           same
         model        |   from parsing yaml     |    from input argument
         freeze       |           O             |             X
         img size     |         same            |           same
         batch size   |      autobatch          |    from input argument
         dataset chk  |         same            |           same
         optimizer    |         same            |           same
         scheduler    |         same            |           same
         ema          |         same            |           same
         resume       |           O             |             X
         dp mode      |         same            |           same
         sync bn      |         same            |           same
         dataloader   |         same            |           same
         testloader   |         same            |           same
         ddp mode     |         same            |           same
         model param  |      update hyp         |         fixed hyp
         save args    |           O             |             X
         loss func    |         same            |           same
         amp          |          use            |         not use
         train        |         same            |           same
         test         |  conf=0.001, iou=0.6    |    conf=0.001, iou=0.7
         save best    |  incl. train results    |    w/o train results
         test fused   |  conf=0.001, iou=0.7    |             X
         strip optim. |           O             |             O
         =======================================================================
'''

import logging
import math
import os
import gc
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread

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
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from . import status_update, Info
from . import test
from tango.common.models.experimental import attempt_load
from tango.common.models.yolo               import Model
from tango.common.models.supernet_yolov7    import NASModel
# from tango.common.models import *
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
from tango.utils.loss import ComputeLoss, ComputeLossOTA
from tango.utils.plots import   (   plot_images,
                                    plot_labels,
                                    plot_results,
                                    plot_evolution
                                )
from tango.utils.torch_utils import (   ModelEMA,
                                        select_device,
                                        intersect_dicts,
                                        torch_distributed_zero_first,
                                        is_parallel
                                    )


logger = logging.getLogger(__name__)


def finetune(proj_info, subnet, hyp, opt, data_dict, tb_writer=None):
    # Options ------------------------------------------------------------------
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.fintune_epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

    userid, project_id, task, nas, hpo, target, target_acc = \
        proj_info['userid'], proj_info['project_id'], proj_info['task_type'], \
        proj_info['nas'], proj_info['hpo'], proj_info['target_info'], proj_info['acc']

    info = Info.objects.get(userid=userid, project_id=project_id)

    # Directories --------------------------------------------------------------
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    d_str = '_'
    for d in subnet.yaml['depth_list']:
        d_str += str(d)
    search_best = wdir / f'subnet{d_str}.pt'

    # Device -------------------------------------------------------------------
    device = select_device(opt.device)

    # Configure ----------------------------------------------------------------
    # plots = not opt.evolve  # create plots
    plots = False  # create plots
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)

    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    ch = int(data_dict.get('ch', 3))
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check
    
    # Model --------------------------------------------------------------------
    subnet.to(device)

    # import pprint
    # pprint.pprint(torch.cuda.memory_stats(device=device))

    # Image sizes --------------------------------------------------------------
    gs = max(int(subnet.stride.max()), 32)  # grid size (max stride)
    nl = subnet.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # Batch size ---------------------------------------------------------------
    autobatch_rst = get_batch_size_for_gpu( userid,
                                            project_id,
                                            subnet,
                                            ch,
                                            imgsz,
                                            amp_enabled=False,
                                            max_search=False )
    print(f"autobatch result = {autobatch_rst}, supernet_batchsize = {batch_size}")

    # Dataset ------------------------------------------------------------------
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    train_path = data_dict['train']
    test_path = data_dict['val']
    is_coco = True if data_dict['dataset_name'] == 'coco' and 'coco' in train_path else False

    # Optimizer ----------------------------------------------------------------
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in subnet.named_modules():
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
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA ----------------------------------------------------------------------
    ema = ModelEMA(subnet) if rank in [-1, 0] else None
    logger.info('Using ModelEMA()')
    
    # DP mode ------------------------------------------------------------------
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        subnet = torch.nn.DataParallel(subnet)
        logger.info('Using DataParallel()')

    # SyncBatchNorm ------------------------------------------------------------
    if opt.sync_bn and cuda and rank != -1:
        subnet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(subnet).to(device)
        logger.info('Using SyncBatchNorm()')
        
    # Trainloader --------------------------------------------------------------
    dataloader, dataset = create_dataloader(
                                        userid,
                                        project_id,
                                        train_path,
                                        imgsz,
                                        batch_size,
                                        gs,
                                        opt,
                                        hyp=hyp,
                                        augment=True,
                                        cache=opt.cache_images,
                                        rect=opt.rect,
                                        rank=rank,
                                        world_size=opt.world_size,
                                        workers=opt.workers,
                                        image_weights=opt.image_weights,
                                        quad=opt.quad,
                                        prefix='train')
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Process 0 (TestDataLoader) -----------------------------------------------
    if rank in [-1, 0]:
        testloader = create_dataloader(
                                    userid,
                                    project_id,
                                    test_path,
                                    imgsz_test,
                                    batch_size, # * 2,
                                    gs,
                                    opt,
                                    hyp=hyp,
                                    cache=opt.cache_images and not opt.notest,
                                    rect=True,
                                    rank=-1,
                                    world_size=opt.world_size,
                                    workers=opt.workers,
                                    pad=0.5,
                                    prefix='val')[0]

        labels = np.concatenate(dataset.labels, 0)
        c = torch.tensor(labels[:, 0])  # classes
        # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
        # model._initialize_biases(cf.to(device))
        # if plots:
        #     plot_labels(labels, names, save_dir, loggers)
        #     if tb_writer:
        #         tb_writer.add_histogram('classes', c, 0)
                
        # Anchors
        if not opt.noautoanchor:
            check_anchors(userid, project_id, dataset, model=subnet, thr=hyp['anchor_t'], imgsz=imgsz)
        subnet.half().float()  # pre-reduce anchor precision
        
    # DDP mode -----------------------------------------------------------------
    if cuda and rank != -1:
        subnet = DDP(subnet, device_ids=[opt.local_rank], output_device=opt.local_rank,
                    # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
                    find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in subnet.modules()))

    # Model parameters ---------------------------------------------------------
    # hyp['box'] *= 3. / nl  # scale to layers
    # hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    # hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    # hyp['label_smoothing'] = opt.label_smoothing
    subnet.nc = nc  # attach number of classes to model
    subnet.hyp = hyp  # attach hyperparameters to model
    subnet.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    subnet.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    subnet.names = names
    
    # loss function ------------------------------------------------------------
    compute_loss_ota = ComputeLossOTA(subnet)  # init loss class
    compute_loss = ComputeLoss(subnet)  # init loss class

    # Start fine tuning --------------------------------------------------------

    start_epoch, best_fitness = 0, 0.0
    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    # scaler = amp.GradScaler(enabled=cuda)

    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                # f'Logging results to {save_dir}\n'
                f'Starting finetuing for {epochs} epochs...')
    # torch.save(model, wdir / 'init.pt')
    for epoch in range(start_epoch, epochs):
        subnet.train()

        # Update image weights (optional)
        if opt.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                cw = subnet.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
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
    
        # mean loss
        mloss = torch.zeros(4, device=device)  # mean losses

        # distribute data
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)

        # progress bar
        pbar = enumerate(dataloader)
        # logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        # if rank in [-1, 0]:
        #     pbar = tqdm(pbar, total=nb)  # progress bar

        # optimizer.zero_grad()
        # finetuing batches start ==============================================
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
    
            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])
            
            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)     
                    
            # Forward
            # with amp.autocast(enabled=cuda):
            pred = subnet(imgs)  # forward
            if 'loss_ota' not in hyp or hyp['loss_ota'] == 1:
                loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)  # loss scaled by batch_size
            else:
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
            if rank != -1:
                loss *= opt.world_size  # gradient averaged between devices in DDP mode
            if opt.quad:
                loss *= 4.    
            
            # Backward
            # scaler.scale(loss).backward()
            loss.requires_grad_(True)
            loss.backward()

            # Optimize
            if ni % accumulate == 0:
                # scaler.step(optimizer)  # optimizer.step
                # scaler.update()
                optimizer.step()
                optimizer.zero_grad()
                if ema:
                    ema.update(subnet)
                    
            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                # pbar.set_description(s)
        # finetuning batches end ===============================================
        
        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()
        
        # DDP process 0 or single-GPU test =====================================
        if rank in [-1, 0]:
            # mAP
            ema.update_attr(subnet, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP
                # wandb_logger.current_epoch = epoch + 1
                results, maps, times =  test.test(proj_info,
                                                  data_dict,
                                                  batch_size=opt.batch_size, # * 2,
                                                  imgsz=imgsz_test,
                                                  model=ema.ema,
                                                  conf_thres=0.001,
                                                  iou_thres=0.7,
                                                  single_cls=opt.single_cls,
                                                  dataloader=testloader,
                                                  verbose=nc < 50 and final_epoch,
                                                  # save_dir=save_dir,
                                                  save_json=False,
                                                  plots=False,
                                                  is_coco=is_coco,
                                                  v5_metric=opt.v5_metric)
            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
                
            if best_fitness == fi:
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        # 'training_results': results_file.read_text(),
                        'model': deepcopy(subnet.module if is_parallel(subnet) else subnet).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict()}
                torch.save(ckpt, search_best)
                del ckpt
        # end validation =======================================================

    # end finetuning -----------------------------------------------------------
    if not rank in [-1, 0]:
        dist.destroy_process_group()

    # delete Model -------------------------------------------------------------
    del subnet
    torch.cuda.empty_cache()
    gc.collect()

    # strip optimizer ----------------------------------------------------------
    # [TENACE] what does strip_optimizer do:
    # 1. load .pt
    # 2. 'model' <- 'ema'
    # 3. 'ema', 'optimizer', 'training_results', 'wandb_id', 'updates' -> None
    # 4. 'epoch' -> -1
    # 5. model.half()   *** removed by tenace ***
    # 6. requires_grad -> False
    # 7. save .pt
    subnet_pt = search_best if search_best.exists() else None
    if subnet_pt:
        strip_optimizer(subnet_pt)

    return str(subnet_pt), results
