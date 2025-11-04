import gc
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from tango.common.models.yolo import Model
from tango.common.models.supernet_yolov7 import NASModel as NASModelV7
from tango.common.models.supernet_yolov9 import NASModel as NASModelV9
from tango.common.models.resnet_cifar10 import ClassifyModel
from tango.common.models.experimental import attempt_load

from tango.utils.autoanchor import check_anchors
from tango.utils.autobatch import get_batch_size_for_gpu
from tango.utils.django_utils import safe_update_info
from tango.utils.datasets import create_dataloader, create_dataloader_v9
from tango.utils.general import (
    labels_to_class_weights,
    labels_to_class_weights_v9,
    labels_to_image_weights,
    init_seeds,
    init_seeds_v9,
    fitness,
    strip_optimizer,
    check_dataset,
    check_img_size,
    one_cycle,
    colorstr,
)
from tango.utils.loss import (
    ComputeLoss,
    ComputeLossOTA,
    ComputeLossAuxOTA,
    ComputeLossTAL,
    ComputeLoss_v9,
    FocalLossCE,
)
from tango.utils.torch_utils import (
    ModelEMA,
    intersect_dicts,
    torch_distributed_zero_first,
    is_parallel,
    de_parallel,
)

from . import test


logger = logging.getLogger(__name__)


def finetune(proj_info, subnet, hyp, opt, data_dict, device, tb_writer=None):
    # Options ------------------------------------------------------------------
    save_dir, epochs, total_batch_size, world_size, rank, local_rank = \
        Path(opt.save_dir), opt.finetune_epochs, opt.batch_size, \
        opt.world_size, opt.global_rank, opt.local_rank

    userid, project_id = proj_info['userid'], proj_info['project_id']
    is_v9 = (opt.loss_name == 'TAL')

    # Directories --------------------------------------------------------------
    wdir = save_dir / 'nas' / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    d_str = '_'
    for d in subnet.yaml['depth_list']:
        d_str += str(d)
    search_best = wdir / f'subnet{d_str}.pt'

    # Configure ----------------------------------------------------------------
    cuda = device.type != 'cpu'
    seed = opt.seed if opt.seed else 0
    if is_v9:
        init_seeds_v9(seed + 1 + rank, deterministic=True) # from yolov9
    else:
        init_seeds(2 + rank)
    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check
    
    # Model --------------------------------------------------------------------
    subnet.to(device)

    # Image sizes --------------------------------------------------------------
    gs = max(int(subnet.stride.max()), 32)  # grid size (max stride)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # Batch size ---------------------------------------------------------------
    batch_size = total_batch_size // max(world_size, 1)

    # SyncBatchNorm ------------------------------------------------------------
    if opt.sync_bn and cuda and rank != -1:
        subnet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(subnet).to(device)
        logger.info('Using SyncBatchNorm()')

    # Dataset ------------------------------------------------------------------
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    train_path = data_dict['train']
    test_path = data_dict['val']
    is_coco = True if data_dict['dataset_name'] == 'coco' and 'coco' in train_path else False

    # Optimizer ----------------------------------------------------------------
    nbs = 96  # nominal batch size
    accumulate = max(round(nbs / max(batch_size, 1)), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
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
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA ----------------------------------------------------------------------
    ema = ModelEMA(subnet) if rank in [-1, 0] else None
    logger.info('Using ModelEMA()')
    
    # DP mode ------------------------------------------------------------------
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        subnet = torch.nn.DataParallel(subnet)
        logger.info('Using DataParallel(): not recommended, use DDP instead.')
        
    # Trainloader --------------------------------------------------------------
    if is_v9: # v9
        dataloader, dataset = create_dataloader_v9(
            train_path,
            imgsz,
            batch_size, # Single or DP -> total, DDP -> per-GPU
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
    else:
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
            rank=rank, # local_rank?
            world_size=opt.world_size,
            workers=opt.workers,
            image_weights=opt.image_weights,
            quad=opt.quad,
            prefix='train'
        )
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Process 0 (TestDataLoader) -----------------------------------------------
    if rank in [-1, 0]:
        if is_v9: #v9
            testloader = create_dataloader_v9(
                test_path,
                imgsz,
                batch_size * 2, # * 2 may lead out-of-memory
                gs,
                opt.single_cls,
                hyp=hyp,
                cache=opt.cache_images and not opt.notest,
                rect=True,
                rank=-1,
                workers=opt.workers * 2,
                pad=0.5,
                close_mosaic=True, # always use torch.utils.data.Dataloader
                prefix='val',
                shuffle=False,
                uid=userid,
                pid=project_id,
            )[0]
        else:
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
                prefix='val'
            )[0]

        labels = np.concatenate(dataset.labels, 0)
        c = torch.tensor(labels[:, 0])  # classes
        # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
        # model._initialize_biases(cf.to(device))
                
        # Anchors
        if not opt.noautoanchor:
            check_anchors(userid, project_id, dataset, model=subnet,
                          thr=hyp['anchor_t'], imgsz=imgsz)
        subnet.half().float()  # pre-reduce anchor precision
        
    # DDP mode -----------------------------------------------------------------
    if cuda and rank != -1:
        subnet = DDP(subnet, device_ids=[opt.local_rank], 
                     output_device=opt.local_rank, 
                     find_unused_parameters=True)

    # Model parameters ---------------------------------------------------------
    base = de_parallel(subnet)
    if is_v9: # v9
        class_weights = labels_to_class_weights_v9(dataset.labels, nc).to(device) * nc  # attach class weights
    else: # v7
        logger.info(f'loss = {opt.loss_name}')
        nl = base.model[-1].nl
        hyp['box'] *= 3. / nl  # scale to layers
        hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
        hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
        class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
        gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
        base.gr = subnet.gr = gr
    hyp['label_smoothing'] = opt.label_smoothing

    base.nc = subnet.nc = nc  # attach number of classes to model
    base.hyp = subnet.hyp = hyp  # attach hyperparameters to model
    base.class_weights = subnet.class_weights = class_weights
    base.names = subnet.names = names
    
    # loss function ------------------------------------------------------------
    if opt.loss_name == 'TAL':
        compute_loss = ComputeLoss_v9(subnet)
    else:
        if opt.loss_name == 'OTA':
            compute_loss_ota = ComputeLossOTA(subnet)  # init loss class
        elif opt.loss_name == 'AuxOTA':
            compute_loss_ota = ComputeLossAuxOTA(subnet)  # init loss class
        else:
            compute_loss_ota = None
        compute_loss = ComputeLoss(subnet)  # init loss class

    # Start fine tuning --------------------------------------------------------

    start_epoch, best_fitness = 0, 0.0

    nb = len(dataloader)  # number of batches == steps per epoch
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training

    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)

    scheduler.last_epoch = start_epoch - 1  # do not move

    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'       Starting training for {epochs} epochs...\n'
                f'       Warming up for the first {nw} iters({nw//nb} epochs)')
    
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
    
        # Stop mosaic augmentation
        if is_v9 and epoch == (epochs - opt.close_mosaic):
            logger.info("Closing dataloader mosaic")
            dataset.mosaic = False
    
        # mean loss
        if is_v9:
            mloss = torch.zeros(3, device=device)
        else:
            mloss = torch.zeros(4, device=device)  # mean losses

        # distribute data
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)

        # progress bar
        pbar = enumerate(dataloader)
        if is_v9:
            title_s = ('\n' + '%11s' * 7) % (
                'Epoch', 'GPU_Mem', 'box', 'cls', 'dfl', 'Labels', 'Img_Size'
            )
        else: # v7
            title_s = ('\n' + '%11s' * 8) % (
                'Epoch', 'GPU_Mem', 'Box', 'Obj', 'Cls', 'Total', 'Labels', 'Img_Size'
            )

        optimizer.zero_grad()
        # finetuing batches start ==============================================
        for i, (imgs, targets, _, _) in pbar:
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
    
            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / max(batch_size, 1)]).round())
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
            pred = subnet(imgs)  # forward
            if opt.loss_name == "OTA":
                loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)  # loss scaled by batch_size
            else:
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
            if rank != -1:
                loss *= opt.world_size  # gradient averaged between devices in DDP mode
            if opt.quad:
                loss *= 4.    
            
            # Backward
            loss.requires_grad_(True)
            loss.backward()

            # Optimize
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                if ema:
                    ema.update(de_parallel(subnet))
                    
            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                if is_v9:
                    s = ('%11s' * 2 + '%11.4g' * 5) % (
                        '%g/%g' % (epoch, epochs - 1), mem,         # epoch/total, gpu_mem
                        *mloss, targets.shape[0], imgs.shape[-1]    # box, dfl, cls, labels, imgsz
                    )
                else:
                    s = ('%10s' * 2 + '%11.4f' * 4 + '%11.0f' * 2) % (
                        '%g/%g' % (epoch, epochs - 1), mem,         # epoch/total, gpu_mem
                        *mloss, targets.shape[0], imgs.shape[-1]    # box, obj, cls, total, labels, imgsz
                    )                
                ten_percent_cnt = int((i+1)/nb*10+0.5)
                bar = '|'+ '#'*ten_percent_cnt + ' '*(10-ten_percent_cnt)+'|'
                content_s = s + (f'{bar}{(i+1)/nb*100:3.0f}% {i+1:4.0f}/{nb:4.0f}')
                if (i % 50) == 0:
                    logger.info(title_s)
                logger.info(content_s)
        # finetuning batches end ===============================================
        
        # Scheduler
        scheduler.step()
        
        # DDP process 0 or single-GPU test =====================================
        if rank in [-1, 0]:
            # mAP
            if is_v9:
                ema.update_attr(de_parallel(subnet),
                        include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            else:
                ema.update_attr(de_parallel(subnet),
                        include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            
            final_epoch = (epoch + 1 == epochs)

            results = None
            if not opt.notest or final_epoch:  # Calculate mAP
                results, maps, _ =  test.test(
                    proj_info,
                    data_dict,
                    batch_size=opt.batch_size * 2,
                    imgsz=imgsz_test,
                    model=ema.ema,
                    conf_thres=0.001,
                    iou_thres=0.7,
                    single_cls=opt.single_cls,
                    dataloader=testloader,
                    verbose=nc < 50 and final_epoch,
                    plots=False,
                    half_precision = False,
                    compute_loss=None,
                    save_json=False,
                    is_coco=is_coco,
                    metric=opt.metric
                )
            # Update best mAP
            fi = None
            if results is not None:
                fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        
            if not opt.nosave or final_epoch:
                if fi is not None and fi > best_fitness:
                    ckpt = {'epoch': epoch,
                            'best_fitness': best_fitness,
                            'model': deepcopy(de_parallel(subnet)).half(),
                            'ema': deepcopy(ema.ema).half(),
                            'updates': ema.updates,
                            'optimizer': optimizer.state_dict()}
                    torch.save(ckpt, search_best)
                    del ckpt

                    mb = os.path.getsize(search_best) / 1E6 # file size
                    logger.info(f"epoch {epoch} : {search_best} {mb:.1f} MB")

                    best_fitness = float(fi)

        # DDP Sync
        if dist.is_available() and dist.is_initialized(): 
            dist.barrier()
        # end validation =======================================================
    
    # end finetuning -----------------------------------------------------------

    subnet_pt = search_best if search_best.exists() else None

    if rank in [-1, 0]:
        if not search_best.exists():
            logger.warning(
                f'{colorstr("NAS: ")}No best subnet checkpoint saved; writing current weights to {search_best}'
            )
            ckpt = {
                'epoch': epochs,
                'best_fitness': best_fitness,
                'model': deepcopy(de_parallel(subnet)).half(),
                'ema': deepcopy(ema.ema).half() if ema else None,
                'updates': ema.updates if ema else None,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(ckpt, search_best)
            del ckpt
            subnet_pt = search_best

    # strip optimizer ----------------------------------------------------------
    # [TENACE] what does strip_optimizer do:
    # 1. load .pt
    # 2. 'model' <- 'ema'
    # 3. 'ema', 'optimizer', 'training_results', 'updates' -> None
    # 4. 'epoch' -> -1
    # 5. model.half()   *** removed by tenace ***
    # 6. requires_grad -> False
    # 7. save .pt
    # if subnet_pt:
    #     strip_optimizer(subnet_pt)

    # cleanup ------------------------------------------------------------------
    del ema
    del optimizer
    del subnet
    torch.cuda.empty_cache()
    gc.collect()
    if dist.is_available() and dist.is_initialized(): 
        dist.destroy_process_group()

    return str(subnet_pt) if subnet_pt else None, results


def finetune_hyp(proj_info, basemodel, hyp, opt, data_dict, device, tb_writer=None):
    # Options ------------------------------------------------------------------
    save_dir, epochs, total_batch_size, world_size, rank, local_rank, gen = \
        Path(opt.save_dir), opt.finetune_epochs, opt.batch_size, \
        opt.world_size, opt.global_rank, opt.local_rank, opt.gen
    
    userid, project_id, task, lt, nas, hpo, target, target_acc = \
        proj_info['userid'], proj_info['project_id'], proj_info['task_type'], proj_info['learning_type'], \
        proj_info['nas'], proj_info['hpo'], proj_info['target_info'], proj_info['acc']
    assert hpo and lt == 'HPO', f'HPO: project information mismatch. Learning type {lt}, HPO ? {hpo}'
    is_v9 = (opt.loss_name == 'TAL')

    # Directories --------------------------------------------------------------
    wdir = save_dir / 'hpo' / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    search_best = wdir / f'hyp_{gen:04d}.pt'

    # Device -------------------------------------------------------------------
    # device = select_device(opt.device)

    # Configure ----------------------------------------------------------------
    cuda = device.type != 'cpu'
    seed = opt.seed if opt.seed else 0
    if is_v9:
        init_seeds_v9(seed + 1 + rank, deterministic=True) # from yolov9
    else:
        init_seeds(2 + rank)
    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    # ch = int(data_dict.get('ch', 3))
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Model --------------------------------------------------------------------
    if basemodel:
        opt.weights = str(basemodel)
    net = attempt_load(opt.weights, map_location=device, fused=False) # make sure it is not a fused model
    # ckpt = torch.load(opt.weights, map_location='cpu')  # load checkpoint
    # exclude = []
    # if task == 'classification':
    #     net = ClassifyModel(opt.cfg or ckpt['model'].yaml, ch=ch, nc=nc)
    # elif task == 'detection':
    #     if nas or target == 'Galaxy_S22':
    #         if is_v9:
    #             net = NASModelV9(opt.cfg or ckpt['model'].yaml, ch=ch, nc=nc, anchors=hyp.get('anchors'))
    #         else:
    #             net = NASModelV7(opt.cfg or ckpt['model'].yaml, ch=ch, nc=nc, anchors=hyp.get('anchors'))
    #     else:
    #         net = Model(opt.cfg or ckpt['model'].yaml, ch=ch, nc=nc, anchors=hyp.get('anchors'))
    #     exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys

    # logger.info(f'HPO: Loading and overwrite weights from the basemodel...')
    # state_dict = ckpt['model'].float().state_dict()  # to FP32
    # state_dict = intersect_dicts(state_dict, net.state_dict(), exclude=exclude)  # intersect
    # net.load_state_dict(state_dict, strict=False)  # load
    # net.to(device)
    # logger.info('HPO: Transferred %g/%g items from %s' % (len(state_dict), len(net.state_dict()), opt.weights))  # report

    # Image sizes --------------------------------------------------------------
    gs = max(int(net.stride.max()), 32)  # grid size (max stride)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # Batch size ---------------------------------------------------------------
    batch_size = total_batch_size // max(world_size, 1)

    # SyncBatchNorm ------------------------------------------------------------
    if opt.sync_bn and cuda and rank != -1:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).to(device)
        logger.info('Using SyncBatchNorm()')

    # Dataset ------------------------------------------------------------------
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    train_path = data_dict['train']
    test_path = data_dict['val']
    is_coco = True if data_dict['dataset_name'] == 'coco' and 'coco' in train_path else False

    # Optimizer ----------------------------------------------------------------
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / max(batch_size, 1)), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in net.named_modules():
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
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA ----------------------------------------------------------------------
    ema = ModelEMA(net) if rank in [-1, 0] else None
    logger.info('Using ModelEMA()')
    
    # DP mode ------------------------------------------------------------------
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
        logger.info('Using DataParallel(): not recommended, use DDP instead.')
        
    # Trainloader --------------------------------------------------------------
    if is_v9: # v9
        dataloader, dataset = create_dataloader_v9(
            train_path,
            imgsz,
            batch_size, # Single or DP -> total, DDP -> per-GPU
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
    else:
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
            prefix='train'
        )
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Process 0 (TestDataLoader) -----------------------------------------------
    if rank in [-1, 0]:
        if is_v9: # v9
            testloader = create_dataloader_v9(
                test_path,
                imgsz,
                batch_size * 2, # * 2 may lead out-of-memory
                gs,
                opt.single_cls,
                hyp=hyp,
                cache=opt.cache_images and not opt.notest,
                rect=True,
                rank=-1,
                workers=opt.workers * 2,
                pad=0.5,
                close_mosaic=True, # always use torch.utils.data.Dataloader
                prefix='val',
                shuffle=False,
                uid=userid,
                pid=project_id,
            )[0]
        else:
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
                prefix='val'
            )[0]

        labels = np.concatenate(dataset.labels, 0)
        c = torch.tensor(labels[:, 0])  # classes
        # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
        # model._initialize_biases(cf.to(device))
                
        # Anchors
        if not opt.noautoanchor:
            check_anchors(userid, project_id, dataset, model=net, 
                          thr=hyp['anchor_t'], imgsz=imgsz)
        net.half().float()  # pre-reduce anchor precision
        
    # DDP mode -----------------------------------------------------------------
    if cuda and rank != -1:
        net = DDP(net, device_ids=[opt.local_rank], 
                  output_device=opt.local_rank,
                  find_unused_parameters=True)

    # Model parameters ---------------------------------------------------------
    base = de_parallel(net)
    if is_v9: # v9
        class_weights = labels_to_class_weights_v9(dataset.labels, nc).to(device) * nc  # attach class weights
    else: # v7
        logger.info(f'loss = {opt.loss_name}')
        nl = base.model[-1].nl
        hyp['box'] *= 3. / nl  # scale to layers
        hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
        hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
        class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
        gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
        base.gr = net.gr = gr
    hyp['label_smoothing'] = opt.label_smoothing

    base.nc = net.nc = nc  # attach number of classes to model
    base.hyp = net.hyp = hyp  # attach hyperparameters to model
    base.class_weights = net.class_weights = class_weights
    base.names = net.names = names
    
    # loss function ------------------------------------------------------------
    if task == 'classification':
        if opt.loss_name == 'CE':
            compute_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        elif opt.loss_name == 'FL':
            compute_loss = FocalLossCE()
        else:
            logger.warning(f'not supported loss function {opt.loss_name}')
    else: # if task == 'detection':
        if opt.loss_name == 'TAL':
            compute_loss = ComputeLossTAL(net)
        else:
            if opt.loss_name == 'OTA':
                compute_loss_ota = ComputeLossOTA(net)  # init loss class
            elif opt.loss_name == 'AuxOTA':
                compute_loss_ota = ComputeLossAuxOTA(net)  # init loss class
            else:
                compute_loss_ota = None
            compute_loss = ComputeLoss(net)  # init loss class

    # Start fine tuning --------------------------------------------------------

    start_epoch, best_fitness = 0, 0.0
    t0 = time.time()
    nb = len(dataloader)  # number of batches == steps per epoch
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training

    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)

    scheduler.last_epoch = start_epoch - 1  # do not move

    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'       Starting training for {epochs} epochs...\n'
                f'       Warming up for the first {nw} iters({nw//nb} epochs)')

    for epoch in range(start_epoch, epochs):
        net.train()

        # Update image weights (optional)
        if opt.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                cw = net.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()
    
        # Stop mosaic augmentation
        if is_v9 and epoch == (epochs - opt.close_mosaic):
            logger.info("Closing dataloader mosaic")
            dataset.mosaic = False
    
        # mean loss
        if is_v9:
            mloss = torch.zeros(3, device=device)
        else:
            mloss = torch.zeros(4, device=device)  # mean losses

        # distribute data
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)

        # progress bar
        pbar = enumerate(dataloader)
        if is_v9:
            title_s = ('\n' + '%11s' * 7) % (
                'Epoch', 'GPU_Mem', 'box', 'cls', 'dfl', 'Labels', 'Img_Size'
            )
        else: # v7
            title_s = ('\n' + '%11s' * 8) % (
                'Epoch', 'GPU_Mem', 'Box', 'Obj', 'Cls', 'Total', 'Labels', 'Img_Size'
            )

        optimizer.zero_grad()
        # finetuing batches start ==============================================
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
    
            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / max(batch_size, 1)]).round())
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
            # optimizer.zero_grad()
            pred = net(imgs)  # forward
            # if 'loss_ota' not in hyp or hyp['loss_ota'] == 1:
            if 'OTA' in opt.loss_name: #opt.loss_name == 'OTA':
                loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)  # loss scaled by batch_size
            else:
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
            if rank != -1:
                loss *= opt.world_size  # gradient averaged between devices in DDP mode
            if opt.quad:
                loss *= 4.    
            
            # Backward
            loss.requires_grad_(True)
            loss.backward()

            # Optimize
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                if ema:
                    ema.update(de_parallel(net))
                    
            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                if is_v9:
                    s = ('%11s' * 2 + '%11.4g' * 5) % (
                        '%g/%g' % (epoch, epochs - 1), mem,         # epoch/total, gpu_mem
                        *mloss, targets.shape[0], imgs.shape[-1]    # box, dfl, cls, labels, imgsz
                    )
                else:
                    s = ('%10s' * 2 + '%11.4f' * 4 + '%11.0f' * 2) % (
                        '%g/%g' % (epoch, epochs - 1), mem,         # epoch/total, gpu_mem
                        *mloss, targets.shape[0], imgs.shape[-1]    # box, obj, cls, total, labels, imgsz
                    )
                ten_percent_cnt = int((i+1)/nb*10+0.5)
                bar = '|'+ '#'*ten_percent_cnt + ' '*(10-ten_percent_cnt)+'|'
                content_s = s + (f'{bar}{(i+1)/nb*100:3.0f}% {i+1:4.0f}/{nb:4.0f}')
                if (i % 50) == 0:
                    logger.info(title_s)
                logger.info(content_s)
        # finetuning batches end ===============================================
        
        # Scheduler
        scheduler.step()
        
        # DDP process 0 or single-GPU test =====================================
        if rank in [-1, 0]:
            # mAP
            if is_v9:
                ema.update_attr(de_parallel(net),
                        include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            else:
                ema.update_attr(de_parallel(net),
                        include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])

            final_epoch = (epoch + 1 == epochs)

            results = None
            if not opt.notest or final_epoch:  # Calculate mAP
                results, maps, times =  test.test(
                    proj_info,
                    data_dict,
                    batch_size=opt.batch_size, # * 2,
                    imgsz=imgsz_test,
                    model=ema.ema,
                    conf_thres=0.001,
                    iou_thres=0.7,
                    single_cls=opt.single_cls,
                    dataloader=testloader,
                    verbose=nc < 50 and final_epoch,
                    plots=False,
                    half_precision=False,
                    compute_loss=None,
                    save_json=False,
                    is_coco=is_coco,
                    metric=opt.metric
                )
            # Update best mAP
            fi = None
            if results is not None:
                fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]

            if not opt.nosave or final_epoch:
                if fi is not None and fi > best_fitness:
                    ckpt = {'epoch': epoch,
                            'best_fitness': best_fitness,
                            'model': deepcopy(de_parallel(net)).half(),
                            'ema': deepcopy(ema.ema).half(),
                            'updates': ema.updates,
                            'optimizer': optimizer.state_dict()}
                    torch.save(ckpt, search_best)
                    del ckpt

                    mb = os.path.getsize(search_best) / 1E6 # file size
                    logger.info(f"epoch {epoch} : {search_best} {mb:.1f} MB")

                    best_fitness = float(fi)

        # DDP Sync
        if dist.is_available() and dist.is_initialized(): 
            dist.barrier()
        # end validation =======================================================

    # end finetuning -----------------------------------------------------------

    # net_pt = search_best if search_best.exists() else None

    # strip optimizer ----------------------------------------------------------
    # [TENACE] what does strip_optimizer do:
    # 1. load .pt
    # 2. 'model' <- 'ema'
    # 3. 'ema', 'optimizer', 'training_results', 'updates' -> None
    # 4. 'epoch' -> -1
    # 5. model.half()   *** removed by tenace ***
    # 6. requires_grad -> False
    # 7. save .pt
    # if net_pt:
    #     strip_optimizer(net_pt)

    # cleanup ------------------------------------------------------------------
    del ema
    del optimizer
    del net
    torch.cuda.empty_cache()
    gc.collect()
    if dist.is_available() and dist.is_initialized(): 
        dist.destroy_process_group()

    return results
