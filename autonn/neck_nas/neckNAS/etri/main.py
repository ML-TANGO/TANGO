""" autonn entry point

    MIT License
    Copyright (c) 2022 Hyunwoo Cho
"""
# python-embedded packages
import json
# import logging
import os
import sys
import argparse
from pathlib import Path
import yaml
import numpy as np

# pytorch
import torch
import torch.distributed as dist
from torch.optim import SGD, Adam, AdamW, lr_scheduler

# nni
import nni.retiarii.nn.pytorch as nn
from nni.retiarii.fixed import fixed_arch

# yolov5
from yolov5_utils.general import (LOGGER, print_args, get_latest_run,
                                  check_suffix, check_file, check_yaml,
                                  check_dataset, increment_path,
                                  check_img_size, colorstr, intersect_dicts,
                                  labels_to_class_weights, one_cycle, linear)
from yolov5_utils.loss import ComputeLoss
from yolov5_utils.autoanchor import check_anchors
from yolov5_utils.torch_utils import (select_device,
                                      torch_distributed_zero_first)
from yolov5_utils.plots import plot_lr_scheduler

FILE = Path(__file__).resolve()  # absolute file path
ROOT = FILE.parents[0]  # absolute directory path
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# https://pytorch.org/docs/stable/elastic/run.html
LOCAL_RANK = int(os.getenv('LOCAL_RANK', '-1'))
RANK = int(os.getenv('RANK', '-1'))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', '1'))


def parse_args():
    parser = argparse.ArgumentParser("autonn_necknas")
    # read the base backbone model
    parser.add_argument(
        '--cfg', type=str,
        default='yaml/basemodel.yaml', help='basemodel.yaml path'
    )

    # read the target device information
    parser.add_argument(
        '--target', type=str,
        default='yaml/target.yaml', help='target.yaml path'
    )

    # read the dataset information
    parser.add_argument(
        '--dataset', type=str,
        default='yaml/coco128.yaml', help='dataset.yaml path'
    )

    # configurations for training mode
    parser.add_argument(
        "--train_mode", type=str,
        default='search', choices=['search', 'retrain']
    )
    parser.add_argument(
        "--nas_type", type=str,
        default='ConcatBasedNet',
        choices=['ConcatBasedNet', 'ConcatEntirePath', 'Yolov5Trainer']
    )
    parser.add_argument(
        "--search_type", type=str,
        default='two_stage',
        choices=['one_stage', 'two_stage']
    )
    parser.add_argument(
        '--neck_cfg', type=str,
        default='yaml/neck.yaml', help='neck.yaml'
    )

    # training parameters (ref. yolov6 train.py)
    parser.add_argument(
        '--weights', type=str,
        default='', help='initial weights path'
        # default=ROOT / 'yolov5s.pt', help='initial weights path'
    )
    parser.add_argument(
        '--hyp', type=str,
        default=ROOT / 'yaml/hyp.scratch-low.yaml', help='hyperparameters path'
    )
    parser.add_argument(
        '--epochs', type=int,
        default=300
    )
    parser.add_argument(
        '--batch-size', type=int,
        default=16, help='total batch size for all GPUs, -1 for autobatch'
    )
    parser.add_argument(
        '--imgsz', '--img', '--img-size', type=int,
        default=640, help='train, val image size (pixels)'
    )
    parser.add_argument(
        '--rect',
        action='store_true', help='rectangular training'
    )
    parser.add_argument(
        '--resume', nargs='?', const=True,
        default=False, help='resume most recent training'
    )
    parser.add_argument(
        '--nosave',
        action='store_true', help='only save final checkpoint'
    )
    parser.add_argument(
        '--noval',
        action='store_true', help='only validate final epoch'
    )
    parser.add_argument(
        '--noautoanchor',
        action='store_true', help='disable AutoAnchor'
    )
    parser.add_argument(
        '--noplots',
        action='store_true', help='save no plot files'
    )
    parser.add_argument(
        '--evolve', type=int, nargs='?',
        const=300, help='evolve hyperparameters for x generations'
    )
    parser.add_argument(
        '--cache', type=str, nargs='?',
        const='ram', help='--cache images in "ram" (default) or "disk"'
    )
    parser.add_argument(
        '--image-weights',
        action='store_true', help='use weighted image selection for training'
    )
    parser.add_argument(
        '--device',
        default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu'
    )
    parser.add_argument(
        '--multi-scale',
        action='store_true', help='vary img-size +/- 50%%'
    )
    parser.add_argument(
        '--single-cls',
        action='store_true', help='train multi-class data as single-class'
    )
    parser.add_argument(
        '--optimizer', type=str,
        choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer'
    )
    parser.add_argument(
        '--sync-bn',
        action='store_true', help='use SyncBatchNorm, available in DDP mode'
    )
    parser.add_argument(
        '--workers', type=int,
        default=8, help='max dataloader workers (per RANK in DDP mode)'
    )
    parser.add_argument(
        '--project',
        default=ROOT / 'runs/train', help='save to project/name'
    )
    parser.add_argument(
        '--name',
        default='exp', help='save to project/name'
    )
    parser.add_argument(
        '--exist-ok',
        action='store_true', help='existing project/name ok, do not increment'
    )
    parser.add_argument(
        '--quad',
        action='store_true', help='quad dataloader'
    )
    parser.add_argument(
        '--cos-lr',
        action='store_true', help='cosine LR scheduler'
    )
    parser.add_argument(
        '--label-smoothing', type=float,
        default=0.0, help='Label smoothing epsilon'
    )
    parser.add_argument(
        '--patience', type=int,
        default=100, help='EarlyStopping patience (epochs without improvement)'
    )
    parser.add_argument(
        '--freeze', nargs='+', type=int,
        default=[0], help='Freeze layers: backbone=10, first3=0 1 2'
    )
    parser.add_argument(
        '--save-period', type=int,
        default=-1, help='Save checkpoint every x epochs (disabled if < 1)'
    )
    parser.add_argument(
        '--local_rank', type=int,
        default=-1, help='DDP parameter, do not modify'
    )

    # configurations for search
    parser.add_argument(
        "--checkpoint_path",
        default='./bestmodel.pt', type=str
    )
    parser.add_argument(
        "--arch_path",
        default='./arch_path.pt', type=str
    )
    parser.add_argument(
        "--no-warmup", dest='warmup',
        action='store_false'
    )
    parser.add_argument(
        "--log-frequency", dest='log_freq',
        default=10
    )
    parser.add_argument(
        "--arch-lr",
        default=0.001, type=float
    )

    # configurations for retrain
    parser.add_argument(
        "--exported_arch_path",
        default=None, type=str
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # ------- arguments -------------------------------------------------------
    args = parse_args()
    if RANK in (-1, 0):
        print_args(vars(args))

    # -------- print global vars ----------------------------------------------
    print(
        "\nFILE=%s, ROOT=%s, LOCAL_RANK=%g, RANK=%g, WORLD_SIZE=%g\n"
        % (FILE, ROOT, LOCAL_RANK, RANK, WORLD_SIZE)
    )

    # --------- resume or search ----------------------------------------------
    if args.resume and not args.evolve:
        # resume training from where it was stopped
        ckpt = args.resume if isinstance(args.resume, str) \
            else get_latest_run()
        assert os.path.isfile(ckpt), \
            'ERROR: --resume checkpoint does not exist'
        with open(
                Path(ckpt).parent.parent / 'args.yaml',
                encoding='utf-8', errors='ignore') as f:
            args = argparse.Namespace(**yaml.safe_load(f))
        args.cfg, args.weights, args.resume = '', ckpt, True
        LOGGER.info('Resuming training from %s', ckpt)
    else:
        # start searching from the scatch
        args.dataset, args.cfg, args.hyp, args.weights, args.project = \
            check_file(args.dataset), check_yaml(args.cfg), \
            check_yaml(args.hyp), str(args.weights), str(args.project)
        assert len(args.cfg) or len(args.weights), \
            'either --cfg or --weights must by sepcified'
        if args.evolve:
            # if default project name, rename to runs/evolve
            if args.project == str(ROOT / 'runs/train'):
                args.project = str(ROOT / 'runs/evolve')
            # pass resume to exist_ok and disable resume
            args.exist_ok, args.resume = args.resume, False
        if args.name == 'cfg':
            args.name = Path(args.cfg).stem  # use basemodel as name
        args.save_dir = str(increment_path(Path(args.project) / args.name,
                                           exist_ok=args.exist_ok))
        LOGGER.info('Start searching from %s', args.cfg)

    # --------- retrain -------------------------------------------------------
    neck_path_freezing = None
    if args.train_mode == 'retrain':
        if args.exported_arch_path is None:
            LOGGER.error(
                'When --train_mode is retrain,'
                ' --exported_arch_path must be specified.'
            )
            sys.exit(-1)
        else:
            if args.nas_type == 'ConcatEntirePath':
                with open('neck_path.json', 'r', encoding='utf-8') as f:
                    neck_path_freezing = json.load(f)

    # --------- device (cuda:0 or cpu) ----------------------------------------
    device = select_device(args.device, batch_size=args.batch_size)

    # --------- DDP mode ------------------------------------------------------
    if LOCAL_RANK != -1:  # multiple GPUs
        MESSAGE = 'is not compatible with AutoNN Multi-GPU DDP training'
        assert not args.image_weights, f'--image-weights {MESSAGE}'
        assert not args.evolve, f'--evolve {MESSAGE}'
        assert args.batch_size != -1, \
            f'AutoBatch with --batch-size -1' \
            f'{MESSAGE}, please pass a valid --batch-size'
        assert args.batch_size % WORLD_SIZE == 0, \
            f'--batch-size {args.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, \
            'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo"
        )

    # =========================================================================

    # --------- directories ---------------------------------------------------
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # wdir = save_dir / 'weights'
    # (wdir.parent if args.evolve else wdir)\
    #     .mkdir(parents=True, exist_ok=True)
    # last, best = wdir / 'lastmodel.pt', wdir / 'bestmodel.pt'

    # --------- hyper-parameters ----------------------------------------------
    with open(args.hyp, encoding='utf-8', errors='ignore') as f:
        hyp = yaml.safe_load(f)
    LOGGER.info(colorstr('hyperparameters: ')
                + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # --------- save run settings ---------------------------------------------
    # (-> evaludator.py)
    # TODO: how can we save arguments in evaluator.py ?

    # --------- loggers -------------------------------------------------------
    # TODO: comment out at this moment
    data = None

    # --------- configuration -------------------------------------------------
    plots = not args.evolve and not args.noplots
    # cuda = device.type != 'cpu'
    # init_seeds(1 + RANK)
    with open(args.dataset, encoding='utf-8', errors='ignore') as f:
        data = yaml.safe_load(f)
    # with torch_distributed_zero_first(LOCAL_RANK):
    #     data = data or check_dataset(data)
    data = check_dataset(data)
    anchors, nc, names = data['anchors'], data['num_classes'], data['names']
    # if args.single_cls and len(data['names']) != 1:
    #     LOGGER.warning(f"Not support single class detection")
    train_path, val_path = data['train'], data['val']
    # check the number of classes
    assert len(names) == nc, \
        f'{len(names)} names found for nc={nc} dataset in {args.dataset}'

    # --------- build model ---------------------------------------------------
    # model(nn.Module) is created from classes at model.py
    # 1. SearchNeck instance has its own attributes
    #           .model          (nn.Sequential)
    #           .save           (jumping layers)
    #           .stride         (grid stride of Detect)
    #  Following attributes will be appended afterward
    #           .nc             (number of classes)
    #           .hyp            (hyper-parameters)
    #           .names          (class names)
    #           .class_weights  (weight factor for class balance)
    from model import SearchNeck, SearchSpaceWrap, ConcatEntirePathNeck
    if args.train_mode == 'retrain':
        assert os.path.isfile(args.exported_arch_path), \
            f"exported_arch_path {args.exported_arch_path} should be a file."
        with fixed_arch(args.exported_arch_path):
            if args.nas_type == 'ConcatBasedNet':
                model = SearchNeck(backbone_yaml=args.cfg,
                                   nc=nc,
                                   anchors=anchors)
            elif args.nas_type == 'ConcatEntirePath':
                from head import Head
                model = SearchSpaceWrap(args.cfg, ConcatEntirePathNeck, Head,
                                        args.neck_cfg, np=3,
                                        dataset=args.dataset,
                                        anchors=anchors, nc=nc, device=device,
                                        path_freezing=neck_path_freezing)
            else:
                print("[!] NAS Type Error: Check option --nas_type")
                exit(-1)
    else:
        if args.nas_type in ['ConcatBasedNet', 'Yolov5Trainer']:
            model = SearchNeck(backbone_yaml=args.cfg,
                               nc=nc,
                               anchors=anchors)
        elif args.nas_type == 'ConcatEntirePath':
            from head import Head
            model = SearchSpaceWrap(args.cfg,
                                    ConcatEntirePathNeck,
                                    Head, args.neck_cfg, np=3,
                                    dataset=args.dataset, anchors=anchors,
                                    nc=nc, device=device,
                                    path_freezing=neck_path_freezing)
        else:
            print("[!] NAS Type Error: Check option --nas_type")
            exit(-1)

    model.to(device)  # (tenace comment: yes, we push model to cuda here)
    LOGGER.info('model create done')

    # --------- initialize weights (or load weights if retrain) ---------------
    # (tenace comment) this is buggy.. it has to be reviewed afterward
    # model.init_model()
    # LOGGER.info('model init done')
    # model.info(verbose=False)

    # --------- pretrained ----------------------------------------------------
    check_suffix(args.weights, '.pt')
    pretrained = args.weights.endswith('.pt')
    if pretrained:
        ckpt = torch.load(args.weights, map_location='cpu')
        ckpt_dict = ckpt['model'].float().state_dict()
        exclude = ['anchor'] \
            if (args.cfg or args.hyp.get('anchors')) and not args.resume \
            else []
        ckpt_dict = intersect_dicts(ckpt_dict,
                                    model.state_dict(), exclude=exclude)
        model.load_state_dict(ckpt_dict, strict=False)
        LOGGER.info(f'model {len(ckpt_dict)}/{len(model.state_dict())}'
                    f' transferred from {args.weights}')

    # --------- freeze model if any -------------------------------------------
    freeze = [f'model.{x}.'
              for x in (args.freeze if len(args.freeze) > 1
                        else range(args.freeze[0]))]
    for k, v in model.named_parameters():
        v.required_grad = True  # train all layers
        if any(x in k for x in freeze):
            LOGGER.info(f'freezeing {k}')
            v.requires_grad = False

    # --------- image size ----------------------------------------------------
    # typically, final resolutions(grids) are 8 x 8, 16 x 16, and 32 x 32
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    # verify imgsz is gs-multiple
    imgsz = check_img_size(args.imgsz, gs, floor=gs * 2)
    LOGGER.info(f'input image size={imgsz}')

    # --------- batch size ----------------------------------------------------
    batch_size = args.batch_size
    if RANK == -1 and batch_size == -1:
        # single-GPU only, estimate best batch size
        from yolov5_utils.autobatch import check_train_batch_size
        batch_size = check_train_batch_size(model, imgsz)
    LOGGER.info(f'optimal batch size={batch_size}')

    # --------- create data loaders -------------------------------------------
    LOGGER.info('creating data loaders...')
    # data_provider = datasets.CocoDataProvider(data) # nni-style
    import datasets
    train_loader, dataset = datasets.create_dataloader(
        train_path,
        imgsz,
        batch_size // WORLD_SIZE,
        gs,
        args.single_cls,
        hyp=hyp,
        augment=True,
        cache=None if args.cache == 'val' else args.cache,
        pad=0.0,
        rect=args.rect,
        rank=LOCAL_RANK,
        workers=args.workers,
        image_weights=args.image_weights,
        quad=args.quad,
        prefix=colorstr('train: '),
        shuffle=True
    )
    mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}.' \
                     f' Possible class labels are 0-{nc - 1}'
    if RANK in [-1, 0]:
        val_loader = datasets.create_dataloader(
            val_path,
            imgsz,
            batch_size // WORLD_SIZE * 2,
            gs,
            args.single_cls,
            hyp=hyp,
            cache=None if args.noval else args.cache,
            rect=True,
            rank=-1,
            workers=args.workers * 2,
            pad=0.5,
            prefix=colorstr('val: ')
        )[0]
        if not args.resume:
            labels = np.concatenate(dataset.labels, 0)
            if plots:
                from yolov5_utils.plots import plot_labels
                plot_labels(labels, names, Path(args.save_dir))
            # Anchors
            if not args.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'],
                              imgsz=imgsz)
    LOGGER.info('creating data loaders done')

    # --------- model attributes ----------------------------------------------
    nl = model.model[-1].nl  # number of detection layers (to scale hyps)
    hyp['box'] *= 3 / nl  # scale to layers
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp['label_smoothing'] = args.label_smoothing
    model.nc = nc
    model.hyp = hyp
    model.class_weights = \
        labels_to_class_weights(dataset.labels, nc).to(device) * nc
    model.names = names

    # --------- optimizer -----------------------------------------------------
    # (--> evaluator.py)
    NBS = 64  # nominal batch size
    # accumulate loss before optimizing
    accumulate = max(round(NBS / batch_size), 1)
    hyp['weight_decay'] *= batch_size * accumulate / NBS  # scale weight_decay
    LOGGER.info(f"scaled weight_decay = {hyp['weight_decay']}")

    g = [], [], []  # optimizer parameter groups
    # normalization layers, i.e. BatchNorm2d()
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            # bias
            g[2].append(v.bias)
        if isinstance(v, bn):
            # weight (no decay)
            g[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            # weight (with decay)
            g[0].append(v.weight)

    # g[2]
    if args.optimizer == 'Adam':
        optimizer = Adam(
            g[2], lr=hyp['lr0'],
            betas=(hyp['momentum'], 0.999)
        )  # adjust beta1 to momentum
    elif args.optimizer == 'AdamW':
        optimizer = AdamW(
            g[2], lr=hyp['lr0'],
            betas=(hyp['momentum'], 0.999)
        )  # adjust beta1 to momentum
    else:
        optimizer = SGD(
            g[2], lr=hyp['lr0'],
            momentum=hyp['momentum'], nesterov=True
        )

    # add g[0] with weight_decay (ex, weights of conv, fc, etc)
    optimizer.add_param_group(
        {'params': g[0],
         'weight_decay': hyp['weight_decay']}
    )
    len_params_with_weight_decay = len(g[0])

    # add g[1] w/o weight_decay (ex. weights of bn)
    optimizer.add_param_group(
        {'params': g[1]}
    )

    # if args.nas_type in ['ConcatEntirePath', 'Yolov5Trainer'] \
    if args.nas_type in ['ConcatEntirePath'] \
            and args.search_type == 'one_stage':
        optimizer.add_param_group(
            {'params': model.neck_module.return_list,
             'weight_decay': hyp['weight_decay']})
        len_params_with_weight_decay += len(model.neck_module.return_list)

    LOGGER.info(
        f"{colorstr('optimizer:')} {type(optimizer).__name__}"
        f" with parameter groups "
        f"{len(g[1])} weight (no decay),"
        f" {len_params_with_weight_decay} weight, {len(g[2])} bias"
    )
    del g

    # --------- scheduler -----------------------------------------------------
    # (--> evaluator.py)
    if args.cos_lr:
        # cosine 1->hyp['lrf']
        lf = one_cycle(1, hyp['lrf'], args.epochs)
    else:
        # linear
        lf = linear(hyp['lrf'], args.epochs)
        # lf = lambda \
        #     x: (1 - x / args.epochs) * (1.0 - hyp['lrf']) + hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, args.epochs)

    # --------- Exponential Moving Average ------------------------------------
    # (--> evaluator.py)
    # ema = ModelEMA(model) if RANK in [-1, 0] else None

    # --------- loss function -------------------------------------------------
    compute_loss = ComputeLoss(model)  # init loss class
    # from yolov4_utils.general import compute_loss  #yolov4

    # --------- execute -------------------------------------------------------
    from evaluator import ConcatBasedNetTrainer, ConcatEntirePathTrainer
    if args.train_mode == 'search':
        # save run settings
        # Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        # with open(Path(args.save_dir) / 'hyp.yaml', 'w') as f:
        #     yaml.safe_dump(hyp, f, sort_keys=False)
        # with open(Path(args.save_dir) / 'args.yaml', 'w') as f:
        #     yaml.safe_dump(vars(args), f, sort_keys=False)

        if args.nas_type == 'ConcatEntirePath':
            trainer = ConcatEntirePathTrainer(
                model,
                args=args,
                hyp=hyp,
                loss=compute_loss,
                optimizer=optimizer,
                lf=lf,
                scheduler=scheduler,
                num_epochs=args.epochs,
                dataset=dataset,
                train_data=train_loader,
                val_data=val_loader,
                img_size=imgsz,
                batch_size=batch_size,
                workers=args.workers,
                device=device,
                log_freq=args.log_freq,
                arc_lr=args.arch_lr,
                search_type=args.search_type  # 'two_stage'
            )
        elif args.nas_type == 'ConcatBasedNet':
            trainer = ConcatBasedNetTrainer(
                model,
                loss=compute_loss,
                optimizer=optimizer,
                lf=lf,
                scheduler=scheduler,
                num_epochs=args.epochs,
                dataset=dataset,
                train_data=train_loader,
                val_data=val_loader,
                img_size=imgsz,
                batch_size=batch_size,
                workers=args.workers,
                device=device,
                log_frequency=args.log_freq,
                arc_lr=args.arch_lr,
                save_dir=args.save_dir
            )
        elif args.nas_type == 'Yolov5Trainer':
            import train
            train.run(
                args=vars(args),
                model=model,
                # compute_loss=compute_loss,
                # optimizer=optimizer,
                # lf=lf,
                # scheduler=scheduler,
                # data=dataset,
                # train_data=train_loader,
                # val_data=val_loader,
                # imgsz=imgsz,
                # batch_size=batch_size,
                # device=device,
            )
        else:
            raise NotImplementedError('Not supported NAS trainer.')

        if args.nas_type != 'Yolov5Trainer':
            trainer.fit()
            print('Final architecture:', trainer.export())
            with open('neck_path.json', 'w', encoding='utf-8') as f:
                json.dump([x.detach().cpu().tolist()
                          for x in trainer.export()], f)
    elif args.train_mode == 'retrain':
        # this is retrain
        '''
        trainer = Retrain(model,
                          optimizer,
                          device,
                          train_loader,
                          val_loader,
                          test_loader,
                          n_epochs=300)
        trainer.run()
        '''
        trainer = ConcatEntirePathTrainer(
            model,
            args=args,
            hyp=hyp,
            loss=compute_loss,
            optimizer=optimizer,
            lf=lf,
            scheduler=scheduler,
            num_epochs=args.epochs,
            dataset=dataset,
            train_data=train_loader,
            val_data=val_loader,
            img_size=imgsz,
            batch_size=batch_size,
            workers=args.workers,
            device=device,
            log_freq=args.log_freq,
            arc_lr=args.arch_lr,
            search_type='retrain'
        )
        trainer.fit()
