import json
import logging
import math
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
# from torchvision import transforms
from nni.retiarii.fixed import fixed_arch

# from retrain import Retrain

### for Scaled-YOLOv4
import yaml

from datasets import create_dataloader
from syolo_utils.general import (
    check_img_size, compute_loss, get_latest_run, check_git_status, check_file, increment_dir)
from syolo_utils.torch_utils import select_device
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
# for DDP
import torch.distributed as dist
###

# for visualization
from tensorboardX import SummaryWriter

logger = logging.getLogger('nni_proxylessnas')

if __name__ == "__main__":
    parser = ArgumentParser("proxylessnas")
    parser.add_argument("--model-name", default='yolov4', type=str)
    # configurations for dataset
    parser.add_argument("--dataset", default='coco', type=str)
    # parser.add_argument("--n-worker", default=5, type=int)
    # configurations for YOLOv4 model
    parser.add_argument("--weights", default='', type=str, help='initial weights path')
    parser.add_argument("--cfg", default=None, type=str, help='model.yaml path')
    parser.add_argument('--data', default='yaml/coco.yaml', type=str, help='data.yaml path')
    # configurations for training 
    parser.add_argument("--train-task", default='classification', type=str, choices=['classification', 'detection'])
    parser.add_argument("--train-mode", default='search', type=str, choices=['search', 'retrain'])
    parser.add_argument("--hyp", default='', type=str, help='hyperparameters path, i.e. yaml/hyp.scratch.yaml')
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--img-size", nargs='+', default=[640, 640], type=int, help='train, test sizes')
    parser.add_argument("--rect", action='store_true', help='rectangular training')
    parser.add_argument("--resume", nargs='?', const='get_last', default=False, help='resume from given path/last.pt, or most recent run if blank')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    # parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', default='', type=str, help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local-rank', type=int, default=-1, help='DDP parameter, do not modify')

    parser.add_argument("--optimizer", default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument("--arch-lr", default=0.001, type=float)
    # configurations for search
    parser.add_argument("--search-epochs", default=200, type=int)
    parser.add_argument("--device", default='1', type=str)
    parser.add_argument("--no-warmup", dest='warmup', action='store_false')
    parser.add_argument("--logdir", type=str, default='runs/', help='logging directory')
    # configurations for retrain
    parser.add_argument("--exported-arch-path", default=None, type=str)
    parser.add_argument("--retrain-epochs", default=300, type=int)

    args = parser.parse_args()

    # Resume
    if args.resume:
        last = get_latest_run() if args.resume == 'get_last' else args.resume  # resume from most recent run
        if last and not args.weights:
            print(f'Resuming training from {last}')
        args.weights = last if args.resume and not args.weights else args.weights
    if args.local_rank == -1 or ("RANK" in os.environ and os.environ["RANK"] == "0"):
        check_git_status()

    args.hyp = args.hyp or ('yaml/hyp.finetune.yaml' if args.weights else 'yaml/hyp.scratch.yaml')
    args.data, args.cfg, args.hyp = check_file(args.data), check_file(args.cfg), check_file(args.hyp)   # check files
    assert len(args.cfg) or len(args.weights), 'either --cfg or --weights must be specified'

    args.img_size.extend([args.img_size[-1]] * (2 - len(args.img_size)))    # extend to 2 sizes (train, test)
    device = select_device(args.device, batch_size=args.batch_size)
    args.total_batch_size = args.batch_size
    args.world_size = 1
    args.global_rank = -1

    # DDP mode
    if args.local_rank != -1:
        assert troch.cuda.device_count() > args.local_rank
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')   # distributed backend
        args.world_size = dist.get_world_size()
        args.global_rank = dist.get_rank()
        assert args.batch_size % args.world_size == 0, '--batch-size must be multiple of CUDA device count'
        args.batch_size = args.total_batch_size // args.world_size

    print(args)
    with open(args.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
    with open(args.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict

    if args.train_mode == 'retrain' and args.exported_arch_path is None:
        logger.error('When --train_mode is retrain, --exported_arch_path must be specified.')
        sys.exit(-1)

    if args.model_name == 'yolov4':
        from model import SearchYolov4

        # model generation
        if args.train_mode == 'retrain':
            assert os.path.isfile(args.exported_arch_path), \
                "exported_arch_path {} should be a file.".format(args.exported_arch_path)
            with fixed_arch(args.exported_arch_path):
                model = SearchYolov4(cfg=args.cfg,
                                    names=data_dict['item'] if args.single_cls else data_dict['names'],
                                    hyp=hyp,
                                    weights=args.weights,
                                    ch=3,
                                    nc=1 if args.single_cls else data_dict['nc'])
            log_path = args.exported_arch_path.split('/search')[0] + "/retrain"
            writer = SummaryWriter(log_dir=log_path)
        else:
            model = SearchYolov4(cfg=args.cfg,
                                 names=data_dict['item'] if args.single_cls else data_dict['names'],
                                 hyp=hyp,
                                 weights=args.weights,
                                 ch=3,
                                 nc=1 if args.single_cls else data_dict['nc'])
            writer = SummaryWriter(log_dir=increment_dir(Path(args.logdir + 'Yolov4/') / 'exp', args.name) + "/search")
        logger.info('SearchYolov4 model create done')

    # initialize or load model weights
    model.init_model()
    logger.info('Model init done')

    logger.info('Creating data provider...')
    # create data loaders
    if args.dataset == 'coco':
        gs = int(max(model.stride)) # grid size (max stride)
        imgsz, imgsz_test = [check_img_size(x, gs) for x in args.img_size]  # verify imgsz are gs-multiples
        trainloader, dataset = create_dataloader(data_dict['train'], imgsz, args.batch_size, gs, args, hyp=hyp, augment=True,
                                                                       cache=args.cache_images, rect=args.rect, local_rank=args.global_rank,
                                                                       world_size=args.world_size)
        testloader = create_dataloader(data_dict['val'], imgsz_test, args.batch_size, gs, args, hyp=hyp, augment=False,
                                               cache=args.cache_images, rect=True, local_rank=-1, world_size=args.world_size)[0]

    logger.info('Creating data provider done')

    # Optimizer
    nbs = 64    # nominal batch size
    total_batch_size = args.batch_size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_parameters():
        v.requires_grad = True
        if '.bias' in k:
            pg2.append(v)   # biases
        elif '.weight' in k and '.bn' not in k:
            pg1.append(v)   # apply weight decay
        else:
            pg0.append(v)   # all else

    if args.optimizer == 'sgd': # SGD
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    elif args.optimizer == 'adam':  # Adam
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']}) # add pg1 with weight_dacay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    if args.train_mode == 'search':
        epochs = args.search_epochs
    elif args.train_mode == 'retrain':
        epochs = args.retrain_epochs
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)


    if args.train_mode == 'search':
        if args.train_task == 'detection':
            # for object detection model
            from search import ProxylessDetTrainer
            trainer = ProxylessDetTrainer(model,
                                          args=args,
                                          hyp=hyp,
                                          loss=compute_loss,
                                          imgsz=imgsz,
                                          imgsz_test=imgsz_test,
                                          train=trainloader,
                                          dataset=dataset,
                                          test=testloader,
                                          writer=writer,
                                          optimizer=optimizer,
                                          lf=lf,
                                          scheduler=scheduler,
                                          num_epochs=epochs,
                                          arc_learning_rate=args.arch_lr,
                                          device=device)

        trainer.fit()
        print('Final architecture:', trainer.export())
        json.dump(trainer.export(), open(Path(writer.logdir) / 'checkpoint.json', 'w'))

    elif args.train_mode == 'retrain':
        # this is retrain, TODO
        # trainer = Retrain(model, optimizer, device, data_provider, logdir=log_path, writer=writer, n_epochs=epochs)
        # trainer.run()
        pass
