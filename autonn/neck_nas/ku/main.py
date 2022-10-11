"""a module for initiating search or retrain"""

import json
import logging
import math
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch import optim
from torch.optim import lr_scheduler
# for DDP
import torch.distributed as dist
# from torchvision import transforms
# for visualization
from tensorboardX import SummaryWriter
from nni.retiarii.fixed import fixed_arch

from .retrain import Retrain

# for Scaled-YOLOv4
import yaml

from .syolo_utils.datasets import create_dataloader

from .syolo_utils.general import (
    check_img_size, compute_loss, get_latest_run, check_git_status,
    check_file, increment_dir)
from .syolo_utils.torch_utils import select_device

logger = logging.getLogger('nni_proxylessnas')

def run_nas():
    print("__________run_nas__________________")
    # parser = ArgumentParser("proxylessnas")
    # parser.add_argument("--args-yaml", default='yaml/args.yaml',
    #                     type=str, help='search/retrain args yaml')

    # opts = parser.parse_args()

    # load search/retrain arguments from yaml
    # with open(opts.args_yaml, encoding='utf8') as f:
    #     args = yaml.load(f, Loader=yaml.FullLoader)
    with open('neck_nas/ku/yaml/args.yaml', encoding='utf8') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    # Resume
    if args['resume']:
        if args['resume'] == 'get_last':
            last = get_latest_run()     # resume from most recent run
        if last and not args['weights']:
            print(f'Resuming training from {last}')
        if args['resume'] and not args['weights']:
            args['weights'] = last
    if args['local_rank'] == -1 or \
            ("RANK" in os.environ and os.environ["RANK"] == "0"):
        check_git_status()

    args['train_hyp'] = \
        args['train_hyp'] or ('yaml/hyp.finetune.yaml' if args['weights']
                              else 'yaml/hyp.scratch.yaml')
    args['data_cfg'], args['model_cfg'], args['train_hyp'] = \
        check_file(args['data_cfg']), check_file(args['model_cfg']), \
        check_file(args['train_hyp'])
    assert len(args['model_cfg']) or len(args['weights']), \
        'either --cfg or --weights must be specified'

    args['img_size'].extend([args['img_size'][-1]] *
                            (2 - len(args['img_size'])))
    # extend to 2 sizes (train, test)
    device = select_device(args['gpu_device'], batch_size=args['batch_size'])
    args['total_batch_size'] = args['batch_size']
    args['world_size'] = 1
    args['global_rank'] = -1

    # DDP mode TODO
    if args['local_rank'] != -1:
        assert torch.cuda.device_count() > args['local_rank']
        torch.cuda.set_device(args['local_rank'])
        device = torch.device('cuda', args['local_rank'])
        # distributed backend
        dist.init_process_group(backend='nccl', init_method='env://')
        args['world_size'] = dist.get_world_size()
        args['global_rank'] = dist.get_rank()
        assert args['batch_size'] % args['world_size'] == 0, \
            '--batch-size must be multiple of CUDA device count'
        args['batch_size'] = args['total_batch_size'] // args['world_size']

    print(args)
    with open(args['train_hyp'], encoding='utf8') as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
    with open(args['data_cfg'], encoding='utf8') as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict

    if args['train_mode'] == 'retrain' and args['exported_arch_path'] is None:
        logger.error('When --train_mode is retrain, \
            --exported_arch_path must be specified.')
        sys.exit(-1)

    if args['model_name'] == 'yolov4':
        from .model import SearchYolov4

        # model generation
        if args['train_mode'] == 'retrain':    # for retrain
            assert os.path.isfile(args['exported_arch_path']), \
                f"exported_arch_path {args['exported_arch_path']} \
                    should be a file."
            with fixed_arch(args['exported_arch_path']):
                model = \
                    SearchYolov4(cfg=args['model_cfg'],
                                 names=(data_dict['item'] if args['single_cls']
                                        else data_dict['names']),
                                 hyp=hyp, weights=args['weights'], ch=3,
                                 nc=1 if args['single_cls']
                                 else data_dict['nc'])
            if args['resume']:
                log_path = args['weights'].split('/weights')[0]
            else:
                log_path = increment_dir(
                    Path(args['exported_arch_path'].split('search')[0])
                    / 'retrain', args['name'])

        else:   # for search
            model = SearchYolov4(cfg=args['model_cfg'],
                                 names=(data_dict['item'] if args['single_cls']
                                        else data_dict['names']),
                                 hyp=hyp, weights=args['weights'], ch=3,
                                 nc=1 if args['single_cls']
                                 else data_dict['nc'])
            if args['resume']:
                log_path = args['weights'].split('/weights')[0]
            else:
                log_path = increment_dir(
                    Path(args['log_dir'] + 'ScaledYolov4/')
                    / 'exp', args['name']) + "/search"
        writer = SummaryWriter(log_dir=log_path)
        logger.info('SearchYolov4 model create done')

    # initialize or load model weights
    model.init_model()
    logger.info('Model init done')

    logger.info('Creating data provider...')
    # create data loaders
    if args['dataset'] == 'coco':
        gs = int(max(model.stride))     # grid size (max stride)
        # verify imgsz are gs-multiples
        imgsz, imgsz_test = [check_img_size(x, gs) for x in args['img_size']]
        train_loader, dataset = \
            create_dataloader(data_dict['train'], imgsz, args['batch_size'],
                              gs, args['single_cls'], hyp=hyp, augment=True,
                              cache=args['cache_images'], rect=args['rect'],
                              local_rank=args['global_rank'],
                              world_size=args['world_size'])
        test_loader = \
            create_dataloader(data_dict['val'], imgsz_test, args['batch_size'],
                              gs, args['single_cls'], hyp=hyp, augment=False,
                              cache=args['cache_images'], rect=True,
                              local_rank=-1, world_size=args['world_size'])[0]

    logger.info('Creating data provider done')

    # Optimizer
    NBS = 64    # nominal batch size
    total_batch_size = args['batch_size']
    # accumulate loss before optimizing
    accumulate = max(round(NBS / total_batch_size), 1)
    # scale weight_decay
    hyp['weight_decay'] *= total_batch_size * accumulate / NBS

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_parameters():
        v.requires_grad = True
        if '.bias' in k:
            pg2.append(v)   # biases
        elif '.weight' in k and '.bn' not in k:
            pg1.append(v)   # apply weight decay
        else:
            pg0.append(v)   # all else

    if args['weight_optim'] == 'sgd':     # SGD
        hyp['weight_optim'] = 'sgd'
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'],
                              nesterov=True)
    elif args['weight_optim'] == 'adam':  # Adam
        hyp['weight_optim'] = 'adam'
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'],
                               0.999))  # adjust beta1 to momentum

    # add pg1 with weight_dacay
    optimizer.add_param_group({'params': pg1,
                               'weight_decay': hyp['weight_decay']})
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    print(f'Optimizer groups: {len(pg2)} .bias, {len(pg1)} \
          conv.weight, {len(pg0)} other')
    del pg0, pg1, pg2

    epochs = args['epoch']

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR

    def lambda_func(x_val):
        """A lambda function for the scheduler"""
        return (((1 + math.cos(x_val * math.pi / epochs)) / 2) ** 1.0) \
            * 0.8 + 0.2  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)

    if args['train_mode'] == 'search':
        if args['train_task'] == 'detection':
            # for object detection model
            from .search import ProxylessDetTrainer
            trainer = ProxylessDetTrainer(model,
                                          args=args,
                                          hyp=hyp,
                                          loss=compute_loss,
                                          optimizer=optimizer,
                                          imgsz=imgsz,
                                          imgsz_test=imgsz_test,
                                          train=train_loader,
                                          dataset=dataset,
                                          test=test_loader,
                                          writer=writer,
                                          lf=lambda_func,
                                          scheduler=scheduler,
                                          num_epochs=epochs,
                                          arc_learning_rate=args['arch_lr'],
                                          device=device)

        trainer.fit()
        print('Final architecture:', trainer.export())
        with open(Path(writer.logdir) / 'final_arch.json',
                  'w', encoding='utf8') as f:
            json.dump(trainer.export(), f)

    elif args['train_mode'] == 'retrain':
        # this is retrain, TODO
        trainer = Retrain(model,
                          args=args,
                          hyp=hyp,
                          loss=compute_loss,
                          optimizer=optimizer,
                          imgsz=imgsz,
                          imgsz_test=imgsz_test,
                          train=train_loader,
                          dataset=dataset,
                          test=test_loader,
                          writer=writer,
                          lf=lambda_func,
                          scheduler=scheduler,
                          num_epochs=epochs,
                          device=device)
        trainer.run()


# if __name__ == "__main__":
#     parser = ArgumentParser("proxylessnas")
#     parser.add_argument("--args-yaml", default='yaml/args.yaml',
#                         type=str, help='search/retrain args yaml')

#     opts = parser.parse_args()

#     # load search/retrain arguments from yaml
#     with open(opts.args_yaml, encoding='utf8') as f:
#         args = yaml.load(f, Loader=yaml.FullLoader)

#     # Resume
#     if args['resume']:
#         if args['resume'] == 'get_last':
#             last = get_latest_run()     # resume from most recent run
#         if last and not args['weights']:
#             print(f'Resuming training from {last}')
#         if args['resume'] and not args['weights']:
#             args['weights'] = last
#     if args['local_rank'] == -1 or \
#             ("RANK" in os.environ and os.environ["RANK"] == "0"):
#         check_git_status()

#     args['train_hyp'] = \
#         args['train_hyp'] or ('yaml/hyp.finetune.yaml' if args['weights']
#                               else 'yaml/hyp.scratch.yaml')
#     args['data_cfg'], args['model_cfg'], args['train_hyp'] = \
#         check_file(args['data_cfg']), check_file(args['model_cfg']), \
#         check_file(args['train_hyp'])
#     assert len(args['model_cfg']) or len(args['weights']), \
#         'either --cfg or --weights must be specified'

#     args['img_size'].extend([args['img_size'][-1]] *
#                             (2 - len(args['img_size'])))
#     # extend to 2 sizes (train, test)
#     device = select_device(args['gpu_device'], batch_size=args['batch_size'])
#     args['total_batch_size'] = args['batch_size']
#     args['world_size'] = 1
#     args['global_rank'] = -1

#     # DDP mode TODO
#     if args['local_rank'] != -1:
#         assert torch.cuda.device_count() > args['local_rank']
#         torch.cuda.set_device(args['local_rank'])
#         device = torch.device('cuda', args['local_rank'])
#         # distributed backend
#         dist.init_process_group(backend='nccl', init_method='env://')
#         args['world_size'] = dist.get_world_size()
#         args['global_rank'] = dist.get_rank()
#         assert args['batch_size'] % args['world_size'] == 0, \
#             '--batch-size must be multiple of CUDA device count'
#         args['batch_size'] = args['total_batch_size'] // args['world_size']

#     print(args)
#     with open(args['train_hyp'], encoding='utf8') as f:
#         hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
#     with open(args['data_cfg'], encoding='utf8') as f:
#         data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict

#     if args['train_mode'] == 'retrain' and args['exported_arch_path'] is None:
#         logger.error('When --train_mode is retrain, \
#             --exported_arch_path must be specified.')
#         sys.exit(-1)

#     if args['model_name'] == 'yolov4':
#         from model import SearchYolov4

#         # model generation
#         if args['train_mode'] == 'retrain':    # for retrain
#             assert os.path.isfile(args['exported_arch_path']), \
#                 f"exported_arch_path {args['exported_arch_path']} \
#                     should be a file."
#             with fixed_arch(args['exported_arch_path']):
#                 model = \
#                     SearchYolov4(cfg=args['model_cfg'],
#                                  names=(data_dict['item'] if args['single_cls']
#                                         else data_dict['names']),
#                                  hyp=hyp, weights=args['weights'], ch=3,
#                                  nc=1 if args['single_cls']
#                                  else data_dict['nc'])
#             if args['resume']:
#                 log_path = args['weights'].split('/weights')[0]
#             else:
#                 log_path = increment_dir(
#                     Path(args['exported_arch_path'].split('search')[0])
#                     / 'retrain', args['name'])

#         else:   # for search
#             model = SearchYolov4(cfg=args['model_cfg'],
#                                  names=(data_dict['item'] if args['single_cls']
#                                         else data_dict['names']),
#                                  hyp=hyp, weights=args['weights'], ch=3,
#                                  nc=1 if args['single_cls']
#                                  else data_dict['nc'])
#             if args['resume']:
#                 log_path = args['weights'].split('/weights')[0]
#             else:
#                 log_path = increment_dir(
#                     Path(args['log_dir'] + 'ScaledYolov4/')
#                     / 'exp', args['name']) + "/search"
#         writer = SummaryWriter(log_dir=log_path)
#         logger.info('SearchYolov4 model create done')

#     # initialize or load model weights
#     model.init_model()
#     logger.info('Model init done')

#     logger.info('Creating data provider...')
#     # create data loaders
#     if args['dataset'] == 'coco':
#         gs = int(max(model.stride))     # grid size (max stride)
#         # verify imgsz are gs-multiples
#         imgsz, imgsz_test = [check_img_size(x, gs) for x in args['img_size']]
#         train_loader, dataset = \
#             create_dataloader(data_dict['train'], imgsz, args['batch_size'],
#                               gs, args['single_cls'], hyp=hyp, augment=True,
#                               cache=args['cache_images'], rect=args['rect'],
#                               local_rank=args['global_rank'],
#                               world_size=args['world_size'])
#         test_loader = \
#             create_dataloader(data_dict['val'], imgsz_test, args['batch_size'],
#                               gs, args['single_cls'], hyp=hyp, augment=False,
#                               cache=args['cache_images'], rect=True,
#                               local_rank=-1, world_size=args['world_size'])[0]

#     logger.info('Creating data provider done')

#     # Optimizer
#     NBS = 64    # nominal batch size
#     total_batch_size = args['batch_size']
#     # accumulate loss before optimizing
#     accumulate = max(round(NBS / total_batch_size), 1)
#     # scale weight_decay
#     hyp['weight_decay'] *= total_batch_size * accumulate / NBS

#     pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
#     for k, v in model.named_parameters():
#         v.requires_grad = True
#         if '.bias' in k:
#             pg2.append(v)   # biases
#         elif '.weight' in k and '.bn' not in k:
#             pg1.append(v)   # apply weight decay
#         else:
#             pg0.append(v)   # all else

#     if args['weight_optim'] == 'sgd':     # SGD
#         hyp['weight_optim'] = 'sgd'
#         optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'],
#                               nesterov=True)
#     elif args['weight_optim'] == 'adam':  # Adam
#         hyp['weight_optim'] = 'adam'
#         optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'],
#                                0.999))  # adjust beta1 to momentum

#     # add pg1 with weight_dacay
#     optimizer.add_param_group({'params': pg1,
#                                'weight_decay': hyp['weight_decay']})
#     optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
#     print(f'Optimizer groups: {len(pg2)} .bias, {len(pg1)} \
#           conv.weight, {len(pg0)} other')
#     del pg0, pg1, pg2

#     epochs = args['epoch']

#     # Scheduler https://arxiv.org/pdf/1812.01187.pdf
#     # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR

#     def lambda_func(x_val):
#         """A lambda function for the scheduler"""
#         return (((1 + math.cos(x_val * math.pi / epochs)) / 2) ** 1.0) \
#             * 0.8 + 0.2  # cosine
#     scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)

#     if args['train_mode'] == 'search':
#         if args['train_task'] == 'detection':
#             # for object detection model
#             from search import ProxylessDetTrainer
#             trainer = ProxylessDetTrainer(model,
#                                           args=args,
#                                           hyp=hyp,
#                                           loss=compute_loss,
#                                           optimizer=optimizer,
#                                           imgsz=imgsz,
#                                           imgsz_test=imgsz_test,
#                                           train=train_loader,
#                                           dataset=dataset,
#                                           test=test_loader,
#                                           writer=writer,
#                                           lf=lambda_func,
#                                           scheduler=scheduler,
#                                           num_epochs=epochs,
#                                           arc_learning_rate=args['arch_lr'],
#                                           device=device)

#         trainer.fit()
#         print('Final architecture:', trainer.export())
#         with open(Path(writer.logdir) / 'final_arch.json',
#                   'w', encoding='utf8') as f:
#             json.dump(trainer.export(), f)

#     elif args['train_mode'] == 'retrain':
#         # this is retrain, TODO
#         trainer = Retrain(model,
#                           args=args,
#                           hyp=hyp,
#                           loss=compute_loss,
#                           optimizer=optimizer,
#                           imgsz=imgsz,
#                           imgsz_test=imgsz_test,
#                           train=train_loader,
#                           dataset=dataset,
#                           test=test_loader,
#                           writer=writer,
#                           lf=lambda_func,
#                           scheduler=scheduler,
#                           num_epochs=epochs,
#                           device=device)
#         trainer.run()
