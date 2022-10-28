import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch.distributed as dist
import torch.utils.data
import torch.nn.functional as F
import yaml
from torch.cuda import amp
# from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

# import test
from .test import test
from .syolo_utils.general \
    import (labels_to_class_weights, plot_labels, check_anchors,
            strip_optimizer, plot_results, labels_to_image_weights,
            plot_images, fitness)
from .syolo_utils.torch_utils import init_seeds, ModelEMA, intersect_dicts


class Retrain:
    """
    Retrainer for the final architecture.
    Based on the framework of Scaled YOLOv4 (YOLOv5)

    Parameters
    ----------
    model : nn.Module
        PyTorch model to be trained.
    args : dict
        Arguments from yaml.
    hyp : dict
        Hyperparameters from the yaml.
    loss : callable
        Receives logits and ground truth label, return a loss tensor.
    optimizer : Optimizer
        The optimizer used for optimizing the model weights.
    imgsz : int
        Image size for train.
    imgssz_test : int
        Image size for test.
    train : torch.utils.data.DataLoader
        Train data loader.
    dataset : LoadImageAndLabels
        Train dataset.
    test : torch.utils.data.DataLoader
        Test data loader.
    writer : SummaryWriter
        SummaryWriter for Tensorboard.
    lf : lambda function
        Lambda function for learning rate scheduling.
    scheduler : torch.optim.lr_scheduler.LambdaLR
        Learning rate scheduler.
    num_epochs : int
        Number of epochs planned for training.
    device : torch.device
        ``torch.device("cpu")`` or ``torch.device("cuda")``.
    """
    def __init__(self, model, args, hyp, loss, optimizer,
                 imgsz, imgsz_test, train, dataset, test, writer,
                 lf, scheduler, num_epochs, device):
        self.model = model
        self.args = args
        self.hyp = hyp
        self.loss = loss
        self.optimizer = optimizer
        self.image_size = imgsz
        self.image_size_test = imgsz_test
        self.train_loader = train
        self.dataset = dataset
        self.test_loader = test
        self.writer = writer
        self.lf = lf
        self.scheduler = scheduler
        self.epochs = num_epochs
        self.device = device

        self.num_workers = self.train_loader.num_workers

        print(f'Hyperparameters {self.hyp}')
        self.log_dir = Path(self.writer.logdir)
        # weights directory
        self.wdir = str(self.log_dir / 'weights') + os.sep
        os.makedirs(self.wdir, exist_ok=True)
        self.last = self.wdir + 'last.pt'
        self.best = self.wdir + 'best.pt'
        self.results_file = str(self.log_dir / 'results.txt')
        self.batch_size, self.total_batch_size, self.rank = \
            args['batch_size'], args['total_batch_size'], args['global_rank']

        # Save run settings
        with open(self.log_dir / 'hyp.yaml', 'w') as f:
            yaml.dump(self.hyp, f, sort_keys=False)
        with open(self.log_dir / 'args.yaml', 'w') as f:
            yaml.dump(self.args, f, sort_keys=False)

        self.cuda = self.device.type != 'cpu'
        init_seeds(2 + self.rank)
        with open(self.args['data_cfg']) as f:
            data_dict = yaml.load(f, Loader=yaml.FullLoader)
        # number classes, names
        self.nc, self.names = \
            (1, ['item']) if self.args['single_cls'] \
            else (int(data_dict['nc']), data_dict['names'])
        self.model.to(self.device)

        # pretrained (resume)
        pretrained = self.args['weights'].endswith('.pt')
        self.start_epoch, self.best_fitness = 0, 0.0
        if pretrained:
            ckpt = torch.load(self.args['weights'], map_location=self.device)

            # Model
            exclude = ['anchor'] if args['model_cfg'] else []  # exclude keys
            state_dict = ckpt['model'].float().state_dict()     # to FP32
            state_dict = intersect_dicts(state_dict, self.model.state_dict(),
                                         exclude=exclude)  # intersect
            self.model.load_state_dict(state_dict, strict=False)    # load
            print('Transferred %g/%g items from %s' %
                  (len(state_dict), len(self.model.state_dict()),
                   self.args['weights']))    # report

            # Optimizer
            if ckpt['optimizer'] is not None:
                self.optimizer.load_state_dict(ckpt['optimizer'])
                self.best_fitness = ckpt['best_fitness']

            # Results
            if ckpt.get('training_results') is not None:
                with open(self.results_file, 'w') as file:
                    file.write(ckpt['training_results'])    # write results.txt

            # Epochs
            self.start_epoch = ckpt['epoch'] + 1
            if self.epochs < self.start_epoch:
                print('%s has been trained for %g epochs. \
                      Fine-tuning for %g additional epochs.' %
                      (self.args['weights'], ckpt['epoch'], self.epochs))
                self.epochs += ckpt['epoch']    # finetune additional epochs
            del ckpt, state_dict

        # DP mode
        if self.cuda and self.rank == -1 and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        # SyncBatchNorm
        convert_sync_batchnorm = torch.nn.SyncBatchNorm.convert_sync_batchnorm
        if self.args['sync_bn'] and self.cuda and self.rank != -1:
            self.model = convert_sync_batchnorm(self.model).to(self.device)
            print('Using SyncBatchNorm()')

        # Exponential moving avearage
        self.ema = ModelEMA(self.model) if self.rank in [-1, 0] else None
        self.nb = len(self.train_loader)    # number of batches
        self.nbs = 64
        self.accumulate = max(round(self.nbs / self.total_batch_size), -1)
        if self.rank in [-1, 0]:
            # set EMA updates ***
            self.ema.updates = self.start_epoch * self.nb // self.accumulate

        # Model parameters
        # scale coco-tuned hyp['cls'] to current dataset
        self.hyp['cls'] *= self.nc / 80.
        # attach class weights
        self.model.class_weights = \
            labels_to_class_weights(self.dataset.labels,
                                    self.nc).to(self.device)

        # Class frequency
        if self.rank in [-1, 0]:
            labels = np.concatenate(self.dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.
            # model._initialize_biases(cf.to(device))
            plot_labels(labels, save_dir=self.log_dir)
            if self.writer:
                self.writer.add_histogram('classes', c, 0)

            # Check anchors
            if not self.args['noautoanchor']:
                check_anchors(self.dataset, model=self.model,
                              thr=self.hyp['anchor_t'], imgsz=self.image_size)

        # number of warmup iterations, max(3 epochs, 1k iterations)
        self.nw = max(3 * self.nb, 1e3)
        self.gs = int(max(self.model.stride))   # grid size (max stride)
        self.maps = np.zeros(self.nc)   # mAP per class
        # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Obj', 'val Cls'
        self.results = (0, 0, 0, 0, 0, 0, 0)
        self.scheduler.last_epoch = self.start_epoch - 1    # do not move
        self.scaler = amp.GradScaler(enabled=self.cuda)
        if self.rank in [0, -1]:
            print('Image sizes %g train, %g test' % (self.image_size,
                                                     self.image_size_test))
            print('Using %g dataloader workers' % self.num_workers)
            print('Starting search for %g epochs...' % self.epochs)

    def run(self):
        # self.model = torch.nn.DataParallel(self.model)

        # train
        t0 = time.time()
        for epoch in range(self.start_epoch, self.epochs):
            self._train_one_epoch(epoch)

        if self.rank in [-1, 0]:
            # Strip optimizers
            n = ('_' if len(self.args['name']) and not
                 self.args['name'].isnumeric() else '') + self.args['name']
            fresults, flast, fbest = \
                'results%s.txt' % n, self.wdir + 'last%s.pt' % n, \
                self.wdir + 'best%s.pt' % n
            for f1, f2 in zip([self.wdir + 'last.pt', self.wdir + 'best.pt',
                              'results.txt'], [flast, fbest, fresults]):
                if os.path.exists(f1):
                    os.rename(f1, f2)  # rename
                    ispt = f2.endswith('.pt')  # is *.pt
                    if ispt:
                        # strip optimizer
                        strip_optimizer(f2, f2.replace('.pt', '_strip.pt'))
                    else:
                        None
            # Finish
            plot_results(save_dir=self.log_dir)  # save as results.png
            print('%g epochs completed in %.3f hours.\n' %
                  (self.epochs - self.start_epoch + 1,
                   (time.time() - t0) / 3600))

        # TODO
        # validate
        # self.validate(is_test=False)
        # test
        # self.validate(is_test=True)

        dist.destroy_process_group() if self.rank not in [-1, 0] else None
        torch.cuda.empty_cache()
        # return self.results

    def _train_one_epoch(self, epoch):
        self.model.train()

        # Update image weights (optional)
        if self.dataset.image_weights:
            # Generate indices
            if self.rank in [-1, 0]:
                w = self.model.class_weights.cpu().numpy() \
                    * (1 - self.maps) ** 2  # class weights
                image_weights = \
                    labels_to_image_weights(self.dataset.labels, nc=self.nc,
                                            class_weights=w)
                # rand weighted idx
                self.dataset.indices = random.choices(range(self.dataset.n),
                                                      weights=image_weights,
                                                      k=self.dataset.n)
            # Broadcast if DDP
            if self.rank != -1:
                indices = torch.zeros([self.dataset.n], dtype=torch.int)
                if self.rank == 0:
                    indices[:] = torch.from_tensor(self.dataset.indices,
                                                   dtype=torch.int)
                dist.broadcast(indices, 0)
                if self.rank != 0:
                    self.dataset.indices = indices.cpu().numpy()

        mloss = torch.zeros(4, device=self.device)
        if self.rank != -1:
            self.train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(self.train_loader)
        if self.rank in [-1, 0]:
            print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU',
                                         'obj', 'cls', 'total',
                                         'targets', 'img_size'))
            pbar = tqdm(pbar, total=self.nb)    # progress bar
        self.optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:
            # batch ---------------------------------------------------------
            # number integrated batches (since train start)
            ni = i + self.nb * epoch
            # uint8 to float32, 0-255 to 0.0-1.0
            imgs = imgs.to(self.device, non_blocking=True).float() / 255.0

            # Warmup
            if ni <= self.nw:
                xi = [0, self.nw]   # x interp
                self.accumulate = \
                    max(1, np.interp(ni, xi, [1, self.nbs /
                                     self.total_batch_size]).round())
                for j, x in enumerate(self.optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0,
                    # all other lrs rise from 0.0 to lr0
                    x['lr'] = \
                        np.interp(ni, xi, [0.1 if j == 2 else 0.0,
                                           x['initial_lr'] * self.lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi,
                                                  [0.9, self.hyp['momentum']])

            # Multi-scale
            if self.args['multi_scale']:
                sz = random.randrange(self.image_size * 0.5,
                                      self.image_size * 0.5 + self.gs) \
                                      // self.gs * self.gs  # size
                sf = sz / max(imgs.shape[2:])   # scale factor
                if sf != 1:
                    # new shape (stretched to gs-multiple)
                    ns = [math.ceil(x * sf / self.gs) *
                          self.gs for x in imgs.shape[2:]]
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear',
                                         align_corners=False)

            # Autocast
            with amp.autocast(enabled=self.cuda):
                # Forward
                pred = self.model(imgs)
                # Loss
                loss, loss_items = self.loss(pred, targets.to(self.device),
                                             self.model)
                if self.rank != -1:
                    # gradient averaged between devices in DDP mode
                    loss *= self.args['world_size']

            # Backward
            # loss.backward()
            self.scaler.scale(loss).backward()

            # Optimize
            if ni % self.accumulate == 0:
                self.scaler.step(self.optimizer)    # optimizer.step
                self.scaler.update()
                self.optimizer.zero_grad()
                if self.ema is not None:
                    self.ema.update(self.model)

            # Print
            if self.rank in [-1, 0]:
                # update mean losses
                mloss = (mloss * i + loss_items) / (i + 1)
                mem = '%.3gG' % \
                    (torch.cuda.memory_reserved(device=self.device) /
                     1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % \
                    ('%g/%g' % (epoch, self.epochs - 1), mem, *mloss,
                     targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # Plot
                if ni < 3:
                    # filename
                    f = str(self.log_dir / ('train_batch%g.jpg' % ni))
                    result = plot_images(images=imgs, targets=targets,
                                         paths=paths, fname=f)
                    if self.writer and result is not None:
                        self.writer.add_image(f, result, dataformats='HWC',
                                              global_step=epoch)
                        # add model to tensorboard
                        # self.writer.add_graph(self.model, imgs)
            # end batch ----------------------------------------------------

        # Scheduler
        self.scheduler.step()

        # DDP process 0 or single-GPU
        if self.rank in [-1, 0]:
            # mAP
            if self.ema is not None:
                self.ema.update_attr(self.model,
                                     include=['yaml', 'nc', 'hyp',
                                              'gr', 'names', 'stride'])
            final_epoch = epoch + 1 == self.epochs
            if not self.args['notest'] or final_epoch:  # Calculate mAP
                self.results, self.maps, _ = \
                    test.test(self.args['data_cfg'],
                              batch_size=self.batch_size,
                              imgsz=self.image_size_test,
                              save_json=final_epoch and
                              self.args['data_cfg'].endswith(os.sep +
                                                             'coco.yaml'),
                              model=self.ema.ema.module
                              if hasattr(self.ema.ema, 'module')
                              else self.ema.ema,
                              single_cls=self.args['single_cls'],
                              dataloader=self.test_loader,
                              save_dir=self.log_dir)

            # Write
            with open(self.results_file, 'a') as f:
                # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
                f.write(s + '%10.4g' * 7 % self.results + '\n')

            # Tensorboard
            if self.writer:
                tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                        'metrics/precision', 'metrics/recall',
                        'metrics/mAP_0.5', 'metrics/mAP_0.5_0.95',
                        'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
                for x, tag in zip(list(mloss[:-1]) + list(self.results), tags):
                    self.writer.add_scalar(tag, x, epoch)

            # Update best mAP
            # fitness_i = weighted combination of [P, R, mAP, F1]
            fi = fitness(np.array(self.results).reshape(1, -1))
            if fi > self.best_fitness:
                self.best_fitness = fi

            # Save model
            save = (not self.args['nosave']) or final_epoch
            if save:
                with open(self.results_file, 'r') as f:
                    # create checkpoint
                    ckpt = {'epoch': epoch,
                            'best_fitness': self.best_fitness,
                            'training_results': f.read(),
                            'model': self.ema.ema.module
                            if hasattr(self.ema, 'module')
                            else self.ema.ema,
                            'optimizer': None if final_epoch
                            else self.optimizer.state_dict()}
                # Save last, best and delete
                torch.save(ckpt, self.last)
                if epoch >= (self.epochs-5):
                    torch.save(ckpt,
                               self.last.replace('.pt',
                                                 '_{:03d}.pt'.format(epoch)))
                if self.best_fitness == fi:
                    torch.save(ckpt, self.best)
                del ckpt
        # end epoch --------------------------------------------------------
