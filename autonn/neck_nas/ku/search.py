'''A module for search'''

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from nni.retiarii.oneshot.interface import BaseOneShotTrainer
from nni.retiarii.oneshot.pytorch.utils \
    import replace_layer_choice, replace_input_choice
from nni.retiarii.fixed import fixed_arch

# Scaled YOLOv4 trainer
from pathlib import Path
import os
import random
import time
import math
import numpy as np

import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP

import yaml
import test    # import test.py to get mAP after each epoch
from model import SearchYolov4
from syolo_utils.general \
    import (labels_to_class_weights, labels_to_image_weights,
            check_anchors, plot_images, fitness, strip_optimizer,
            plot_results, plot_labels)
from syolo_utils.torch_utils \
    import init_seeds, ModelEMA, intersect_dicts, is_parallel
from tqdm import tqdm


_logger = logging.getLogger(__name__)


class ArchGradientFunction(torch.autograd.Function):
    '''A class for updating gradients of architectural params'''
    @staticmethod
    def forward(ctx, x, binary_gates, run_func, backward_func):
        ctx.run_func = run_func
        ctx.backward_func = backward_func

        detached_x = x.detach()
        detached_x.requires_grad = x.requires_grad
        with torch.enable_grad():
            output = run_func(detached_x)
        ctx.save_for_backward(detached_x, output)
        return output.data

    @staticmethod
    def backward(ctx, grad_output):
        detached_x, output = ctx.saved_tensors

        grad_x = torch.autograd.grad(output, detached_x, grad_output,
                                     only_inputs=True)
        # compute gradients w.r.t. binary_gates
        binary_grads = ctx.backward_func(detached_x.data, output.data,
                                         grad_output.data)

        return grad_x[0], binary_grads, None, None


class ProxylessLayerChoice(nn.Module):
    '''A class for layer choice with ProxylessNAS'''
    def __init__(self, ops):
        super(ProxylessLayerChoice, self).__init__()
        self.ops = nn.ModuleList(ops)
        self.alpha = nn.Parameter(torch.randn(len(self.ops)) * 1E-3)
        self._binary_gates = nn.Parameter(torch.randn(len(self.ops)) * 1E-3)
        self.sampled = None

    def forward(self, *args):
        def run_function(ops, active_id):
            def forward(_x):
                return ops[active_id](_x)
            return forward

        def backward_function(ops, active_id, binary_gates):
            def backward(_x, _output, grad_output):
                binary_grads = torch.zeros_like(binary_gates.data)
                with torch.no_grad():
                    for k in range(len(ops)):
                        if k != active_id:
                            out_k = ops[k](_x.data)
                        else:
                            out_k = _output.data
                        grad_k = torch.sum(out_k * grad_output)
                        binary_grads[k] = grad_k
                return binary_grads
            return backward

        assert len(args) == 1
        x = args[0]
        return ArchGradientFunction.apply(
            x, self._binary_gates, run_function(self.ops, self.sampled),
            backward_function(self.ops, self.sampled, self._binary_gates))

    def resample(self):
        probs = F.softmax(self.alpha, dim=-1)
        sample = torch.multinomial(probs, 1)[0].item()
        self.sampled = sample
        with torch.no_grad():
            self._binary_gates.zero_()
            self._binary_gates.grad = torch.zeros_like(self._binary_gates.data)
            self._binary_gates.data[sample] = 1.0

    def finalize_grad(self):
        binary_grads = self._binary_gates.grad
        with torch.no_grad():
            if self.alpha.grad is None:
                self.alpha.grad = torch.zeros_like(self.alpha.data)
            probs = F.softmax(self.alpha, dim=-1)
            for i in range(len(self.ops)):
                for j in range(len(self.ops)):
                    self.alpha.grad[i] += binary_grads[j] * probs[j] \
                                            * (int(i == j) - probs[i])

    def export(self):
        return torch.argmax(self.alpha).item()

    def export_prob(self):
        return F.softmax(self.alpha, dim=-1)


class ProxylessInputChoice(nn.Module):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('Input choice is not supported \
                                    for ProxylessNAS.')


class ProxylessDetTrainer(BaseOneShotTrainer):
    """
    Proxyless trainer.

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
    arc_learning_rate : float
        Learning rate of architecture parameters optimizer.
    device : torch.device
        ``torch.device("cpu")`` or ``torch.device("cuda")``.
    """

    def __init__(self, model, args, hyp, loss, optimizer, imgsz, imgsz_test,
                 train, dataset, test, writer, lf, scheduler,
                 num_epochs, arc_learning_rate=1.0E-3, device=None):
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
        self.test_loader_iterator = iter(self.test_loader)
        self.writer = writer
        self.lf = lf
        self.scheduler = scheduler
        self.epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available()
                                   else 'cpu') if device is None else device
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
        with open(args['data_cfg']) as f:
            data_dict = yaml.load(f, Loader=yaml.FullLoader)

        # number classes, names
        self.nc, self.names = (1, ['item']) if self.args['single_cls'] \
            else (int(data_dict['nc']), data_dict['names'])
        assert len(self.names) == self.nc, \
            '%g names found for nc=%g dataset in %s' % \
            (len(self.names), self.nc, self.args['data_cfg'])  # check

        self.model.to(self.device)
        self.nas_modules = []
        replace_layer_choice(self.model, ProxylessLayerChoice,
                             self.nas_modules)
        replace_input_choice(self.model, ProxylessInputChoice,
                             self.nas_modules)
        for _, module in self.nas_modules:
            module.to(self.device)

        # we do not support deduplicate control parameters
        # with same label (like DARTS) yet.
        self.ctrl_optim = \
            torch.optim.Adam([m.alpha for _, m in self.nas_modules],
                             arc_learning_rate, weight_decay=0,
                             betas=(0, 0.999), eps=1e-8)
        # Resume
        pretrained = self.args['weights'].endswith('.pt')
        self.start_epoch, self.best_fitness = 0, 0.0
        if pretrained:
            ckpt = torch.load(self.args['weights'], map_location=self.device)

            # Load model
            # exclude keys
            exclude = ['anchor'] if self.args['model_cfg'] else []
            state_dict = ckpt['model'].float().state_dict()     # to FP32
            state_dict = intersect_dicts(state_dict, self.model.state_dict(),
                                         exclude=exclude)  # intersect
            self.model.load_state_dict(state_dict, strict=False)    # load
            print('Transferred %g/%g items from %s' %
                  (len(state_dict), len(self.model.state_dict()),
                   self.args['weights']))    # report

            # Load optimizer
            if ckpt['optimizer'] is not None:
                self.optimizer.load_state_dict(ckpt['optimizer'])
                self.best_fitness = ckpt['best_fitness']

            # Load results
            if ckpt.get('training_results') is not None:
                with open(self.results_file, 'w') as file:
                    file.write(ckpt['training_results'])    # write results.txt

            # Load epochs
            self.start_epoch = ckpt['epoch'] + 1
            if self.epochs < self.start_epoch:
                print('%s has been trained for %g epochs. Fine-tuning for %g \
                      additional epochs.' %
                      (self.args['weights'], ckpt['epoch'], self.epochs))
                self.epochs += ckpt['epoch']    # finetune additional epochs

            # Load ctrl_optimizer
            if self.args['resume'] and ckpt['ctrl_optimizer'] is not None:
                self.ctrl_optim.load_state_dict(ckpt['ctrl_optimizer'])
            del ckpt, state_dict

        # DP mode
        if self.cuda and self.rank == -1 and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        # SyncBatchNorm
        convert_sync_batchnorm = nn.SyncBatchNorm.convert_sync_batchnorm
        if self.args['sync_bn'] and self.cuda and self.rank != -1:
            self.model = \
                convert_sync_batchnorm(self.model).to(self.device)
            print('Using SyncBatchNorm()')

        # Exponential moving average
        self.ema = ModelEMA(self.model) if self.rank in [-1, 0] else None
        self.nb = len(self.train_loader)    # number of batches
        self.nbs = 64
        self.accumulate = max(round(self.nbs / self.total_batch_size), 1)
        if self.rank in [-1, 0]:
            self.ema.updates = self.start_epoch * self.nb \
                                // self.accumulate    # set EMA updates ***

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
        self.maps = np.zeros(self.nc)    # mAP per class
        # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Obj', 'val Cls'
        self.results = (0, 0, 0, 0, 0, 0, 0)
        self.scheduler.last_epoch = self.start_epoch - 1    # do not move
        if self.rank in [0, -1]:
            print('Image sizes %g train, %g test' %
                  (self.image_size, self.image_size_test))
            print('Using %g dataloader workers' % self.num_workers)
            print('Starting search for %g epochs...' % self.epochs)

    def _train_one_epoch(self, epoch):
        self.model.train()

        # save current architecture prob for tracking updates
        probs_history = {}
        export_architecture = {}
        for module_name, module in self.nas_modules:
            probs = module.export_prob()
            probs_history[module_name] = probs.detach().cpu().tolist()
            export_architecture[module_name] = module.export()
        with open(self.log_dir / 'probs_history.txt', 'a') as f:
            print({str(epoch): probs_history}, file=f)
        with open(self.log_dir / 'export_history.txt', 'a') as f:
            print({str(epoch): export_architecture}, file=f)

        # Update image weights (optional)
        if self.dataset.image_weights:
            # Generate indices
            if self.rank in [-1, 0]:
                w = self.model.class_weights.cpu().numpy() * \
                    (1 - self.maps) ** 2  # class weights
                image_weights = labels_to_image_weights(self.dataset.labels,
                                                        nc=self.nc,
                                                        class_weights=w)
                self.dataset.indices = \
                    random.choices(range(self.dataset.n),
                                   weights=image_weights,
                                   k=self.dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if self.rank != -1:
                indices = torch.zeros([self.dataset.n], dtype=torch.int)
                if self.rank == 0:
                    indices[:] = \
                        torch.from_tensor(self.dataset.indices,
                                          dtype=torch.int)
                dist.broadcast(indices, 0)
                if self.rank != 0:
                    self.dataset.indices = indices.cpu().numpy()

        mloss = torch.zeros(4, device=self.device)
        if self.rank != -1:
            self.train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(self.train_loader)
        if self.rank in [-1, 0]:
            print(('\n' + '%10s' * 8) %
                  ('Epoch', 'gpu_mem', 'GIoU', 'obj',
                  'cls', 'total', 'targets', 'img_size'))
            pbar = tqdm(pbar, total=self.nb)    # progress bar

        self.optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:
            # batch -------------------------------------------------------
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

            if ni > self.nw:
                # 1) train architecture parameters
                for _, module in self.nas_modules:
                    module.resample()
                self.ctrl_optim.zero_grad()
                try:    # len(test_loader) < len(train_loader)
                    imgs_test, targets_test, _, _ = \
                        next(self.test_loader_iterator)
                except StopIteration:
                    self.test_loader_iterator = \
                        iter(self.test_loader)
                    imgs_test, targets_test, _, _ = \
                        next(self.test_loader_iterator)
                imgs_test = imgs_test.to(self.device,
                                         non_blocking=True).float() / 255.0
                targets_test = targets_test.to(self.device)
                loss, _ = self._loss_and_items_for_arch_update(imgs_test,
                                                               targets_test)
                loss.backward()
                for _, module in self.nas_modules:
                    module.finalize_grad()
                self.ctrl_optim.step()

            # 2) train model parameters
            for module_name, module in self.nas_modules:
                module.resample()
            loss, loss_items = \
                self._loss_and_items_for_weight_update(imgs,
                                                       targets.to(self.device))
            loss.backward()

            # Optimize
            if ni % self.accumulate == 0:
                self.optimizer.step()
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
            # end batch ---------------------------------------------------

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

                # to fix architecture for test
                arch = dict()
                for name, module in self.nas_modules:
                    if name not in arch:
                        arch[name] = module.export()
                with fixed_arch(arch):
                    self.model_test = SearchYolov4(cfg=self.args['model_cfg'],
                                                   names=self.names,
                                                   hyp=self.hyp,
                                                   ch=3,
                                                   nc=self.nc).to(self.device)
                ckpt_ema = self.ema.ema.module.state_dict().copy() \
                    if is_parallel(self.ema.ema) \
                    else self.ema.ema.state_dict().copy()

                # to match keys with ema's state_dict
                for k, v in arch.items():
                    m_num = k.split('_')[-1]
                    op_num = str(v)
                    for key in list(ckpt_ema.keys()):
                        ks = key.split('.')
                        if 'ops' not in ks:
                            continue
                        if m_num == ks[1] and op_num == ks[4]:
                            new_key = ks[:3] + ks[5:]
                            new_key = '.'.join(new_key)
                            ckpt_ema[new_key] = ckpt_ema.pop(key)

                exclude = []
                state_dict = intersect_dicts(ckpt_ema,
                                             self.model_test.state_dict(),
                                             exclude=exclude)   # intersect
                self.model_test.load_state_dict(state_dict,
                                                strict=False)   # load

                self.results, self.maps, _ = \
                    test.test(self.args['data_cfg'],
                              batch_size=self.batch_size,
                              imgsz=self.image_size_test,
                              save_json=final_epoch and
                              self.args['data_cfg'].endswith(os.sep +
                                                             'coco.yaml'),
                              model=self.model_test,
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
            save = not self.args['nosave']
            if save:
                with open(self.results_file, 'r') as f:  # create checkpoint
                    ckpt = {'epoch': epoch,
                            'best_fitness': self.best_fitness,
                            'training_results': f.read(),
                            'model': self.ema.ema.module
                            if hasattr(self.ema.ema, 'module')
                            else self.ema.ema,
                            # 'model': self.model_test,
                            'optimizer': None if final_epoch
                            else self.optimizer.state_dict(),
                            'ctrl_optimizer': None if final_epoch
                            else self.ctrl_optim.state_dict()}

                # Save last, best and delete
                torch.save(ckpt, self.last)
                if epoch >= (self.epochs-5):
                    torch.save(ckpt,
                               self.last.replace('.pt',
                                                 '_{:03d}.pt'.format(epoch)))
                if self.best_fitness == fi:
                    torch.save(ckpt, self.best)
                del ckpt
        # end epoch ---------------------------------------------------------

    def _loss_and_items_for_arch_update(self, imgs, targets):
        ''' return loss and loss_items for architecture parameter update '''
        pred = self.model(imgs)
        loss, loss_items = self.loss(pred, targets, self.model)
        if self.rank != -1:
            # gradient averaged between devices in DDP mode
            loss *= self.args['world_size']
        return loss, loss_items

    def _loss_and_items_for_weight_update(self, imgs, targets):
        ''' return loss and loss_items for weight parameter update '''
        pred = self.model(imgs)
        loss, loss_items = self.loss(pred, targets, self.model)
        if self.rank != -1:
            # gradient averaged between devices in DDP mode
            loss *= self.args['world_size']
        return loss, loss_items

    def fit(self):
        t0 = time.time()
        for epoch in range(self.start_epoch, self.epochs):
            self._train_one_epoch(epoch)

        if self.rank in [-1, 0]:
            # Strip optimizers
            n = ('_' if len(self.args['name']) and not
                 self.args['name'].isnumeric() else '') + self.args['name']
            fresults, flast, fbest = \
                f'results{n}.txt', self.wdir + 'last{n}.pt', \
                self.wdir + f'best{n}.pt'
            for f1, f2 in zip([self.wdir + 'last.pt', self.wdir +
                               'best.pt', 'results.txt'],
                              [flast, fbest, fresults]):
                if os.path.exists(f1):
                    os.rename(f1, f2)  # rename
                    ispt = f2.endswith('.pt')  # is *.pt
                    if ispt:    # strip optimizer
                        strip_optimizer(f2, f2.replace('.pt', '_strip.pt'))
                    else:
                        None
            # Finish
            plot_results(save_dir=self.log_dir)  # save as results.png
            print('%g epochs completed in %.3f hours.\n' %
                  (self.epochs - self.start_epoch + 1,
                   (time.time() - t0) / 3600))

        dist.destroy_process_group() if self.rank not in [-1, 0] else None
        torch.cuda.empty_cache()
        # return self.results

    @torch.no_grad()
    def export(self):
        result = dict()
        for name, module in self.nas_modules:
            if name not in result:
                result[name] = module.export()
        return result
