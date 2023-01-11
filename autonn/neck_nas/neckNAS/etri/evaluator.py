import os
import sys
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
from datetime import datetime

import torch
# import torch.nn as nn
# import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader, Dataset, dataloader, distributed

# import nni.retiarii.nn.pytorch as nn  # (use original pytorch)
from nni.retiarii.oneshot.interface import BaseOneShotTrainer
from nni.retiarii.oneshot.pytorch.utils import to_device

import val
from datasets import LoadImagesAndLabels
from yolov5_utils.general import (LOGGER, colorstr, init_seeds,
                                  one_cycle, linear)
from yolov5_utils.torch_utils import (de_parallel, ModelEMA, EarlyStopping)
from yolov5_utils.metrics import fitness
# from yolov5_utils.loss import ComputeLoss

FILE = Path(__file__).resolve()  # absolute file path
ROOT = FILE.parents[0]  # absolute directory path
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# https://pytorch.org/docs/stable/elastic/run.html
LOCAL_RANK = int(os.getenv('LOCAL_RANK', '-1'))
RANK = int(os.getenv('RANK', '-1'))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', '1'))


class ConcatBasedNetTrainer(BaseOneShotTrainer):
    """
    Concatenation-based Network trainer (ref.ProxylessNAS trainer)

    Parameters
    ----------
    model : nn.Module
        PyTorch model to be trained.
    loss : callable
        Receives logits and ground truth label, return a loss tensor.
    optimizer : Optimizer
        The optimizer used for optimizing the model.
    lf : lambda function
        Lambda function for scheduling learning rate.
    scheduler : torch.optim.lr_scheduler.LambdaR
        The scheduler used for scheduling learning rate.
    num_epochs : int
        Number of epochs planned for training.
    dataset : LoadImagesAndLabels
        Dataset for training model & architecture.
    img_size : int
        Image size for train.
    batch_size : int
        Batch size.
    workers : int
        Workers for data loading.
    device : torch.device
        ``torch.device("cpu")`` or ``torch.device("cuda")``.
    log_frequency : int
        Step count per logging.
    arc_lr : float
        Learning rate of architecture parameters.
    """
    def __init__(self, model, loss, optimizer, lf, scheduler,
                 num_epochs, dataset, train_data, val_data, img_size,
                 batch_size=64, workers=4, device=None, log_frequency=None,
                 arc_lr=1.0E-3, save_dir=None):

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.lf = lf
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.dataset = dataset
        self.train_data = train_data
        self.val_data = val_data
        self.img_size = img_size
        self.batch_size = batch_size
        self.workers = workers
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu') if device is None \
            else device
        self.log_frequency = log_frequency

        # --------- ref. YOLO v.5 trainer -------------------------------------
        # directories ---------------------------------------------------------
        self.save_dir = Path(save_dir)
        self.weight_dir = self.save_dir / 'weights'
        # self.w = self.weight_dir.parent if args.evolve else self.weight_dir
        self.weight_dir.mkdir(parents=True, exist_ok=True)
        self.last = self.weight_dir / 'lastmodel.pt'
        self.best = self.weight_dir / 'bestmodel.pt'

        # hyperparements ------------------------------------------------------
        self.hyp = self.model.hyp
        # LOGGER.info(colorstr('hyperparameters: ') \
        #     + ', '.join(f'{k}={v}' for k, v in self.hyp.items()))

        # save run settings ---------------------------------------------------
        # with open(Path(save_dir) / 'hyp.yaml', 'w') as f:
        #     yaml.safe_dump(self.hyp, f, sort_keys=False)
        # with open(Path(save_dir) / 'args.yaml', 'w') as f:
        #     yaml.safe_dump(vars(args), f, sort_keys=False)

        # configurations from CLI(args)
        # self.args = args

        # create plots
        # self.plots = not self.args.evolve and not self.args.noplots

        # whether cuda is used or not
        self.cuda = self.device.type != 'cpu'

        # random seed
        init_seeds(1 + RANK)

        # Resume
        self.start_epoch, self.best_fitness = 0, 0.0

        # number of batches
        self.nb = len(self.train_data)

        # number of warmup iterations, max(3 epochs, 1000 iterations)
        self.nw = max(round(self.hyp['warmup_epochs'] * self.nb), 100)

        # optimizing step
        self.last_opt_step = -1

        # mAP per class
        self.maps = np.zeros(self.model.nc)

        # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        self.results = (0, 0, 0, 0, 0, 0, 0)

        self.scheduler.last_epoch = self.start_epoch - 1  # do not move

        # (amp) automatic mixed precision
        # https://pytorch.org/docs/stable/amp.html
        self.scaler = amp.GradScaler(enabled=self.cuda)

        # early stoopping (needs args) : TODO
        # args_patience = 100
        # self.stopper = EarlyStopping(patience=args_patience)

        # (ema) exponential moving average
        self.ema = ModelEMA(self.model) if RANK in [-1, 0] else None

        # --------- ref. ProxylessNAS trainer ---------------------------------
        # self.model.to(self.device)
        # self.nas_modules = []
        # replace_layer_choice(self.model,
        #                      ProxylessLayerChoice,
        #                      self.nas_modules)
        # replace_input_choice(self.model,
        #                      ProxylessInputChoice,
        #                      self.nas_modules)
        # for _, module in self.nas_modules:
        #     module.to(self.device)

        # optimizer for architecure parameters
        #   we do not support deduplicate control parameters
        #   with same label (like DARTS) yet.
        # self.ctrl_optim = torch.optim.Adam(
        #     [m.alpha for _, m in self.nas_modules],
        #     arc_lr,
        #     weight_decay=0,
        #     betas=(0, 0.999),
        #     eps=1e-8
        # )

        # optimizer for model parameters (weights and biases)
        # it is duplicate (TODO: see optimizer part in main.py)
        self.nbs = 64
        self.accumulate = max(round(self.nbs / self.batch_size), 1)

        # self._init_dataloader() # (-> main.py)

    def _init_dataloader(self):
        n_train = len(self.dataset)
        split = n_train // 2
        indices = list(range(n_train))
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(
            indices[:split]
        )
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(
            indices[split:]
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.workers
        )
        self.valid_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=valid_sampler,
            num_workers=self.workers
        )

    def _train_one_epoch(self, epoch):
        self.model.train()

        self.mloss = torch.zeros(3, device=self.device)  # mean losses
        if RANK != -1:
            self.train_data.sampler.set_epoch(epoch)

        pbar = enumerate(self.train_data)
        LOGGER.info(
            ('\n' + '%10s' * 7)
            % ('epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size')
        )
        if RANK in (-1, 0):
            pbar = tqdm(pbar, total=self.nb,
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

        self.optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:
            # batch -----------------------------------------------------------
            # number integrated batches (since train start)
            ni = i + self.nb * epoch
            imgs = imgs.to(self.device, non_blocking=True).float() / 255.0
            # targets = targets.to(self.device)

            # warmup ----------------------------------------------------------
            if ni <= self.nw:
                xi = [0, self.nw]  # x interp
                # interp 'accumulate'
                self.accumulate = max(
                    1, np.interp(ni, xi,
                                 [1, self.nbs / self.batch_size]).round())
                for j, x in enumerate(self.optimizer.param_groups):
                    # interp 'lr'
                    # bias learning rate falls from 0.1 to lr0,
                    # all other learning rates rise from 0.0 to lr0
                    x['lr'] = np.interp(
                        ni, xi,
                        [self.hyp['warmup_bias_lr'] if j == 2 else 0.0,
                         x['initial_lr'] * self.lf(epoch)])
                    # interp 'momentum'
                    if 'momentum' in x:
                        x['momentum'] = np.interp(
                            ni, xi,
                            [self.hyp['warmup_momentum'],
                             self.hyp['momentum']])

            # multi-scale (needs args and grid size) --------------------------
            # if args.multi_scale:
            #     sz = random.randrange(self.img_size * 0.5,
            #                           self.img_size * 1.5 + gs) // gs * gs
            #     sf = sz / max(imgs.shape[2:])  # scale factor
            #     if sf != 1:
            #         # new shape (stretched to gs-multiple)
            #         ns = [math.ceil(x * sf / gs) * gs
            #               for x in imgs.shape[2:]]
            #         imgs = nn.functional.interpolate(imgs, size=ns,
            #                                          mode='bilinear',
            #                                          align_corners=False)

            # 1) train architecture params (w/val_data) -----------------------
            if ni > self.nw:
                pass
                # print('\ttrain & update architecture parameters')
            #     for _, module in self.nas_modules:
            #         module.resample()
            #     # self.ctrl_optim.zero_grad()
            #     # forward
            #     # backward
            #     # optimize

            # 2) train model weights and biases (w/train_data) ----------------
            # for _, module in self.nas_modules:
            #     module.resample()
            # self.optimizer.zero_grad() <-- (tenace: got you!! it's buggy)

            # forward
            loss, loss_items = \
                self._loss_and_items_for_weight_update(imgs,
                                                       targets)

            # backward
            self.scaler.scale(loss).backward()

            # optimize (init gradients to zero per 'accumulate' )
            # if ni % self.accumulate == 0:
            if ni - self.last_opt_step >= self.accumulate:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                if self.ema:
                    self.ema.update(self.model)
                self.last_opt_step = ni

            # log -------------------------------------------------------------
            if RANK in (-1, 0):
                # mean losses
                self.mloss = (self.mloss * i + loss_items) / (i + 1)
                gb = 0
                if torch.cuda.is_available():
                    gb = torch.cuda.memory_reserved() / 1E9
                else:
                    gb = 0
                mem = f'{gb:.3g}G'  # cuda memory (GB)
                pbar.set_description(
                    ('%10s' * 2 + '%10.4g' * 5)
                    % (f'{epoch}/{self.num_epochs - 1}',    # epoch
                       mem,                                 # gpu mem
                       *self.mloss,                         # box, obj, cls
                       targets.shape[0],
                       imgs.shape[-1]))
            # end batch -------------------------------------------------------

    def _validate(self, epoch):
        # scheduler
        self.lr = [x['lr'] for x in self.optimizer.param_groups]  # for loggers
        self.scheduler.step()

        if RANK in (-1, 0):
            self.ema.update_attr(self.model,
                                 include=['yaml',
                                          'nc',
                                          'hyp',
                                          'names',
                                          'stride',
                                          'class_weights'])

            # final_epoch = (epoch + 1 == self.num_epochs) \
            #     or self.stopper.possible_stop
            final_epoch = (epoch + 1 == self.num_epochs)

            self_noval = False  # save every epoch
            self_nosave, self_evolve = False, False
            if not self_noval or final_epoch:
                # calculate mAP -----------------------------------------------
                self.results, self.maps, _ = val.run(
                    batch_size=self.batch_size // WORLD_SIZE * 2,
                    imgsz=self.img_size,
                    save_dir=Path(self.save_dir),
                    # half=False,
                    model=self.ema.ema,  # self.model,
                    dataloader=self.val_data,
                    compute_loss=self.loss)
                # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                fi = fitness(np.array(self.results).reshape(1, -1))

                # log ---------------------------------------------------------
                # lmloss, lresults = [], []
                # for _loss in self.mloss:
                #     lmloss.append(_loss.item())  # accumulated mean loss

                # # mean loss, current result, current learning rate
                # log_vals = lmloss + list(self.results) + self.lr
                # LOGGER.info(
                #     colorstr("bright_green", f'on fit epoch end:')
                #     + f' {log_vals}, best fitness={self.best_fitness},'
                #     f' current fitness={fi}')

                # save model --------------------------------------------------
                if (not self_nosave) or (final_epoch and not self_evolve):
                    self._log_n_save(epoch, fi)

                # update best mAP ---------------------------------------------
                # if fi > self.best_fitness:
                #     self.best_fitness = fi

    def _log_n_save(self, epoch, fi):
        ckpt = {
            'epoch': epoch,
            'best_fitness': fi
            if fi > self.best_fitness else self.best_fitness,
            'model': deepcopy(de_parallel(self.model)),
            'ema': deepcopy(self.ema.ema),
            'updates': self.ema.updates,
            'optimizer': self.optimizer.state_dict(),
            'wandb_id': None,
            'date': datetime.now().isoformat()}

        # torch.save(ckpt, self.last)
        if epoch == 0 or self.best_fitness < fi:
            torch.save(ckpt, self.best)
            LOGGER.info(
                colorstr("bright_green", f'save the best model:')
                + f' path = {self.best} epoch = {epoch} '
                f'previous best = {self.best_fitness} current = {fi}')
            self.best_fitness = fi
        if ((epoch > 0) and (self.log_frequency > 0)
                and (((epoch + 1) % self.log_frequency == 0))):
            # torch.save(ckpt, self.weight_dir / f'epoch{epoch}.pt')
            torch.save(ckpt, self.last)
            LOGGER.info(
                colorstr("bright_yellow", f'save the last model:')
                + f' path = {self.last} epoch = {epoch} '
                f'best = {self.best_fitness} current = {fi}')
        del ckpt

    def _loss_and_items_for_weight_update(self, imgs, targets):
        ''' return loss and loss_items for weight update '''
        with amp.autocast(enabled=self.cuda):
            pred = self.model(imgs)  # 3 x b x 3 x h x w x 85
            compute_loss = self.loss  # ComputeLoss(self.model)
            loss, loss_items = compute_loss(pred, targets.to(self.device))
            # print(f"\ttotal loss: {loss.item():0.3f},"
            #       f" [box, obj, class] = [{loss_items[0].item():0.3f}, "
            #       f"{loss_items[1].item():0.3f}, "
            #       f"{loss_items[2].item():0.3f}]")
            # gradient averaged between devices in DDP mode
            if RANK != -1:
                loss *= WORLD_SIZE
            # if args.quad:
            #     loss *= 4
        return loss, loss_items

    def fit(self):
        LOGGER.info(f"Image sizes {self.img_size} train, {self.img_size} val\n"
                    f"Using {self.train_data.num_workers * WORLD_SIZE} "
                    f"dataloader workers\n"
                    f"Logging results to {colorstr('bold',self.save_dir)}\n"
                    f"Starting training for {self.num_epochs} epoches...")

        t0 = time.time()
        self.scheduler.last_epoch = self.start_epoch - 1
        for i in range(self.start_epoch, self.num_epochs):
            self._train_one_epoch(i)
            self._validate(i)
            # if RANK == -1 and self.stopper(epoch=i, fitness=fi):
            #     break
        s = f'\n{self.num_epochs - self.start_epoch} epochs completed'\
            f' in {(time.time() - t0) / 3600:.3f} hours.'
        LOGGER.info(colorstr("bold", s))
        torch.cuda.empty_cache()

    @torch.no_grad()
    def export(self):
        res = dict()
        # for name, module in self.nas_modules:
        #     if name not in res:
        #         res[name] = module.export()
        return res


class ConcatEntirePathTrainer(BaseOneShotTrainer):
    def __init__(self, model, args, hyp, loss, optimizer, lf, scheduler,
                 num_epochs, dataset, train_data, val_data, img_size,
                 batch_size=64, workers=4, device=None, log_freq=None,
                 warmup_epochs=5, arc_lr=1.0E-3, search_type='one_stage'):
        super(ConcatEntirePathTrainer, self).__init__()

        self.model = model
        self.args = args
        self.hyp = hyp
        self.loss = loss
        self.optimizer = optimizer
        self.lf = lf
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.dataset = dataset
        self.train_data = train_data
        self.val_data = val_data
        self.img_size = img_size
        self.batch_size = batch_size
        self.workers = workers
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu') if device is None \
            else device
        self.log_frequency = log_freq

        # ref. YOLO v.5 trainer
        # directories ---------------------------------------------------------
        self.weight_dir = Path(args.save_dir) / 'weights'
        self.w = self.weight_dir.parent if args.evolve else self.weight_dir
        self.w.mkdir(parents=True, exist_ok=True)
        self.last_dir = self.weight_dir / 'last.pt'
        self.best_dir = self.weight_dir / 'best.pt'

        # cuda ----------------------------------------------------------------
        self.cuda = self.device.type != 'cpu'

        # random seed ---------------------------------------------------------
        init_seeds(1 + RANK)

        # resume --------------------------------------------------------------
        self.start_epoch, self.best_fitness = 0, 0.0
        # TODO: need to codes for resuming train

        # number of batch -----------------------------------------------------
        self.nb = len(self.train_data)

        # warmup batches ------------------------------------------------------
        self.nw = \
            max(round(self.hyp['warmup_epochs'] * self.nb), 100)

        # mAP -----------------------------------------------------------------
        self.maps = np.zeros(self.model.nc)

        # P, R, mAP@0.5, mAP@.5:.95, val_loss(box, obj, cls) ------------------
        self.results = (0, 0, 0, 0, 0, 0, 0)

        # self.model.to(self.device)  # (tenace comment: bug!!)
        # self.optimizer = optimizer  # (tenace comment: duplicate)

        # (amp) automatic mixed precision -------------------------------------
        self.scaler = amp.GradScaler(enabled=self.cuda)

        # (ema) exponential moving average ------------------------------------
        self.ema = None  # ModelEMA(self.model) if RANK in [-1, 0] else None

        self.search_type = search_type
        if search_type == 'one_stage':
            self._search_method = self._one_stage_search
            # (tenace comment) it was already applied at main.py
            # self.optimizer.add_param_group(
            #     {'params': self.model.neck_module.return_list}
            # )
        elif search_type == 'two_stage':
            self._search_method = self._two_stage_search
            # optimizer for arch. params --------------------------------------
            # self.ctrl_optim = torch.optim.Adam(
            #     self.model.neck_module.return_list,
            #     lr=self.hyp['lr0'],
            #     weight_decay=self.hyp['weight_decay']
            # )
            self.ctrl_optim = torch.optim.SGD(
                self.model.neck_module.return_list,
                lr=arc_lr,  # TODO: replace self.hyp['arc_lr]
                momentum=self.hyp['momentum'],
                weight_decay=self.hyp['weight_decay']
            )
            # scheduler for arch. params --------------------------------------
            if args.cos_lr:
                lf = one_cycle(1, hyp['lrf'], args.epochs)
            else:
                lf = linear(hyp['lrf'], args.epochs)
            self.ctrl_sched = torch.optim.lr_scheduler.LambdaLR(
                self.ctrl_optim, lr_lambda=self.lf)
            # data loader for arch. params ------------------------------------
            self._init_dataloader()
        elif search_type == 'retrain':
            self._search_method = self._one_stage_search

    def _init_dataloader(self):
        n_train = len(self.dataset)
        split = n_train // 2
        indices = list(range(n_train))
        train_sampler1 = torch.utils.data.sampler.SubsetRandomSampler(
            indices[:split]
        )
        train_sampler2 = torch.utils.data.sampler.SubsetRandomSampler(
            indices[split:]
        )
        loader = DataLoader if True else InfiniteDataLoader
        self.train_loader1 = loader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            sampler=train_sampler1,
            pin_memory=True,
            collate_fn=LoadImagesAndLabels.collate_fn4 if False
            else LoadImagesAndLabels.collate_fn
        )
        self.train_loader2 = loader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            sampler=train_sampler2,
            pin_memory=True,
            collate_fn=LoadImagesAndLabels.collate_fn4 if False
            else LoadImagesAndLabels.collate_fn)

    def _train_one_epoch(self, epoch):
        if RANK != -1:
            self.train_data.sampler.set_epoch(epoch)
        mloss = self._search_method(epoch)
        LOGGER.debug(f'on train epoch #{epoch} end')
        return mloss

    def _one_stage_search(self, epoch):
        print('1-stage Searching')
        mloss = torch.zeros(3, device=self.device)  # mean losses
        self.model.train()
        print(('\n'+'%10s'*2 + '%10s'*3 + '%10s'*2)
              % ('EPOCH', 'GPU_mem', 'box_loss', 'obj_loss',
              'cls_loss', 'Instances', 'Size'))
        pbar = tqdm(enumerate(self.train_data), total=len(self.train_data),
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for step, (trn_X, trn_y, paths, _) in pbar:
            trn_X, trn_y = to_device(trn_X, self.device).float() / 255., \
                           to_device(trn_y, self.device).float()

            # train model and architecture parameters
            self.optimizer.zero_grad()
            loss, loss_items = \
                self._loss_and_items_for_weight_update(trn_X, trn_y)
            loss.backward()
            self.optimizer.step()

            if RANK in (-1, 0):
                mloss = (mloss * step + loss_items) / (step + 1)  # mean losses
                gb = torch.cuda.memory_reserved() / 1E9 \
                    if torch.cuda.is_available() else 0
                mem = f'{gb:.3g}G'  # cuda memory (GB)
                pbar.set_description(
                    ('%10s'*2 + '%10.4f'*3 + '%10d'*2)
                    % (f'{epoch}/{self.num_epochs - 1}',
                       mem, *mloss, trn_y.shape[0], trn_X.shape[-1])
                )

        return mloss

    def _two_stage_search(self, epoch):
        LOGGER.debug('2-stage Searching')
        if epoch <= self.warmup_epochs:
            LOGGER.info(f' [{epoch}/{self.warmup_epochs}]'
                        f' While Warm-up,'
                        f' Architecture Parameters will not be updated')
        mloss = torch.zeros(3, device=self.device)  # mean losses
        self.model.train()
        LOGGER.info(('\n'+'%10s'*2 + '%10s'*3 + '%10s'*3)
                    % ('EPOCH', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss',
                       'target_1', 'target_2', 'Size'))
        pbar = tqdm(enumerate(zip(self.train_loader1, self.train_loader2)),
                    total=len(self.train_loader1),
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        self.optimizer.zero_grad()
        self.ctrl_optim.zero_grad()
        for step, ((trn_X1, trn_y1, paths1, _), (trn_X2, trn_y2, paths2, __))\
                in pbar:
            nstep = step + self.nb * epoch
            trn_X1, trn_y1 = to_device(trn_X1, self.device).float() / 255.,\
                to_device(trn_y1, self.device).float()
            trn_X2, trn_y2 = to_device(trn_X2, self.device).float() / 255.,\
                to_device(trn_y2, self.device).float()

            if nstep <= self.nw:
                xi = [0, self.nw]  # x interp
                # interp 'accumulate'
                self.accumulate = max(
                    1, np.interp(nstep, xi,
                                 [1, 64 / self.batch_size]).round())
                for j, x in enumerate(self.optimizer.param_groups):
                    # interp 'lr'
                    # bias learning rate falls from 0.1 to lr0,
                    # all other learning rates rise from 0.0 to lr0
                    x['lr'] = np.interp(
                        nstep, xi,
                        [self.hyp['warmup_bias_lr'] if j == 2 else 0.0,
                         x['initial_lr'] * self.lf(epoch)])
                    # interp 'momentum'
                    if 'momentum' in x:
                        x['momentum'] = np.interp(
                            nstep, xi,
                            [self.hyp['warmup_momentum'],
                             self.hyp['momentum']])

            # train architecture parameters -----------------------------------
            loss_items_for_arch = None
            active_path_for_arch = []
            if epoch >= self.warmup_epochs:
                # self.ctrl_optim.zero_grad()
                loss, loss_items_for_arch = \
                    self._loss_and_items_for_weight_update(trn_X1, trn_y1)
                # loss.backward()
                self.scaler.scale(loss).backward()
                # self.ctrl_optim.step()
                self.scaler.step(self.ctrl_optim)
                self.scaler.update()
                self.ctrl_optim.zero_grad()
                if self.ema:
                    self.ema.update(self.model)
                # for name, module in self.model.neck_module.named_modules():
                #     if hasattr(module, 'get_active_path'):
                #         for path in module.get_active_path():
                #             active_path_for_arch.append(path)

            # train model parameters ------------------------------------------
            # self.optimizer.zero_grad()
            loss, loss_items_for_model = \
                self._loss_and_items_for_weight_update(trn_X2, trn_y2)
            # loss.backward()
            self.scaler.scale(loss).backward()
            # self.optimizer.step()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update(self.model)
            active_path_for_model = []
            for name, module in self.model.neck_module.named_modules():
                if hasattr(module, 'get_active_path'):
                    for path in module.get_active_path():
                        active_path_for_model.append(path)

            # validate model parameters ---------------------------------------
            # TODO
            # freeze the bernoulli gate & validate the model (a.k.a subneck)
            # ... do not need the validation over superneck on the end of epoch
            # ... store the sampling result at training phase for model
            # ... never use arch_weights at validation phase

            if RANK in (-1, 0):
                if loss_items_for_arch is not None:
                    loss_items = \
                        (loss_items_for_arch + loss_items_for_model) / 2
                else:
                    loss_items = loss_items_for_model
                mloss = (mloss * step + loss_items) / (step + 1)  # mean losses
                if torch.cuda.is_available():
                    gb = torch.cuda.memory_reserved() / 1E9
                else:
                    gb = 0
                mem = f'{gb:.3g}G'  # cuda memory (GB)
                pbar.set_description(
                    ('%10s'*2 + '%10.4f'*3 + '%10d'*3)
                    % (f'{epoch}/{self.num_epochs - 1}', mem, *mloss,
                       trn_y1.shape[0], trn_y2.shape[0], trn_X2.shape[-1])
                )
            # if active_path_for_arch:
            #     LOGGER.info(f'active path for arch. params')
            # for v in active_path_for_arch:
            #     print(f'{v}')
            # LOGGER.info(f'active path for model params')
            # for v in active_path_for_model:
            #     print(v)

        return mloss

    def _loss_and_items_for_weight_update(self, imgs, targets):
        ''' return loss and loss_items for weight update '''
        with amp.autocast(enabled=self.cuda):
            pred = self.model(imgs)  # len(pred) = 3 : 3-pyramid output
            loss, loss_items = self.loss(pred, targets)  # scaled by batch_size
            if RANK != -1:
                # gradient averaged between devices in DDP mode
                loss *= WORLD_SIZE
        return loss, loss_items

    def _validate(self):
        # scheduler
        lr = [x['lr'] for x in self.optimizer.param_groups]  # for loggers
        self.scheduler.step()

        if RANK in (-1, 0):
            # validation ------------------------------------------------------
            # self.ema.update_attr(self.model,
            #                      include=['yaml',
            #                               'nc',
            #                               'hyp',
            #                               'names',
            #                               'stride',
            #                               'class_weights'])

            # calculate mAP
            results, maps, _ = val.run(
                batch_size=self.batch_size // WORLD_SIZE * 2,
                imgsz=self.img_size,
                save_dir=Path(self.args.save_dir),
                half=False,
                model=self.model,  # self.ema.ema,
                dataloader=self.val_data,
                compute_loss=self.loss
            )
            # end validation --------------------------------------------------

        return results, lr

    def _log_n_save(self, epoch, mloss, results, lr):
        ckpt = {
            'epoch': epoch,
            'best_fitness': self.best_fitness,
            'model': deepcopy(de_parallel(self.model)),
            # 'ema': deepcopy(self.ema.ema).half(),
            # 'updates': self.ema.updates,
            'optimizer': self.optimizer.state_dict(),
            'ctrl_optim': self.ctrl_optim.state_dict(),
            'date': datetime.now().isoformat()}

        final_epoch = (epoch + 1 == self.num_epochs)

        # update best mAP
        # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        fi = fitness(np.array(results).reshape(1, -1))
        if fi > self.best_fitness:
            LOGGER.info(
                colorstr("bright_green", f'save the best model:')
                + f' path = {self.best_dir} '
                f'pre = {self.best_fitness} cur = {fi} '
                f'lr = {lr}')
            if not self.search_type == 'retrain':
                for name, module in self.model.neck_module.named_modules():
                    if hasattr(module, 'get_arch_weight'):
                        LOGGER.info(f'\tneck {name} : ')
                        for idx, item in enumerate(module.get_arch_weight()):
                            print(f'[path {idx+1}]: {item:0.3f}', end=' ')
                        print('')
            self.best_fitness = fi
            torch.save(ckpt, self.best_dir)
        # log_vals = list(mloss) + list(results) + lr
        # LOGGER.debug(
        #     f'on fit epoch end:'
        #     f' {log_vals}, best fitness={self.best_fitness},'
        #     f' current fitness={fi}'
        # )

        # save model
        # torch.save(ckpt, self.last_dir)
        # if self.best_fitness == fi:
        #     torch.save(ckpt, self.best_dir)
        # if ((epoch > 0) and (opt.save_period > 0)
        #     and (epoch % opt.save_period == 0)):
        # torch.save(ckpt, self.w / f'epoch{epoch}.pt')
        del ckpt
        # LOGGER.debug(
        #     f'on model save: '
        #     f'{self.last_dir} {epoch} {final_epoch} {self.best_fitness} {fi}'
        # )

    def fit(self):
        for i in range(self.num_epochs):
            mloss = self._train_one_epoch(i)
            results, lr = self._validate()
            self._log_n_save(i, mloss, results, lr)
            # if not self.search_type == 'retrain':
            #     for name, module in self.model.neck_module.named_modules():
            #         if hasattr(module, 'get_arch_weight'):
            #             LOGGER.info(f'neck {name} : ')
            #             for idx, item in enumerate(module.get_arch_weight()):
            #                 print(f'[path {idx+1}]: {item:0.3f}', end=' ')
            #             print('')

    @torch.no_grad()
    def export(self):
        return self.model.neck_module.return_list
