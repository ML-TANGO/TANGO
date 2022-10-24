'''
eval model
'''

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..utils.accelerate import check_amp
from ..utils.general import (non_max_suppression, scale_coords,
                             xywh2xyxy)
from ..utils.loss import ComputeLoss
from ..utils.metrics import ap_per_class, box_iou

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# https://pytorch.org/docs/stable/elastic/run.html
LOCAL_RANK = int(os.environ.get('LOCAL_RANK', -1))
RANK = int(os.environ.get('RANK', -1))


def cal_metrics(val_loader, model, _nc, names):
    '''
    get metrics
    '''

    amp = check_amp(model)  # check AMP
    model = fine_tune(val_loader, model, amp)

    # supernet.eval()
    # base_model.eval()
    model.eval()
    # get model device, PyTorch model
    device = next(model.parameters()).device
    # half = False
    # half &= device.type != 'cpu'  # half precision only supported on CUDA
    # model.half() if half else
    model.float()
    # base_model.half() if half else base_model.float()
    # supernet.half() if half else supernet.float()

    cuda = device.type != 'cpu'
    # iou vector for mAP@0.5:0.95
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    niou = iouv.numel()

    seen = 0
    # confusion_matrix = ConfusionMatrix(nc=nc)
    _s = ('%20s' + '%11s' * 6) % ('Class', 'Images',
                                  'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    _p, _r, _mp, _mr, map50, m_ap = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    stats, _ap = [], []

    pbar = tqdm(val_loader, desc=_s,
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    for _, (img, targets, paths, shapes) in enumerate(pbar):
        if cuda:
            img = img.to(device, non_blocking=True)
            targets = targets.to(device)
        # img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img.float()
        img /= 255  # 0 - 255 to 0.0 - 1.0
        _, _, height, width = img.shape  # batch size, channels, height, width

        # Inference
        with torch.no_grad():
            out = model(img)[0]  # inference
        # to pixels
        targets[:, 2:] *= torch.tensor((width,
                                        height, width, height), device=device)
        out = non_max_suppression(
            out, 0.001, 0.6, labels=[], multi_label=True, agnostic=False)

        # Metrics
        for _si, pred in enumerate(out):
            labels = targets[targets[:, 0] == _si, 1:]
            # number of labels, predictions
            _nl, npr = labels.shape[0], pred.shape[0]
            _, shape = Path(paths[_si]), shapes[_si][0]
            correct = torch.zeros(
                npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if _nl:
                    stats.append(
                        (correct, *torch.zeros((3, 0), device=device)))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[_si].shape[1:], predn[:, :4], shape,
                         shapes[_si][1])  # native-space pred

            # Evaluate
            if _nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(img[_si].shape[1:], tbox, shape,
                             shapes[_si][1])  # native-space labels
                # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                correct = process_batch(predn, labelsn, iouv)
                # if plots:
                #    confusion_matrix.process_batch(predn, labelsn)
            # (correct, conf, pcls, tcls)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if stats and stats[0].any():
        # tp, fp, p, r, f1, ap, ap_class
        _, _, _p, _r, _, _ap, _ = ap_per_class(
            *stats, plot=False, save_dir=ROOT / 'save', names=names)
        ap50, _ap = _ap[:, 0], _ap.mean(1)  # AP@0.5, AP@0.5:0.95
        _mp, _mr, map50, m_ap = _p.mean(), _r.mean(), ap50.mean(), _ap.mean()
        # number of targets per class
        _nt = np.bincount(stats[3].astype(int), minlength=_nc)
    else:
        _nt = torch.zeros(1)

    # Print results
    _pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    print(_pf % ('all', seen, _nt.sum(), _mp, _mr, map50, m_ap))
    return m_ap


# change train_loader
def fine_tune(loader, model, amp):
    '''
    fine-tuning
    '''
    # temporal
    device = 0

    # optimizer
    _g = [], [], []  # optimizer parameter groups
    # normalization layers, i.e. BatchNorm2d()
    _bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
    for _v in model.modules():
        if hasattr(_v, 'bias') and isinstance(_v.bias, nn.Parameter):  # bias
            _g[2].append(_v.bias)
        if isinstance(_v, _bn):  # weight (no decay)
            _g[1].append(_v.weight)
        elif ((hasattr(_v, 'weight')
               and isinstance(_v.weight, nn.Parameter))):
            _g[0].append(_v.weight)  # weight (with decay)
    optimizer = torch.optim.Adam(_g[2], lr=0.01, betas=(0.937, 0.999))

    # add g0 with weight_decay
    optimizer.add_param_group({'params': _g[0], 'weight_decay': 0.0005})
    # add g1 (BatchNorm2d weights)
    optimizer.add_param_group({'params': _g[1]})
    del _g

    compute_loss = ComputeLoss(model)

    model.train()
    model.float()

    # mloss = torch.zeros(3, device)
    pbar = enumerate(loader)
    _nb = len(loader)
    # progress bar
    pbar = tqdm(pbar, total=_nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    optimizer.zero_grad()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # batch -------------------------------------------------------------
    for _, (imgs, targets, _, _) in pbar:
        # number integrated batches (since train start)
        # ni = i + nb
        imgs = imgs.to(device, non_blocking=True).float() / \
            255  # uint8 to float32, 0-255 to 0.0-1.0

        # Forward
        with torch.cuda.amp.autocast(amp):
            pred = model(imgs)  # forward
            loss = compute_loss(pred, targets.to(device))[
                0]  # loss scaled by batch_size
            # if RANK != -1:
            #    loss *= WORLD_SIZE
            # # gradient averaged between devices in DDP mode
            # if opt.quad:
            #    loss *= 4.

        # Backward
        scaler.scale(loss).backward()
    return model


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix.
    Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(
        detections.shape[0], iouv.shape[0],
        dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i, _ in enumerate(iouv):
        # IoU > threshold and classes match
        _x = torch.where((iou >= iouv[i]) & correct_class)
        if _x[0].shape[0]:
            matches = torch.cat(
                (torch.stack(_x, 1), iou[_x[0], _x[1]][:, None]),
                1).cpu().numpy()  # [label, detect, iou]
            if _x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(
                    matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(
                    matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct
