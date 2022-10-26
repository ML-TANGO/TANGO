# References:
# [1] YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Model validation metrics
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch


def fitness(_x):
    '''
    Model fitness as a weighted combination of metrics
    _w: weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    '''
    _w = [0.0, 0.0, 0.1, 0.9]
    return (_x[:, :4] * _w).sum(1)


def smooth(_y, _f=0.05):
    '''
    Box filter of fraction f
    number of filter elements (must be odd)
    '''
    _nf = round(len(_y) * _f * 2) // 2 + 1
    _p = np.ones(_nf // 2)  # ones padding
    _yp = np.concatenate((_p * _y[0], _y, _p * _y[-1]), 0)  # y padded
    return np.convolve(_yp, np.ones(_nf) / _nf, mode='valid')  # y-smoothed


def ap_per_class(_tp, conf, pred_cls, target_cls,
                 plot=False, save_dir='.',
                 names=(), eps=1e-16):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    _i = np.argsort(-conf)
    _tp, conf, pred_cls = _tp[_i], conf[_i], pred_cls[_i]

    # Find unique classes
    unique_classes, _nt = np.unique(target_cls, return_counts=True)
    _nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    _px, _py = np.linspace(0, 1, 1000), []  # for plotting
    _ap, _p, _r = np.zeros((_nc, _tp.shape[1])), np.zeros(
        (_nc, 1000)), np.zeros((_nc, 1000))
    for _ci, _c in enumerate(unique_classes):
        _i = pred_cls == _c
        _n_l = _nt[_ci]  # number of labels
        _n_p = _i.sum()  # number of predictions
        if _n_p == 0 or _n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - _tp[_i]).cumsum(0)
        tpc = _tp[_i].cumsum(0)

        # Recall
        recall = tpc / (_n_l + eps)  # recall curve
        # negative x, xp because xp decreases
        _r[_ci] = np.interp(-_px, -conf[_i], recall[:, 0], left=0)

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        _p[_ci] = np.interp(-_px, -conf[_i], precision[:, 0],
                            left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(_tp.shape[1]):
            _ap[_ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                _py.append(np.interp(_px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    _f1 = 2 * _p * _r / (_p + _r + eps)
    # list: only classes that have data
    names = [v for k, v in names.items() if k in unique_classes]
    names = dict(enumerate(names))  # to dict
    if plot:
        plot_pr_curve(_px, _py, _ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(_px, _f1, Path(save_dir) /
                      'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(_px, _p, Path(save_dir) / 'P_curve.png',
                      names, ylabel='Precision')
        plot_mc_curve(_px, _r, Path(save_dir) /
                      'R_curve.png', names, ylabel='Recall')

    _i = smooth(_f1.mean(0), 0.1).argmax()  # max F1 index
    _p, _r, _f1 = _p[:, _i], _r[:, _i], _f1[:, _i]
    _tp = (_r * _nt).round()  # true positives
    _fp = (_tp / (_p + eps) - _tp).round()  # false positives
    return _tp, _fp, _p, _r, _f1, _ap, unique_classes.astype(int)


def compute_ap(recall, precision):
    """ Compute the average precision,
    given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        _x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        _ap = np.trapz(np.interp(_x, mrec, mpre), _x)  # integrate
    else:  # 'continuous'
        # points where x axis (recall) changes
        _i = np.where(mrec[1:] != mrec[:-1])[0]
        _ap = np.sum((mrec[_i + 1] - mrec[_i]) *
                     mpre[_i + 1])  # area under curve

    return _ap, mpre, mrec


class ConfusionMatrix:
    '''
    Updated version of
    https://github.com/kaanakan/object_detection_confusion_matrix
    '''

    def __init__(self, _nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((_nc + 1, _nc + 1))
        self._nc = _nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        _x = torch.where(iou > self.iou_thres)
        if _x[0].shape[0]:
            matches = torch.cat(
                (torch.stack(_x, 1),
                 iou[_x[0], _x[1]][:, None]), 1).cpu().numpy()
            if _x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(
                    matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(
                    matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        _n = matches.shape[0] > 0
        _m0, _m1, _ = matches.transpose().astype(int)
        for i, _gc in enumerate(gt_classes):
            _j = _m0 == i
            if _n and sum(_j) == 1:
                self.matrix[detection_classes[_m1[_j]], _gc] += 1  # correct
            else:
                self.matrix[self._nc, _gc] += 1  # background FP

        if _n:
            for i, _dc in enumerate(detection_classes):
                if not any(_m1 == i):
                    self.matrix[_dc, self._nc] += 1  # background FN

    def tp_fp(self):
        '''tp_fp'''
        _tp = self.matrix.diagonal()  # true positives
        _fp = self.matrix.sum(1) - _tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return _tp[:-1], _fp[:-1]  # remove background class


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, 1), box2.chunk(4, 1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, 1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, 1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


def box_area(box):
    '''box_area'''
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/\
    # master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (_a1, _a2), (_b1, _b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (torch.min(_a2, _b2) - torch.max(_a1, _b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter)


def bbox_ioa(box1, box2, eps=1E-7):
    """
    Returns the intersection over box2 area
    given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2,
                             b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
        (np.minimum(b1_y2,
                    b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area


def wh_iou(wh1, wh2):
    '''
    Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    '''
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    # iou = inter / (area1 + area2 - inter)
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)


def plot_pr_curve(_px, _py, _ap,
                  save_dir=Path('pr_curve.png'),
                  names=()):
    '''Precision-recall curve'''
    fig, _ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    _py = np.stack(_py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, _y in enumerate(_py.T):
            # plot(recall, precision)
            _ax.plot(_px, _y, linewidth=1, label=f'{names[i]} {_ap[i, 0]:.3f}')
    else:
        _ax.plot(_px, _py, linewidth=1, color='grey')
        # plot(recall, precision)

    _ax.plot(_px, _py.mean(1), linewidth=3, color='blue',
             label='all classes %.3f mAP@0.5' % _ap[:, 0].mean())
    _ax.set_xlabel('Recall')
    _ax.set_ylabel('Precision')
    _ax.set_xlim(0, 1)
    _ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(save_dir, dpi=250)
    plt.close()


def plot_mc_curve(_px, _py, save_dir=Path('mc_curve.png'),
                  names=(), xlabel='Confidence', ylabel='Metric'):
    '''Metric-confidence curve'''
    fig, _ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, _y in enumerate(_py):
            # plot(confidence, metric)
            _ax.plot(_px, _y, linewidth=1, label=f'{names[i]}')
    else:
        # plot(confidence, metric)
        _ax.plot(_px, _py.T, linewidth=1, color='grey')

    _y = smooth(_py.mean(0), 0.05)
    _ax.plot(_px, _y, linewidth=3, color='blue',
             label=f'all classes {_y.max():.2f} at {_px[_y.argmax()]:.3f}')
    _ax.set_xlabel(xlabel)
    _ax.set_ylabel(ylabel)
    _ax.set_xlim(0, 1)
    _ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(save_dir, dpi=250)
    plt.close()
