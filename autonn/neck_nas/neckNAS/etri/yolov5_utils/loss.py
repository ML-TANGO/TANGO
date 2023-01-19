# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from .metrics import bbox_iou
from .torch_utils import de_parallel


def smooth_BCE(eps=0.1):
    # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        # must be nn.BCEWithLogitsLoss()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # reduce missing label and false label effects
        # dx = (pred - true).abs()
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(),
    # i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        # required to apply FL to each element
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # non-zero power for gradient stability
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma

        # TF implementation
        # https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/
        # losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(),
    # i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        # required to apply FL to each element
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']],
                                      device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']],
                                      device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        # positive, negative BCE targets
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        # m = de_parallel(model).model[-1]  # Detect() module
        m = model.model[-1]

        # P3-P7
        self.balance = {
            3: [4.0, 1.0, 0.4]
        }.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])

        # stride 16 index
        self.ssi = list(m.stride).index(16) if autobalance else 0

        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = \
            BCEcls, BCEobj, 1.0, h, autobalance

        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets, prt_shape=False):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        # targets
        tcls, tbox, indices, anchors = self.build_targets(p, targets)
        if prt_shape:
            print(f'target shape={targets.shape}')

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            gj, gi = gj.long(), gi.long()
            # tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype,
            #                    device=self.device)  # target obj
            tobj = torch.zeros_like(pi[..., 0], device=self.device)

            n = b.shape[0]  # number of targets
            if n:
                # faster, requires torch 1.8.0
                # pxy, pwh, _, pcls = \
                #     pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)
                # target-subset of predictions
                pxy, pwh, _, pcls = \
                    pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                # iou(prediction, target)
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()
                liou = 1.0 - iou
                lbox += (1.0 - iou).mean()  # iou loss
                import numpy as np
                if np.isnan(lbox.item()):
                    print(f'ibox is nan')
                    print(f'\t{liou.shape}')
                    exit(0)

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    # targets
                    t = torch.full_like(pcls, self.cn, device=self.device)
                    t[range(n), tcls[i]] = self.cp

                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n')
                #         for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)

            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = \
                    self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, \
            torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        """ Build targets for compute_loss(),
            input targets(image,class,x,y,w,h) """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        # normalized to gridspace gain
        gain = torch.ones(7, device=self.device)
        # same as .repeat_interleave(nt)
        ai = torch.arange(na,
                          device=self.device).float().view(na, 1).repeat(1, nt)
        # append anchor indices
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)
        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                # compare
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']
                # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            # (image, class), grid xy, grid wh, anchors
            bc, gxy, gwh, a = t.chunk(4, 1)
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets)
            gi, gj = gij.T  # grid indices

            # Append
            # image, anchor, grid indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1),
                           gi.clamp_(0, gain[2] - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
