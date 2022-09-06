# References:
# [1] YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from .metrics import bbox_iou
from .torch_utils import de_parallel


def smooth_bce(eps=0.1):
    '''
    https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    return positive, negative label smoothing BCE targets
    '''
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    '''
    Wraps focal loss around existing loss_fcn(),
    i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    '''
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        # required to apply FL to each element
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        '''forward'''
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma
        # non-zero power for gradient stability

        # TF implementation
        # https://github.com/tensorflow/addons/blob\
        # /v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class ComputeLoss:
    '''
    Compute Loss
    '''
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        # h = model.hyp  # hyperparameters

        # Define criteria
        bce_cls = nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor([1.0], device=device))
        bce_obj = nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor([1.0], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        # self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))
        # positive, negative BCE targets
        self._cp, self._cn = smooth_bce(eps=0)

        # Focal loss
        _g = 0  # focal loss gamma (efficientDet default gamma=1.5)
        if _g > 0:
            bce_cls, bce_obj = FocalLoss(bce_cls, _g), FocalLoss(bce_obj, _g)

        _m = de_parallel(model).head[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(
            _m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(_m.stride).index(
            16) if autobalance else 0  # stride 16 index
        self.bce_cls, self.bce_obj, self._gr, self.autobalance = \
            bce_cls, bce_obj, 1.0, autobalance
        self._na = _m.na  # number of anchors
        self._nc = _m.nc  # number of classes
        self._nl = _m.nl  # number of layers
        self.anchors = _m.anchors
        self.device = device

    def __call__(self, _p, targets):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(
            _p, targets)  # targets

        # Losses
        for i, _pi in enumerate(_p):  # layer index, layer predictions
            _b, _a, _gj, _gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(
                _pi.shape[:4],
                dtype=_pi.dtype,
                device=self.device)  # target obj

            _n = _b.shape[0]  # number of targets
            if _n:
                # pxy, pwh, _, pcls = \
                # pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)
                # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = _pi[_b, _a, _gj, _gi].split(
                    (2, 2, 1, self._nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                # iou(prediction, target)
                iou = bbox_iou(pbox, tbox[i], c_iou=True).squeeze()
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    _j = iou.argsort()
                    _b, _a, _gj, _gi, iou = (_b[_j],
                                             _a[_j], _gj[_j],
                                             _gi[_j], iou[_j])
                if self._gr < 1:
                    iou = (1.0 - self._gr) + self._gr * iou
                tobj[_b, _a, _gj, _gi] = iou  # iou ratio

                # Classification
                if self._nc > 1:  # cls loss (only if multiple classes)
                    _t = torch.full_like(
                        pcls, self._cn, device=self.device)  # targets
                    _t[range(_n), tcls[i]] = self._cp
                    lcls += self.bce_cls(pcls, _t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') \
                # for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.bce_obj(_pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * \
                    0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= 0.05
        lobj *= 1.0
        lcls *= 0.5
        _bs = tobj.shape[0]  # batch size

        return ((lbox + lobj + lcls) * _bs,
                torch.cat((lbox, lobj, lcls)).detach())

    def build_targets(self, _p, targets):
        '''
        Build targets for compute_loss(),
        input targets(image,class,x,y,w,h)
        '''
        _na, _nt = self._na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        # normalized to gridspace gain
        gain = torch.ones(7, device=self.device)
        _ai = torch.arange(_na, device=self.device).float().view(
            _na, 1).repeat(1, _nt)  # same as .repeat_interleave(nt)
        # append anchor indices
        targets = torch.cat((targets.repeat(_na, 1, 1), _ai[..., None]), 2)

        _g = 0.5  # bias
        off = torch.Tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * _g  # offsets

        for i in range(self._nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.Tensor(_p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            _t = targets * gain  # shape(3,n,7)
            if _nt:
                # Matches
                _r = _t[..., 4:6] / anchors[:, None]  # wh ratio
                _j = torch.max(_r, 1 / _r).max(2)[0] < 4  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']
                # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                _t = _t[_j]  # filter

                # Offsets
                gxy = _t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                _j, _k = ((gxy % 1 < _g) & (gxy > 1)).T
                _l, _m = ((gxi % 1 < _g) & (gxi > 1)).T
                _j = torch.stack((torch.ones_like(_j), _j, _k, _l, _m))
                _t = _t.repeat((5, 1, 1))[_j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[_j]
            else:
                _t = targets[0]
                offsets = 0

            # Define
            # (image, class), grid xy, grid wh, anchors
            _bc, gxy, gwh, _a = _t.chunk(4, 1)
            # anchors, image, class
            _a, (_b, _c) = _a.long().view(-1), _bc.long().T
            gij = (gxy - offsets).long()
            _gi, _gj = gij.T  # grid indices

            # Append
            # image, anchor, grid indices
            indices.append(
                (_b, _a, _gj.clamp_(0, gain[3] - 1),
                 _gi.clamp_(0, gain[2] - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[_a])  # anchors
            tcls.append(_c)  # class

        return tcls, tbox, indices, anch
