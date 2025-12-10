# Simplified Version of autobatch.py of Yolov5, AGPL-3.0 license
import random
import gc
import logging
from copy import deepcopy

import torch

from tango.main import status_update
from tango.common.models.yolo import DualDDetect, TripleDDetect 
from tango.utils.general import colorstr


DEBUG = True
PREFIX = colorstr('AutoBatch: ')

logger = logging.getLogger(__name__)

class TestFuncGen:
    def __init__(self, model, ch, imgsz, v9, num_classes=80, max_boxes=30, amp=False):
        """
        model: torch.nn.Module (detection model)
        ch: input channels
        imgsz: square input size
        v9: YOLOv9 계열 여부(dual/triple head 대비)
        num_classes: 학습 시 클래스 수(대체 로스 경로에서 필요)
        max_boxes: 이미지당 생성할 최대 박스 수(랜덤 0~max)
        amp: autocast 테스트 여부
        """
        self.model = model
        self.ch = ch
        self.imgsz = imgsz
        self.v9 = v9
        self.num_classes = num_classes
        self.max_boxes = max_boxes
        self.amp = amp

        # fake loss for testing
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="mean")
        self.l1  = torch.nn.SmoothL1Loss(beta=1.0, reduction="mean")

    def _make_fake_targets(self, batch_size, device):
        """
        YOLO 계열에서 흔히 쓰는 형식: (N, 6) = [img_idx, cls, x, y, w, h], 모두 0~1 정규화
        배치별로 가변 수의 박스를 만든 뒤 이어붙임
        """
        all_tgts = []
        for b in range(batch_size):
            n = random.randint(0, self.max_boxes)  # 0~max_boxes
            if n == 0:
                continue
            cls = torch.randint(0, self.num_classes, (n, 1), device=device).float()
            xywh = torch.rand(n, 4, device=device)  # 0~1 정규화
            img_idx = torch.full((n, 1), float(b), device=device)
            tgts = torch.cat([img_idx, cls, xywh], dim=1)  # (n,6)
            all_tgts.append(tgts)
        if len(all_tgts) == 0:
            # 박스가 하나도 없는 경우도 실제로 생기므로, 빈 텐서 반환
            return torch.zeros(0, 6, device=device)
        return torch.cat(all_tgts, dim=0)

    def _try_real_loss(self, preds, targets):
        """
        모델에 실제 loss 함수가 있으면 그걸 최대한 사용
            - 우선순위: model.compute_loss → model.criterion → preds가 dict이고 'loss' 키 보유
            - 반환: (loss_tensor or None)
        """
        # (1) YOLOv5/YOLOv8 계열: compute_loss(preds, targets)
        if hasattr(self.model, "compute_loss") and callable(getattr(self.model, "compute_loss")):
            try:
                out = self.model.compute_loss(preds, targets)
                # 일부 구현은 (loss, items) 튜플로 반환
                return out[0] if isinstance(out, (tuple, list)) else out
            except Exception:
                pass

        # (2) criterion(preds, targets)
        if hasattr(self.model, "criterion") and callable(getattr(self.model, "criterion")):
            try:
                out = self.model.criterion(preds, targets)
                return out[0] if isinstance(out, (tuple, list)) else out
            except Exception:
                pass

        # (3) preds 가 dict로 loss를 포함하는 경우
        if isinstance(preds, dict) and "loss" in preds:
            loss_val = preds["loss"]
            if torch.is_tensor(loss_val):
                return loss_val

        return None

    def _synthetic_det_loss(self, preds, batch_size):
        """
        실제 손실이 없을 때, detection head의 텐서 구조를 이용해
        obj(BCE), cls(CE), box(L1) 성분을 '대충' 흉내 내서 메모리 경로를 활성화
            - preds: list/tuple of heads or single tensor
            - 각 head 텐서를 [B, A, H, W, C] 혹은 [B, HW*A, C] 형태로 가정하고,
              C >= 5(+num_classes)라면 [tx, ty, tw, th, obj, cls...]로 분할
        """
        if not isinstance(preds, (list, tuple)):
            preds = [preds]

        total_loss = 0.0
        for p in preds:
            # 다양한 구현을 커버하기 위해 2가지 대표 케이스만 핸들
            if p.dim() == 5:
                # [B, A, H, W, C]
                B, A, H, W, C = p.shape
                P = p.view(B, A*H*W, C)  # [B, N, C]
            elif p.dim() == 3:
                # [B, N, C]
                B, N, C = p.shape
                P = p
            else:
                # 알 수 없는 형태면 그냥 평균 제곱으로라도 경로 열기
                total_loss = total_loss + (p ** 2).mean()
                continue

            # C 판단: 최소 5(ch: bbox4 + obj1) + cls
            if C >= 6:
                num_cls = max(1, C - 5)
            else:
                num_cls = 1  # 아주 작은 C면 억지로 obj만 있다고 가정

            # 분할 (가능하면)
            if C >= 5:
                box = P[..., 0:4]                # bbox 회귀
                obj = P[..., 4:5]                # objectness
                cls = P[..., 5:5+num_cls] if C >= 6 else None
            else:
                # 분할 불가: 그냥 전체를 제곱합
                total_loss = total_loss + (P ** 2).mean()
                continue

            # 랜덤한 '양성' 위치를 이미지당 소량 생성 → obj/cls가 제대로 활성화되도록
            # 이미지마다 서로 다른 양의 샘플 개수로 메모리 경로를 다양화
            pos_per_img = max(1, P.shape[1] // 2000)  # 넉넉하지 않게 소수만
            pos_idx = []
            for b in range(batch_size):
                choice = torch.randperm(P.shape[1], device=P.device)[:pos_per_img]
                pos_idx.append(choice)
            # obj 타깃: 기본 0, 양성 위치만 1
            obj_t = torch.zeros_like(obj)
            for b in range(batch_size):
                obj_t[b, pos_idx[b], 0] = 1.0

            # cls 타깃: one-hot on positives만(있으면)
            if cls is not None and cls.numel() > 0:
                cls_t = torch.zeros_like(cls)
                for b in range(batch_size):
                    # 무작위 클래스 할당
                    rand_cls = torch.randint(0, cls.shape[-1], (pos_per_img,), device=cls.device)
                    cls_t[b, pos_idx[b], :] = 0.0
                    cls_t[b, pos_idx[b], :].scatter_(1, rand_cls.view(-1,1), 1.0)
                # BCE with logits over one-hot
                cls_loss = self.bce(cls, cls_t)
            else:
                cls_loss = 0.0

            # box 타깃: 0~1 정규화 랜덤, 양성 위치만 비교
            box_t = torch.rand_like(box)
            # 양성 위치만 마스크해서 L1
            mask = torch.zeros_like(obj)
            for b in range(batch_size):
                mask[b, pos_idx[b], 0] = 1.0
            mask4 = mask.expand_as(box)
            if mask4.sum() > 0:
                box_loss = self.l1(box[mask4.bool()].view(-1,4), box_t[mask4.bool()].view(-1,4))
            else:
                box_loss = 0.0

            obj_loss = self.bce(obj, obj_t)
            total_loss = total_loss + (obj_loss + cls_loss + box_loss)

        return total_loss

    def __call__(self, batch_size):
        device = next(self.model.parameters()).device
        img = torch.zeros(batch_size, self.ch, self.imgsz, self.imgsz).float()
        img = img.to(device)
        targets = self._make_fake_targets(batch_size, device)

        try:
            y = self.model(img)
            y = y[1] if isinstance(y, tuple) else y
            if self.v9:
                y = y[-1] if isinstance(y, list) else y # one more time (just in case dual or triple heads: yolov9)
            # loss = y.mean()
            # loss = (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum() # from yolov9 code
            # 1) 실제 loss가 있으면 그걸 우선 사용
            loss = self._try_real_loss(y, targets)
            # 2) 실제 loss 없으면, detection 구조를 흉내 낸 합성 loss 사용
            if loss is None:
                _pred_list = None
                if isinstance(y, dict):
                    for k in ["preds", "head_outs", "outputs"]:
                        if k in y:
                            _pred_list = y[k]
                            break
                    if _pred_list is None:
                        _pred_list = list(y.values())
                else:
                    _pred_list = y
                loss = self._synthetic_det_loss(_pred_list, batch_size)

            loss.backward() # need to free the variables of the graph
            del img, targets, y, loss
            return True
        except RuntimeError as e:
            del img
            return False

def binary_search(uid, pid, low, high, test_func, want_to_get):
    logger.info(f'{PREFIX}Start Binary Search')
    low_result = test_func(low)
    high_result = test_func(high)

    batchsize_content = {}
    while True:
        next_test = int((low + high) / 2.)
        if next_test==low or next_test==high:
            logger.info(f'{PREFIX}Binary Search Complete: Max Batch Size = {next_test}')
            return low if low_result==want_to_get else high

        judge = test_func(next_test)
        if judge==low_result:
            low = next_test
            low_result = judge
            if DEBUG:
                low_str  = str(colorstr("underline", str(low)))
                high_str = str(colorstr("end", str(high)))
                logger.info(f'{PREFIX}{low_str:>4s} ⭕ | {high_str:>4s} ❌')
        elif judge == high_result:
            high = next_test
            high_result = judge
            if DEBUG:
                high_str = str(colorstr("underline", str(high)))
                low_str  = str(colorstr("end", str(low)))
                logger.info(f'{PREFIX}{low_str:>4s} ⭕ | {high_str:>4s} ❌')

        batchsize_content['low'] = low
        batchsize_content['high'] = high
        status_update(uid, pid,
                      update_id="batchsize",
                      update_content=batchsize_content)

def get_batch_size_for_gpu(uid, pid, model, ch, imgsz, bs_factor=0.8, amp_enabled=True, max_search=True):
    with torch.cuda.amp.autocast(enabled=amp_enabled):
        return autobatch(uid, pid, model, ch, imgsz, bs_factor, max_search=max_search)

def autobatch(uid, pid, model, ch, imgsz, bs_factor=0.8, batch_size=16, max_search=True):
    # Check device
    # device = torch.device(f'cuda:0')
    device = next(model.parameters()).device
    if device != 'cpu' and torch.cuda.is_available():
       num_dev = torch.cuda.device_count() 
    else:
       logger.info(f'{PREFIX}CUDA not detected, using default CPU batch size {batch_size}')
       return batch_size
    if torch.backends.cudnn.benchmark:
        logger.info(f'{PREFIX}requires cudnn.benchmark=False, using default batch size {batch_size}')
        return batch_size

    # Inspect CUDA memory
    gb = 1 << 30 # bytes in GiB (1024**3)
    d = str(device).upper() # 'CUDA:0'
    properties = torch.cuda.get_device_properties(device)
    t = properties.total_memory / gb            # total
    r = torch.cuda.memory_reserved(device) / gb # reserved
    a = torch.cuda.memory_allocated(device)/ gb # allocated
    f = t - (r + a)                             # free = total - (reserved + allocated)
    logger.info(f'\n{PREFIX}{d} ({properties.name}) {t:.2f}G total, {r:.2f}G reserved, {a:.2f}G allocated, {f:.2f}G free')

    # model.to(device)
    v9 = False
    for _, m in model.named_modules():
        if isinstance(m, (DualDDetect, TripleDDetect)):
            v9 = True
    model.train()
    batchsize_content = {}
    batch_size = 2 # 2, 4, 8, 16, 32, 64, 128, 256, ...
    while True:
        img = torch.zeros(batch_size, ch, imgsz, imgsz).float()
        img = img.to(device)
        try:
            y = model(img)
            y = y[1] if isinstance(y, tuple) else y
            if v9:
                y = y[-1] if isinstance(y, list) else y # one more time (just in case dual or triple heads: yolov9)
            ''' in v9 case,
                training :list   =   [d1, d2]
                inference:tuple  = ( [c1, c2], [d2, d2] )
                export   :tensor =   c1 ⨁ c2
                where, dx:list = [ (64,144,80,80), (64,144,40,40), (64,144,20,20) ]
                       cx:list = [ (64,144, 8400), (64,144, 8400) ]

                in v7 case,
                training :list  = [ (64,3,80,80,85), (64,3,40,40,85), (64,3,20,20,85) ]
                inference:tuple = ( (64, 25200, 85), [ (64,3,80,80,85), (64,3,40,40,85), (64,3,20,20,85) ] )
                export   :list  = [ (64,3,80,80,85), (64,3,40,40,85), (64,3,20,20,85) ]
            '''
            # loss = y.mean()
            loss = (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum() # from yolov9 code
            loss.backward() # need to free the variables of the graph
            if DEBUG: logger.info(f'{PREFIX}{batch_size:>4.0f} ⭕ success')
            batchsize_content['low'] = batch_size
            status_update(uid, pid,
                          update_id="batchsize",
                          update_content=batchsize_content)
            batch_size = batch_size * 2
            del img
        except RuntimeError as e:
            if DEBUG: logger.info(f'{PREFIX}{batch_size:>4.0f} ❌ fail')
            final_batch_size = int(batch_size / 2.)
            batchsize_content['low'] = final_batch_size
            batchsize_content['high'] = batch_size
            status_update(uid, pid,
                          update_id="batchsize",
                          update_content=batchsize_content)
            del img
            break
    torch.cuda.empty_cache()

    base_max_batch_size = final_batch_size
    if max_search: # search maximum batch size (allow size other than multiple of 2)
        test_func = TestFuncGen(model, ch, imgsz, v9)
        final_batch_size = binary_search(uid, pid, final_batch_size, batch_size, test_func, want_to_get=True)
        logger.info(f'{PREFIX}{final_batch_size} x margin({bs_factor}) = {int(final_batch_size * bs_factor)}')
        base_max_batch_size = final_batch_size
        final_batch_size *= bs_factor # need some spare
    final_batch_size = max(final_batch_size, 1.0)
    status_update(
        uid,
        pid,
        update_id="batchsize",
        update_content={
            "low": int(final_batch_size),          # 최종 사용(per-device) 배치 사이즈
            "high": int(base_max_batch_size),      # 검색된 최대(per-device) 배치 사이즈
        },
    )
    torch.cuda.empty_cache()

    gc.collect()
    return final_batch_size

