"""
===========================================================
🧪 TANGO Model Evaluation Script (test.py)
===========================================================
사용법 (CLI)
-----------------------------------------------------------
▶ 1. Validation / Testing
    모델의 성능을 평가하고 mAP, Precision, Recall 등을 측정

    예시:
    cd /source/autonn_core
    python -m tango.main.test \
        --purpose val \
        --weights ../archive/bestmodel.pt \
        --data /shared/datasets/coco/dataset.yaml \
        --img-size 1280 \
        --batch-size 4 \
        --device 0 \
        --iou-thres 0.7 \
        --save-json \
        --no-trace \
        --is-coco \
        --metric v9

▶ 2. Speed Benchmark
    모델의 처리 속도(inference latency, FPS 등)를 측정

    예시:
    python -m tango.main.test \
        --purpose speed \
        --weights ../archive/bestmodel.pt \
        --data /shared/datasets/coco/dataset.yaml \
        --img-size 1280 \
        --batch-size 4 \
        --device 0 \
        --metric v9

▶ 3. Study Mode (Resolution Sweep)
    여러 입력 해상도에서의 성능 변화를 측정하고 결과를 저장/시각화

    예시:
    python -m tango.main.test \
        --purpose study \
        --weights ../archive/bestmodel.pt \
        --data /shared/datasets/coco/dataset.yaml \
        --iou-thres 0.65 \
        --no-trace \
        --is-coco \
        --metric v9
    # 결과물:
    # study_<dataset>_<model>.txt 파일 생성 후 study.zip 압축
    # plot_study_txt(x=x) 로 결과 그래프 자동 생성

===========================================================
"""
import argparse
import logging
import json
import os
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch
import yaml
from tqdm import tqdm
import time
import datetime

from tango.common.models.experimental import attempt_load
from tango.utils.datasets import create_dataloader, create_dataloader_v9
from tango.utils.general import (
    coco80_to_coco91_class,
    check_dataset,
    check_file,
    check_img_size,
    box_iou,
    non_max_suppression,
    non_max_suppression_v9,
    scale_coords,
    xyxy2xywh,
    xyxy2xywh_v9,
    xywh2xyxy,
    xywh2xyxy_v9,
    set_logging,
    increment_path,
    colorstr,
    smart_inference_mode,
    TQDM_BAR_FORMAT,
    TqdmLogger,
)
from tango.utils.metrics import (
    ap_per_class,
    ap_per_class_v9,
    ConfusionMatrix,
    ConfusionMatrix_v9,
)
from tango.utils.plots import plot_images, output_to_target, plot_study_txt
from tango.utils.torch_utils import (
    select_device,
    time_synchronized,
    TracedModel,
)

logger = logging.getLogger(__name__)

def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh_v9(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')

def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh_v9(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)})

def report_progress(userid, 
                    project_id, 
                    statsn, 
                    seen,
                    label_cnt,
                    current_step,
                    total_step,
                    plots, 
                    metric, 
                    save_dir, 
                    names, 
                    latency,
                    start_time,
                    can_update_progress,
):
    _p, _r, _f1, _mp, _mr, _map50, _map = 0., 0., 0., 0., 0., 0., 0.
    _ap, _ap_class = [], []

    if len(statsn) and statsn[0].any():
        if metric == 'v9':
            _tp, _fp, _p, _r, _f1, _ap, _ap_class = ap_per_class_v9(
                *statsn, plot=False, save_dir=save_dir, names=names
            )
        else:
            _p, _r, _ap, _f1, _ap_class = ap_per_class(
                *statsn, plot=False, metric=metric, save_dir=save_dir, names=names
            )
        _ap50, _ap = _ap[:, 0], _ap.mean(1)
        _mp, _mr, _map50, _map = _p.mean(), _r.mean(), _ap50.mean(), _ap.mean()

    # Status update
    if can_update_progress:
        try:
            val_acc = {
                'class': 'all',
                'images': seen,
                'labels': label_cnt,
                'P': _mp,
                'R': _mr,
                'mAP50': _map50,
                'mAP50-95': _map,
                'step': current_step, # batch_i + 1
                'total_step': total_step, # len(dataloader)
                'time': f'{latency:.1f} s',
            }
            safe_status_update(userid, project_id,
                            update_id="val_accuracy",
                            update_content=val_acc)
        except Exception as e:
            pass

    elapsed_sec = time.time() - start_time
    elapsed_time = str(datetime.timedelta(seconds=elapsed_sec)).split('.')[0]
    ten_percent_cnt = int(current_step/total_step*10+0.5)
    bar = '|'+ '#'*ten_percent_cnt + ' '*(10-ten_percent_cnt)+'|'
    pf_c = '%22s' + '%11i' * 2 + '%11.3g' *4
    content = ('all', seen, label_cnt, _mp, _mr, _map50, _map)
    s = pf_c % content
    content_s = s + (f'{bar}{current_step/total_step*100:3.0f}% {current_step:4.0f}/{total_step:4.0f}  {elapsed_time}')
    logger.info(content_s)


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

def safe_status_update(userid, project_id, **kwargs):
    try:
        from . import status_update
        status_update(userid, project_id, **kwargs)
    except Exception:
        pass  # 외부 환경/CLI 등에서 통신 실패는 무시


@smart_inference_mode()
def test(proj_info,
         data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001, # for NMS
         iou_thres=0.7,  # for NMS
         single_cls=False,
         augment=False,
         verbose=False,
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save confidence in save_txt file
         save_json=False, # save json; with it, measuring metrics using pycocotools
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving results
         plots=True,
         compute_loss=None,
         half_precision=False,
         trace=False,
         is_coco=False,
         metric='v5'
):
    # Set device ---------------------------------------------------------------
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device
    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size
        
        if trace:
            model = TracedModel(model, device, imgsz)

    # Half ---------------------------------------------------------------------
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    model.half() if half else model.float()

    # Configure ----------------------------------------------------------------
    userid = (proj_info or {}).get('userid') if isinstance(proj_info, dict) else None
    project_id = (proj_info or {}).get('project_id') if isinstance(proj_info, dict) else None
    can_update_progress = training and userid and project_id

    # Model --------------------------------------------------------------------
    model.eval() # it will set {DualDDetect}.training to False

    # Dataset ------------------------------------------------------------------
    if isinstance(data, str):
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)  # check

    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    
    # =========================
    # [BEGIN PATCH v9-only NMS]
    # =========================
    def _v9_sanitize_and_nms(out, conf_thres, iou_thres, nc, device):
        try:
            from torchvision.ops import nms as tv_nms
        except Exception as e:
            raise RuntimeError(f"torchvision.ops.nms import failed: {e}")

        assert out.dim() == 3, f"v9 head must be 3D, got {tuple(out.shape)}"
        B = out.shape[0]

        if out.shape[-1] == 4 + nc:
            pred = out
        elif out.shape[1] == 4 + nc:
            pred = out.permute(0, 2, 1).contiguous()
        else:
            raise RuntimeError(f"Unexpected v9 output shape: {tuple(out.shape)} (nc={nc})")

        results = []
        for b in range(B):
            p = pred[b]
            boxes_xywh = p[:, :4]
            cls_scores = p[:, 4:]

            conf, cls = cls_scores.max(dim=1)

            keep = conf > conf_thres
            if keep.any():
                boxes_xywh = boxes_xywh[keep]
                conf = conf[keep]
                cls = cls[keep].float()

                boxes_xyxy = xywh2xyxy_v9(boxes_xywh)

                keep_idx = tv_nms(boxes_xyxy, conf, iou_thres)
                det = torch.cat(
                    [boxes_xyxy[keep_idx],
                     conf[keep_idx, None],
                     cls[keep_idx, None]], dim=1
                )
                results.append(det)
            else:
                results.append(torch.zeros((0, 6), device=device))
        return results

    def _maybe_fix_pred_format_v9(pred): ## 
        if pred is None or pred.shape[0] == 0:
            return pred
        wrong_x = (pred[:, 2] < pred[:, 0]).float().mean().item()
        wrong_y = (pred[:, 3] < pred[:, 1]).float().mean().item()
        if wrong_x > 0.5 or wrong_y > 0.5:
            xyxy = xywh2xyxy_v9(pred[:, :4].clone())
            pred = torch.cat([xyxy, pred[:, 4:]], dim=1)
        return pred

    def _v9_safe_nms(out_tensor, targets, nb, conf_thres, iou_thres, save_hybrid): ##
        if out_tensor.dim() == 3 and out_tensor.shape[1] == (4 + model.nc):
            out_for_nms = out_tensor.permute(0, 2, 1).contiguous()
        else:
            out_for_nms = out_tensor

        lb_local = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []
        dets = non_max_suppression_v9(
            out_for_nms,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            labels=lb_local,
            multi_label=True
        )
        dets = [_maybe_fix_pred_format_v9(d) for d in dets]
        return dets
    
    # Dataloader ---------------------------------------------------------------
    if not training: # called directly
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        if metric == 'v9':
            dataloader = create_dataloader_v9(
                data['val'],
                imgsz,
                batch_size,
                gs,
                single_cls,
                # hyp=None, # default:None
                # cache=False, # default:False
                # rank=-1, # default:-1
                # workers=16, # default:8
                pad=0.5,
                close_mosaic=True, # always use torch.utils.data.Dataloader
                rect=True,
                prefix='val',
                #shuffle=False, # default:False
                uid=userid, # None
                pid=project_id, # None
            )[0]
        else:
            dataloader = create_dataloader(
                userid,
                project_id,
                data['val'],
                imgsz,
                batch_size,
                gs,
                opt,
                pad=0.5,
                rect=True,
                prefix='val',
            )[0]

    # Metrics ------------------------------------------------------------------
    # logger.info(f"\n{colorstr('Test: ')}Testing with YOLO{metric} AP metric...")
        
    # V9 head numbers ----------------------------------------------------------
    m = model.model[-1]  # Detect() module
    v9, nh = False, 1
    if 'TripleDDetect' in m.type:
        nh = 3
        v9 = True
    elif 'DualDDetect' in m.type:
        nh = 2
        v9 = True
    elif 'DDetect' in m.type:
        nh = 1
        v9 = True
    
    # Test start ===============================================================

    if metric == 'v9':
        confusion_matrix = ConfusionMatrix_v9(nc=nc)
    else:
        confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    coco91class = coco80_to_coco91_class() if is_coco else list(range(1000))

    seen, label_cnt = 0, 0 # count
    tp, fp = 0., 0. # true positive, false positive
    p, r, f1, mp, mr, map50, map  = 0., 0., 0., 0., 0., 0., 0. # f1: hormonic value of P and R
    t, t0, t1, t2 = 0., 0., 0., 0. # latency
    loss = torch.zeros(3, device=device) # loss
    jdict, stats, ap, ap_class = [], [], [], [] # jdict=json dict, ap=average precision
    
    pf_t = '%22s' + '%11s' * 6
    title = ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@50', 'mAP@50-95')
    logger.info(pf_t % title)

    # pbar = enumerate(dataloader)
    # pbar = tqdm(
    #     dataloader, # pbar,
    #     desc=pf_t % title,
    #     total=len(dataloader),
    #     bar_format=TQDM_BAR_FORMAT,
    # )  # progress bar

    # logger.info(f'dataset size = {len(dataloader)}')
    # for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
    start_time = time.time()
    for batch_i, (img, targets, paths, shapes) in enumerate(dataloader): # enumerate(pbar):
        t = time_synchronized()
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        _t0 = time_synchronized() - t # _t0 = pre-process time
        t0 += _t0

        with torch.no_grad():
            # Run model --------------------------------------------------------
            t = time_synchronized()
            out, train_out = model(img, augment=augment)  # inference and training outputs
            '''
            for single anchor-based head (v7)
                out.shape = (bs, 25200, 5+80) : tensor // 5=x,y,w,h,conf, 80=cls
                train_out.shape = ( (bs,3,80,80,85), (bs,3,40,40,85), (bs,3,20,20,85) ) : list
            for dual anchor-free heads (v9)
                out.shape = ( (bs, 64+80, 8400), (bs, 64+80, 8400) ) : list // 64=bbox, 80=cls
                train_out.shape = ( ((bs,144,80,80), (bs,144,40,40), (bs,144,20,20)),
                                    ((bs,144,80,80), (bs,144,40,40), (bs,144,20,20))  ) : list
            '''
            # --- 중앙대 패치
            # if v9:
            #     if isinstance(out, (list, tuple)):
            #         if len(out) == 2:
            #             out = (out[0] + out[1]) / 2
            #         else:
            #             out = out[-1]
            
            # --- ETRI 패치
            if v9:
                if isinstance(out, (list, tuple)):
                    out = out[nh - 1] if nh > 1 else out[0]

            _t1 = time_synchronized() - t # _t1 = inference time
            t1 += _t1

            # Compute loss -----------------------------------------------------
            if compute_loss:
                if v9:
                    loss = compute_loss(train_out, targets)[1][:3] # box, dfl, cls
                else:
                    loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls (detached)

            # Run NMS ----------------------------------------------------------
            '''
            after nms,
                out = (bs, n, 6) : bs = batch size, n = detected object number, 6 = (x,y,w,h,conf,cls)
                (v7) (bs, 25200, 5+80) ==> NMS   ==> approx. (bs, n, 6)
                (v9) (bs, 64+80, 8400) ==> NMSv9 ==> approx. (bs, n, 6)
            '''
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t = time_synchronized()
            if v9:
                # --- 중앙대 패치 (torchvision nms 사용)
                # if isinstance(out, (list, tuple)):
                #     out = out[-1] if len(out) != 2 else (out[0] + out[1]) / 2
                # _nc = getattr(model, 'nc', None)
                # if _nc is None:
                #     _nc = len(names)

                # out = _v9_sanitize_and_nms(
                #     out, conf_thres=conf_thres, iou_thres=iou_thres,
                #     nc=_nc, device=device
                # )

                # --- ETRI 패치 (v9 저자의 nms 사용)
                out = non_max_suppression_v9(
                    out, conf_thres=conf_thres, iou_thres=iou_thres, 
                    labels=lb, multi_label=True
                )
            else:
                out = non_max_suppression(
                    out, conf_thres=conf_thres, iou_thres=iou_thres,
                    labels=lb, multi_label=True
                )
            _t2 = time_synchronized() - t
            t2 += _t2

            # --- 중앙대 패치 (디버깅용 추측)
            # if batch_i == 0:
            #     total_dets = sum(len(d) for d in out)
            #     top_conf = None
            #     for det in out:
            #         if det is not None and len(det):
            #             val = float(det[:, 4].max().item())
            #             top_conf = val if top_conf is None else max(top_conf, val)

        # Metrics per image ----------------------------------------------------
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0] # number of labels, predictions
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            # Evaluate : assign all predictions as incorrect
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)

            seen += 1
            label_cnt += nl
            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2,0), device=device), labels[:,0]))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tcls_tensor = labels[:, 0]
                # target boxes
                if v9:
                    tbox = xywh2xyxy_v9(labels[:, 1:5])
                else:
                    tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1) # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
                
                # Per target class
                if v9:
                    correct = process_batch(predn, labelsn, iouv)
                else:
                    for cls in torch.unique(tcls_tensor):
                        ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # target indices
                        pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # prediction indices
                        if pi.shape[0]:
                            ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices
                            detected_set = set()
                            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                d = ti[i[j]]  # detected target
                                if d.item() not in detected_set:
                                    detected_set.add(d.item())
                                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                    if len(detected_set) == nl:
                                        break

            # Append statistics (correct, conf, pcls, tcls)
            if v9:
                stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))
            else:
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

            # Save / log
            if save_txt:
                file = save_dir / 'labels' / f'{path.stem}.txt'
                if v9:
                    save_one_txt(predn, save_conf, shapes[si][0], file=file)
                else:
                    gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                    for *xyxy, conf, cls in predn.tolist():
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(file, 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if save_json:
                if v9:
                    save_one_json(predn, jdict, path, coco91class)  # append to COCO-JSON dictionary
                else:
                    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                    box = xyxy2xywh(predn[:, :4])  # xywh
                    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                    for p, b in zip(predn.tolist(), box.tolist()):
                        jdict.append({'image_id': image_id,
                                    'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                    'bbox': [round(x, 3) for x in b],
                                    'score': round(p[4], 5)})

        # Compute intermediate metrics for log-viz -----------------------------
        statsn = deepcopy(stats)
        if v9:
            statsn = [torch.cat(x,0).cpu().numpy() for x in zip(*statsn)]
        else:
            statsn = [np.concatenate(x, 0) for x in zip(*statsn)]
        latency = _t0 + _t1 + _t2

        report_progress(
            userid,
            project_id,
            statsn,
            seen,
            label_cnt,
            batch_i+1,
            len(dataloader),
            plots,
            metric,
            save_dir,
            names,
            latency,
            start_time,
            can_update_progress,
        )
    # Test end =================================================================

    # Compute final metrics ----------------------------------------------------
    if v9:
        stats = [torch.cat(x,0).cpu().numpy() for x in zip(*stats)]
    else:
        stats = [np.concatenate(x, 0) for x in zip(*stats)]

    ap, ap_class = [], []
    if len(stats) and stats[0].any():
        if metric == 'v9':
            tp, fp, p, r, f1, ap, ap_class = ap_per_class_v9(
                *stats, plot=plots, save_dir=save_dir, names=names
            )
        else:
            p, r, ap, f1, ap_class = ap_per_class(
                *stats, plot=plots, metric=metric, save_dir=save_dir, names=names
            )
        ap50, ap = ap[:, 0], ap.mean(1)
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    else:
        mp = mr = map50 = map = 0.0
    nt = np.bincount(stats[3].astype(int), minlength=nc) # number of targets per class

    # Print results ------------------------------------------------------------
    logger.info('-'*100)
    if nt.sum() == 0:
        logger.warning(f'WARNING: no labels found in dataset, can not compute metrics w/o labels')
    pf_c = '%22s' + '%11i' * 2 + '%11.3g' * 4
    logger.info(pf_c % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class --------------------------------------------------
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            logger.info(pf_c % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds -------------------------------------------------------------
    t = tuple(x / seen * 1E3 for x in (t0, t1, t2))
    if not training:
        logger.info('Speed(avg.): %.1fms pre-process, %.1fms inference, %.1fms nms per images' % t)

    # Plots --------------------------------------------------------------------
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))

    # Save JSON ----------------------------------------------------------------
    if save_json: # and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        DATASET_ROOT = Path("/shared/datasets")
        anno_json = str(DATASET_ROOT / 'coco' / 'annotations' / 'instances_val2017.json')
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        logger.info('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            logger.warning(f'pycocotools unable to run: {e}')

    # Return results -----------------------------------------------------------
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        logger.info(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def test_cls(proj_info,
             data,
             weights=None,
             batch_size=32,
             imgsz=640,
             augment=False,
             verbose=False,
             model=None,
             dataloader=None,
             save_dir=Path(''),  # for saving images
             save_txt=False,  # for auto-labelling
             save_hybrid=False,  # for hybrid auto-labelling
             save_conf=False,  # save auto-label confidences
             plots=False,
             compute_loss=None,
             half_precision=True):
    # Set device ---------------------------------------------------------------
    training = model is not None
    if training: # called by train.py
        device = next(model.parameters()).device  # get model device
    else: # called directly
        # logging
        set_logging()

        # device
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model

    # Half ---------------------------------------------------------------------
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure ----------------------------------------------------------------
    userid = (proj_info or {}).get('userid') if isinstance(proj_info, dict) else None
    project_id = (proj_info or {}).get('project_id') if isinstance(proj_info, dict) else None
    can_update_progress = training and userid and project_id

    model.eval()

    # Dataset ------------------------------------------------------------------
    # if isinstance(data, str):
    #     with open(data) as f:
    #         data = yaml.load(f, Loader=yaml.SafeLoader)
    # test_path = data['val']
    # is_imgnet = True if data['dataset_name'] == 'imagenet' and 'imagenet' in test_path else False

    # Test start ===============================================================
    val_loss = torch.zeros(1, device=device)
    val_accuracy = torch.zeros(1, device=device)
    accumulated_data_count = 0
    elapsed_time = 0
    val_acc = {}
    for i, (img, target) in enumerate(dataloader):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()
        target = target.to(device)

        with torch.no_grad():
            # run model
            t = time_synchronized()
            output = model(img, augment=augment)

            # compute loss
            if compute_loss:
                loss = compute_loss(output, target)
                val_loss += loss.item()

            # compute top-1 accuracy
            _, predicted = output.max(1)
            acc = torch.eq(predicted, target).sum()
            val_accuracy += acc
            t0 = time_synchronized()-t

        accumulated_data_count += len(target)
        elapsed_time += t0
        # status update
        if can_update_progress: # called by train.py
            try:
                step = i + 1
                total_step = len(dataloader)
                images_so_far = accumulated_data_count
                num_classes = int(output.size(1))  # for reference

                # 평균 loss와 top1 정확도 계산 (안전 나눗셈)
                avg_loss = (val_loss.item() / step) if step > 0 else float('nan')
                top1 = (val_accuracy.item() / images_so_far) if images_so_far > 0 else 0.0

                val_acc = {
                    'step': step,
                    'total_step': total_step,
                    'images': images_so_far,
                    'labels': num_classes,         # 분류 task에서는 "클래스 수"로 사용 중
                    'P': images_so_far,            # 기존 스키마 호환(원래의 의미와 다름)
                    'R': val_accuracy.item(),      # 누적 correct 수
                    'mAP50': avg_loss,             # <- 분류: 평균 loss(스키마 호환용 필드)
                    'mAP50-95': top1,              # <- 분류: top-1 accuracy
                    'time': f'{t0:.1f} s'
                }
                safe_status_update(
                    userid, project_id,
                    update_id="val_accuracy",
                    update_content=val_acc
                )
            except Exception as e:
                if verbose:
                    logger.debug(f"[test.py] status update skipped: {e}")
                pass
    # Test end =================================================================

    # Compute statistics -------------------------------------------------------
    val_loss /= len(dataloader)
    if len(dataloader.dataset) != accumulated_data_count:
        logger.warning(f"total data = {accumulated_data_count} is not match with lenght of dataset = {len(dataloader.dataset)}")

    logger.info(f'total = {accumulated_data_count}, '
                f'correct={val_accuracy.item()}, '
                f'accuracy = {val_accuracy.item()/accumulated_data_count}')
    val_accuracy /= len(dataloader.dataset)

    # Print results ------------------------------------------------------------

    model.float()
    return (val_accuracy.item(), val_loss.item()), elapsed_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--purpose', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store-true', help='save confidences in --save-txt labels')  # NOTE: original had action='store_true'; keep as-is if needed
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half-precision', dest='half_precision', action='store_true',
                        help='use half precision (FP16) for inference on CUDA devices')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--is-coco', action='store_true', help='enable COCO evaluation with pycocotools')
    parser.add_argument('--metric', type=str, default='v5', help='mAP metrics; v5/v7/v9')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    #check_requirements()

    if opt.purpose in ('train', 'val', 'test'):  # run normally
        # cd /source/autonn_core
        # python -m tango.main.test \
        #   --purpose val \
        #   --weights ../archive/bestmodel.pt \
        #   --data /shared/datasets/coco/dataset.yaml \
        #   --img-size 1280 \
        #   --batch-size 8 \
        #   --device 0 \
        #   --iou-thres 0.75 \
        #   --save-json \
        #   --no-trace \
        #   --is-coco \
        #   --metric v9
        test(
            proj_info=None,
            data=opt.data,
            weights=opt.weights,
            batch_size=opt.batch_size,
            imgsz=opt.img_size,
            conf_thres=opt.conf_thres,
            iou_thres=opt.iou_thres,
            single_cls=opt.single_cls,
            augment=opt.augment,
            verbose=opt.verbose,
            save_txt=opt.save_txt | opt.save_hybrid,
            save_hybrid=opt.save_hybrid,
            save_conf=opt.save_conf,
            save_json=opt.save_json,
            half_precision=opt.half_precision,
            trace=not opt.no_trace,
            is_coco=opt.is_coco,
            metric=opt.metric
        )

    elif opt.purpose == 'speed':  # speed benchmarks
        for w in opt.weights:
            test(
                proj_info=None,
                data=opt.data,
                weights=w,
                batch_size=opt.batch_size,
                imgsz=opt.img_size,
                conf_thres=0.25,
                iou_thres=0.45,
                save_json=False,
                plots=False,
                metric=opt.metric
            )

    elif opt.purpose == 'study':  # run over a range of settings and save/plot
        # cd /source/autonn_core
        # python -m tango.main.test \
        #   --purpose study \
        #   --data coco.yaml \
        #   --iou-thres 0.65 \
        #   --weights yolov9-e.pt \
        #   --no-trace \
        #   --is-coco \
        #   --metric v9
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                logger.info(f'\nRunning {f} point {i}...')
                r, _, t = test(
                    proj_info=None,
                    data=opt.data,
                    weights=w,
                    batch_size=opt.batch_size,
                    imgsz=i,
                    conf_thres=opt.conf_thres,
                    iou_thres=opt.iou_thres,
                    save_json=opt.save_json,
                    plots=False,
                    is_coco=opt.is_coco,
                    metric=opt.metric
                )
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot
