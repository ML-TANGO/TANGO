import argparse
import logging
import json
import os
from pathlib import Path
from threading import Thread
from copy import deepcopy
import numpy as np
import torch
import yaml
from tqdm import tqdm

from . import status_update, Info

from tango.common.models.experimental import attempt_load
from tango.utils.datasets import create_dataloader
from tango.utils.general import (   coco80_to_coco91_class,
                                    check_dataset,
                                    check_file,
                                    check_img_size,
                                    check_requirements,
                                    box_iou,
                                    non_max_suppression,
                                    non_max_suppression_v9,
                                    scale_coords,
                                    xyxy2xywh,
                                    xywh2xyxy,
                                    set_logging,
                                    increment_path,
                                    colorstr
                                )
from tango.utils.metrics import ap_per_class, ConfusionMatrix
from tango.utils.plots import plot_images, output_to_target, plot_study_txt
from tango.utils.torch_utils import select_device, time_synchronized, TracedModel

logger = logging.getLogger(__name__)

def test(proj_info,
         data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001, # for NMS
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         # wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         trace=False,
         is_coco=False,
         metric='v5'):
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
    if half:
        model.half()
    else:
        model.float()

    # Configure ----------------------------------------------------------------
    userid = proj_info['userid']
    project_id = proj_info['project_id']

    # save internally
    info = Info.objects.get(userid=userid, project_id=project_id)
    info.status = "running"
    info.progress = "validation"
    info.save()
    # print('test() enters')
    # info.print()

    model.eval()
    if isinstance(data, str):
        # is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)  # check
    test_path = data['val']
    is_coco = True if data['dataset_name'] == 'coco' and 'coco' in test_path else False

    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging ------------------------------------------------------------------
    # log_imgs = 0
    # if wandb_logger and wandb_logger.wandb:
    #     log_imgs = min(wandb_logger.log_imgs, 100)

    # Dataloader ---------------------------------------------------------------
    if not training: # called directly
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        purpose = opt.purpose if opt.purpose in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[purpose], imgsz, batch_size, gs, opt, pad=0.5, rect=True,
                                       prefix=colorstr(f'{purpose}: '))[0]

    # Metrics ------------------------------------------------------------------
    if metric == 'v5':
        logger.info(f"\n{colorstr('Test: ')}Testing with YOLOv5 AP metric..."
                    f"You can change it metric = 'v7' in args.yaml")
        
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

    val_acc = {}
    seen, label_cnt = 0, 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    coco91class = coco80_to_coco91_class() if is_coco else list(range(1000))
    # s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    # logger.info(s)
    tp, fp = 0., 0.
    p, r, f1, mp, mr, map50, map  = 0., 0., 0., 0., 0., 0., 0.
    t, t0, t1, t2 = 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    # for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
    for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        logger.info(f"      Batch #{batch_i}: {len(paths)} imgs")
        # print('\t'+'_'*100)
        # for cnt, path in enumerate(paths):
        #     s = ''
        #     for target in targets:
        #         idx = int(target[0].item())
        #         if cnt == idx:
        #             cls_idx = int(target[1].item())
        #             s += f"{cls_idx} "
        #     print(f"\t{cnt}: {path}, [ {s}]")
        # print('\t'+'_'*100)

        with torch.no_grad():
            # Run model --------------------------------------------------------
            t = time_synchronized()
            out, train_out = model(img, augment=augment)  # inference and training outputs
            '''
            for single anchor-based head (v7)
                out.shape = (bs, 25200, 85) : tensor
                train_out.shape = ( (bs,3,80,80,85), (bs,3,40,40,85), (bs,3,20,20,85) ) : list
            for dual anchor-free heads (v9)
                out.shape = ( (bs, 84, 8400), (bs, 84, 8400) ) : list
                train_out.shape = ( ((bs,80,80,84), (bs,40,40,84), (bs,20,20,84)),
                                    ((bs,80,80,84), (bs,40,40,84), (bs,20,20,84))  ) : list
            '''
            if v9 and nh > 1:
                out = out[nh-1]
                # train_out = train_out[nh-1]
            _t0 = time_synchronized() - t
            t0 += _t0
            logger.info(f"      Batch #{batch_i}: Running model ({1E3*_t0:.2f} ms)")

            # Compute loss -----------------------------------------------------
            if compute_loss:
                if v9:
                    loss = compute_loss(train_out, targets)[1][:3]
                else:
                    loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj(dfl), cls
                logger.info(f"      Batch #{batch_i}: Computing loss - {loss}")

            # Run NMS ----------------------------------------------------------
            '''
            after nms,
                out = (bs, n, 6) : bs = batch size, n = detected object number, 6 = (x,y,w,h,conf,cls)
                (v7) (bs, 25200, 85) ==> NMS   ==> approx. (bs, n, 6)
                (v9) (bs, 84, 8400)  ==> NMSv9 ==> approx. (bs, n, 6)
                in fact, 'out' is list-type and it can have a different 'n' which image it is on
                out = [ [n1, 6], [n2, 6], ... [nbs, 6] ]

            nms terms for codegen :
              n       = num_detections,
              x,y,w,h = nmsed_boxes (x1,y1,x2,y2),
              conf    = nmsed_scores,
              cls     = nmsed_classes
            '''
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t = time_synchronized()
            if v9:
                out = non_max_suppression_v9(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
            else:
                out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
            _t1 = time_synchronized() - t
            t1 += _t1
            logger.info(f"      batch #{batch_i}: Running NMS   ({_t1*1E3:.2f} ms)")

        # Metrics per image ----------------------------------------------------
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1
            label_cnt += nl

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Evaluate : assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # target indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # prediction indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item()) # set of int
                                detected.append(d) # list of tensor
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break
            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

            # Save / log
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
            
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})
                    
        # Plot images ----------------------------------------------------------
        # if plots and batch_i < 3:
        #     f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
        #     Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
        #     f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
        #     Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

        _p, _r, _f1, _mp, _mr, _map50, _map = 0., 0., 0., 0., 0., 0., 0.
        statsn, _ap, _ap_class = [], [], []
        statsn = deepcopy(stats)
        statsn = [np.concatenate(x, 0) for x in zip(*statsn)]  # to numpy
        if len(statsn) and statsn[0].any():
            _p, _r, _ap, _f1, _ap_class = ap_per_class(*statsn, plot=plots, metric=metric, save_dir=save_dir, names=names)
            _ap50, _ap = _ap[:, 0], _ap.mean(1)  # AP@0.5, AP@0.5:0.95
            _mp, _mr, _map50, _map = _p.mean(), _r.mean(), _ap50.mean(), _ap.mean()
        # Status update
        # label_cnt += targets.size(0)
        val_acc['class'] = 'all'
        val_acc['images'] = seen
        val_acc['labels'] = label_cnt
        val_acc['P'] = _mp
        val_acc['R'] = _mr
        val_acc['mAP50'] = _map50
        val_acc['mAP50-95'] = _map
        val_acc['step'] = batch_i + 1
        val_acc['total_step'] = len(dataloader)
        val_acc['time'] = f'{(t0+t1)-t2:.1f} s'
        status_update(userid, project_id,
                      update_id="val_accuracy",
                      update_content=val_acc)
        t2 = t0 + t1
        s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        logger.info(f"\n{s}")
        logger.info(f"{'all':>20s}{seen:12d}{label_cnt:12d}{_mp:12.3f}{_mr:12.3f}{_map50:12.3f}{_map:12.3f}")
    # Test end =================================================================

    # Compute statistics -------------------------------------------------------
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, metric=metric, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results ------------------------------------------------------------
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    logger.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # nt_sum_value = nt.sum().item()
    # val_acc['class'] = 'all'
    # val_acc['images'] = seen
    # val_acc['labels'] = nt_sum_value
    # val_acc['P'] = mp
    # val_acc['R'] = mr
    # val_acc['mAP50'] = map50
    # val_acc['mAP50-95'] = map
    # # val_acc['time'] = f'{(t0 + t1) :.1f} s' # total time
    # status_update(userid, project_id,
    #           update_id="val_accuracy",
    #           update_content=val_acc)


    # Print results per class --------------------------------------------------
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            logger.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds -------------------------------------------------------------
    # t = (inference time, nms time, total time, imgsz, imgsz, batchsize)
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)
    if not training:
        logger.info('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots --------------------------------------------------------------------
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
    #     if wandb_logger and wandb_logger.wandb:
    #         val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
    #         wandb_logger.log({"Validation": val_batches})
    # if wandb_images:
    #     wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # Save JSON ----------------------------------------------------------------
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        DATASET_ROOT = Path("/shared/datasets")
        anno_json = str(DATASET_ROOT / 'coco' / 'annotations' / 'instances_val2017.json')
        # anno_json = './coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        logger.info('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
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
        set_logginig()

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
    userid = proj_info['userid']
    project_id = proj_info['project_id']
    model.eval()

    # Dataset ------------------------------------------------------------------
    # if isinstance(data, str):
    #     with open(data) as f:
    #         data = yaml.load(f, Loader=yaml.SafeLoader)
    # test_path = data['val']
    # is_imgnet = True if data['dataset_name'] == 'imagenet' and 'imagenet' in test_path else False

    # Test start ===============================================================
    # val_loss = 0
    # val_acc = 0
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
            # val_acc += predicted.eq(target.view_as(predicted)).sum().item()
            val_accuracy += acc
            t0 = time_synchronized()-t

        accumulated_data_count += len(target)
        elapsed_time += t0
        # status update
        val_acc['step'] = i + 1
        val_acc['images'] = accumulated_data_count
        val_acc['labels'] = output.size(1) # val_accuracy.item()
        val_acc['P'] = accumulated_data_count
        val_acc['R'] = val_accuracy.item()
        val_acc['mAP50'] = val_loss.item() / (i+1)
        val_acc['mAP50-95'] = val_accuracy.item() / accumulated_data_count
        val_acc['total_step'] = len(dataloader)
        val_acc['time'] = f'{t0:.1f} s'
        status_update(userid, project_id,
                      update_id="val_accuracy",
                      update_content=val_acc)
    # Test end =================================================================

    # Compute statistics -------------------------------------------------------
    val_loss /= len(dataloader)
    if len(dataloader.dataset) != accumulated_data_count:
        logger.warn(f"total data = {accumulated_data_count} is not match with lenght of dataset = {len(dataloader.dataset)}")

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
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--metric', type=str, default='v5', help='mAP metrics; v5/v7/v9')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    #check_requirements()

    if opt.purpose in ('train', 'val', 'test'):  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             trace=not opt.no_trace,
             metric=opt.metric
             )

    elif opt.purpose == 'speed':  # speed benchmarks
        for w in opt.weights:
            test(opt.data, w, opt.batch_size, opt.img_size, 0.25, 0.45, save_json=False, plots=False, metric=opt.metric)

    elif opt.purpose == 'study':  # run over a range of settings and save/plot
        # python test.py --purpose study --data coco.yaml --iou 0.65 --weights yolov7.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                logger.info(f'\nRunning {f} point {i}...')
                r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False, metric=opt.metric)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot
