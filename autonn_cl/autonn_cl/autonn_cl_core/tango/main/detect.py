import time
from pathlib import Path
COMMON_ROOT = Path("/shared/common")
DATASET_ROOT = Path("/shared/datasets")
CORE_DIR = Path(__file__).resolve().parent.parent.parent # /source/autonn_cl_core
CFG_PATH = CORE_DIR / 'tango' / 'common' / 'cfg'

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from tango.inference.libTango import ModelLoader
from tango.utils.datasets import LoadWebcam, LoadImages
from tango.utils.general import (   check_img_size,
                                    check_file, 
                                    check_imshow,
                                    non_max_suppression,
                                    non_max_suppression_v9,
                                    scale_coords, 
                                    xyxy2xywh,      ) 
from tango.utils.plots import plot_one_box
from tango.utils.torch_utils import (   select_device, 
                                        time_synchronized   )

import logging
logger = logging.getLogger(__name__)

def detect(weights, cfg, view_img=True, save_img=False, save_txt=False):
    # options ------------------------------------------------------------------
    if not weights:
        weights = CORE_DIR / 'tango' / 'inference' / 'yolov9-m.pt'
    if not cfg:
        cfg = CFG_PATH / 'yolov9' / 'yolov9-m.yaml' 
    weights, cfg = check_file(weights), check_file(cfg)     
                                                 
    imgsz = 640
    # since it is hard to use cv2.imshow() and webcam in docker container...                                                     
    view_img = False                                                
    save_txt = True
    save_img = True

    # source = 1 # webcam
    # source = DATASET_ROOT / 'coco' / 'images' / 'val2017' / '*.jpg'  
    source = CORE_DIR / 'tango' / 'inference' / 'horses.jpg'
    source = str(source)                                                
    webcam = source.isnumeric()

    logger.info(f'weights={weights}, cfg={cfg}')

    # Directories --------------------------------------------------------------
    save_dir = CORE_DIR / 'tango' / 'inference'

    # Initialize ---------------------------------------------------------------
    device = select_device()
    half = False # device.type != 'cpu'  # half precision only supported on CUDA

    logger.info(f"device={device}, half={half}")
          
    # Load model ---------------------------------------------------------------
    model = ModelLoader(weights=weights, cfg=cfg, task='detection', device=device, fuse=True)  # load FP32 model
    stride = model.stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # if half:
    #     print(f"model precision : half(fp16)")
    #     model.half()  # to FP16

    logger.info(f"success to load model: .pt model ? {model.pt}, stride={stride}")

    # Set Dataloader -----------------------------------------------------------
    vid_path, vid_writer = None, None
    # if webcam:
    #     view_img = check_imshow()
    #     print(f'check_imshow: {view_img}')
    #     dataset = LoadWebcam(source, img_size=imgsz, stride=stride)
    # else:
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=model.pt)

    logger.info(f'loaded {len(dataset)} images from {source}')

    # Get names and colors -----------------------------------------------------
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # warmup -------------------------------------------------------------------
    logger.info('warmup')
    model.warmup(imgsz=(1, 3, imgsz, imgsz))

    # Run inference ============================================================
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        # Image shaping --------------------------------------------------------
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0) # make a tensor with bs=1 

        # Inference ------------------------------------------------------------
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            # pred = model(img, augment=opt.augment)[0]
            pred = model(img)
            if isinstance(pred, (list, tuple)):
                pred = pred[-1] # for dual or triple heads
        logger.info(f"\t1. predict  : {pred.shape}")
        t2 = time_synchronized()

        # Apply NMS ------------------------------------------------------------
        pred = non_max_suppression_v9(pred) #, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        logger.info(f"\t2. nms      : {len(pred)} imgs, {pred[0].shape}")
        t3 = time_synchronized()
        
        # Process detections ---------------------------------------------------
        for i, det in enumerate(pred):  # detections per image
            logger.info(f"\t3. draw  {len(det)} boxes")
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.stem) + '_result' + str(p.suffix) # img_result.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            txt_path = str(save_dir / p.stem) + '_result.txt' # img_result.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        line = (cls, *xywh, conf)
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS) -------------------------------------
            logger.info(f'\t - {s}({1E3 * (t2 - t1):.1f}ms) for Inference, ({1E3 * (t3 - t2):.1f}ms) for NMS')

            # Stream results ---------------------------------------------------
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections) -----------------------------
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    logger.info(f"\t - the image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
    # End inference ============================================================

    # if save_txt or save_img:
    #     s = f"\n{len(list(save_dir.glob('*.txt')))} labels saved to {save_dir}" if save_txt else ''
    #     print(f"Results saved to {save_dir}{s}")

    logger.info(f'\t4. all done. total elapsed time for {len(pred)} imgs = ({time.time() - t0:.3f}s)')
