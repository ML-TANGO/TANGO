import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import shutil
import yaml
from tqdm import tqdm
import subprocess

import csv

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_yaml, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized, TracedModel

from selector import SVMPredictor, entropy


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
print("root folder is", str(ROOT))  # 보통 . 출력


def select(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
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
         wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         trace=False,
         is_coco=False,
         v5_metric=False,
         size=0
         ):
    device = select_device(opt.device, batch_size=batch_size)

    # Half
    # half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA

    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)  # check

    # Dataloader
    task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
    data[task] = Path("evaluate2017.txt")
    dataloader = create_dataloader(data[task], 640, batch_size, 32, opt, pad=0.5, rect=True,
                                    prefix=colorstr(f'{task}: '))[0]
        
    print(os.getcwd())
    subprocess.run(["python", "generate_txt_file.py"])

    selector = SVMPredictor(9, 7, "../test.pt")
    selector._model.eval()
    
    selected = [0 for i in range(size)]
    
    
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader)):
        # img = img.to(device, non_blocking=True)
        # img = img.half() if half else img.float()  # uint8 to fp16/32
        # img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        
        entropies = []
        for path in paths:
            input_entropies = entropy(path)
            entropies.append(input_entropies)
            
        entropies = torch.Tensor(np.array(entropies)).to(device)
            
        outputs = selector.forward(entropies)

        for output in outputs:
            selected_index = torch.argmax(output).item()
            selected[selected_index] += 1
    
    return np.argmax(selected)



def parse_opt(docker):
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, evaluate, speed or study')
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
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    
    if docker:
        opt = parser.parse_args(["--task", "evaluate", "--batch-size", "4", "--data", "coco.yaml", "--img", "640", "--iou", "0.65", "--half"])
    else:
        opt = parser.parse_args()
        
    opt.data = check_yaml(opt.data)
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    # print(opt)
    return opt



def main(opt, userid="", project_id=""):    
    
    if opt.task in ('train', 'val', 'test', 'evaluate'):  # run normally
        
        models = ["yolov7.pt", "yolov7x.pt", "yolov7-tiny.pt", "yolov7-w6.pt", "yolov7-d6.pt", "yolov7-e6.pt", "yolov7-e6e.pt"]
        size = len(models)
        
        # for model in models:
        val = select(opt.data,
                models[0],
                opt.batch_size,
                opt.img_size,
                opt.conf_thres,
                opt.iou_thres,
                False,
                opt.single_cls,
                opt.augment,
                opt.verbose,
                save_txt=opt.save_txt | opt.save_hybrid,
                save_hybrid=opt.save_hybrid,
                save_conf=opt.save_conf,
                trace=not opt.no_trace,
                v5_metric=opt.v5_metric,
                size=size
                )
        
        print("Most used was " + models[val])
        
        model = attempt_load(models[val], map_location="cpu")  # load FP32 model

        print("Starting the main inference with the best model, ", models[val])
        if os.path.isfile(models[val]):
            shutil.copy(models[val], Path('/shared/common/')/ userid / project_id)
            shutil.copy(os.path.join(Path.cwd(), ROOT, "cfg/training/", models[val][:-3]  + ".yaml"),  Path('/shared/common/')/ userid / project_id / "basemodel.yaml")
            # os.rename(os.path.join(Path('/shared/common/')/ userid / project_id), "/" , models[val][:-3]  + ".yaml", os.path.join(Path('/shared/common/')/ userid / project_id), "./basemodel.yaml")
            #shutil.copy(os.path.join(Path.cwd(), ROOT, "models/", models[val][:-2]  + "yaml"),  Path.cwd() / userid / project_id)
            
        
def docker_run(userid, project_id):
    opt = parse_opt(True)
    print("here is the opt\n")
    print(opt)
    main(opt, userid, project_id) 
    
if __name__ == "__main__":
    opt = parse_opt(False)
    print("here is the opt\n")
    print(opt)
    main(opt)
