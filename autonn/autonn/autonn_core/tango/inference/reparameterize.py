import os

from pathlib import Path
COMMON_ROOT = Path("/shared/common")
DATASET_ROOT = Path("/shared/datasets")
MODEL_ROOT = Path("/shared/models")
CORE_DIR = Path(__file__).resolve().parent.parent.parent # /source/autonn_core
CFG_PATH = CORE_DIR / 'tango' / 'common' / 'cfg'

import torch

from tango.common.models.yolo import Model

import logging
logger = logging.getLogger(__name__)

def convert_small_model(model, ckpt):
    idx = 0
    for k, v in model.state_dict().items():
        if "model.{}.".format(idx) in k:
            if idx < 22:
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(k, "perfectly matched!!")
            elif "model.{}.cv2.".format(idx) in k:
                kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(k, "perfectly matched!!")
            elif "model.{}.cv3.".format(idx) in k:
                kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(k, "perfectly matched!!")
            elif "model.{}.dfl.".format(idx) in k:
                kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(k, "perfectly matched!!")
        else:
            while True:
                idx += 1
                if "model.{}.".format(idx) in k:
                    break
            if idx < 22:
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(k, "perfectly matched!!")
            elif "model.{}.cv2.".format(idx) in k:
                kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(k, "perfectly matched!!")
            elif "model.{}.cv3.".format(idx) in k:
                kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(k, "perfectly matched!!")
            elif "model.{}.dfl.".format(idx) in k:
                kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(k, "perfectly matched!!")
    _ = model.eval()
    return model


def convert_medium_model(model, ckpt):
    idx = 0
    for k, v in model.state_dict().items():
        if "model.{}.".format(idx) in k:
            if idx < 22:
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+1))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(k, "perfectly matched!!")
            elif "model.{}.cv2.".format(idx) in k:
                kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+16))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(k, "perfectly matched!!")
            elif "model.{}.cv3.".format(idx) in k:
                kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+16))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(k, "perfectly matched!!")
            elif "model.{}.dfl.".format(idx) in k:
                kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+16))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(k, "perfectly matched!!")
        else:
            while True:
                idx += 1
                if "model.{}.".format(idx) in k:
                    break
            if idx < 22:
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+1))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(k, "perfectly matched!!")
            elif "model.{}.cv2.".format(idx) in k:
                kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+16))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(k, "perfectly matched!!")
            elif "model.{}.cv3.".format(idx) in k:
                kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+16))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(k, "perfectly matched!!")
            elif "model.{}.dfl.".format(idx) in k:
                kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+16))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(k, "perfectly matched!!")
    _ = model.eval()
    return model


def convert_large_model(model, ckpt):
    idx = 0
    for k, v in model.state_dict().items():
        if "model.{}.".format(idx) in k:
            if idx < 29:
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(k, "perfectly matched!!")
            elif idx < 42:
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(k, "perfectly matched!!")
            elif "model.{}.cv2.".format(idx) in k:
                kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(k, "perfectly matched!!")
            elif "model.{}.cv3.".format(idx) in k:
                kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(k, "perfectly matched!!")
            elif "model.{}.dfl.".format(idx) in k:
                kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(k, "perfectly matched!!")
        else:
            while True:
                idx += 1
                if "model.{}.".format(idx) in k:
                    break
            if idx < 29:
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(k, "perfectly matched!!")
            elif idx < 42:
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(k, "perfectly matched!!")
            elif "model.{}.cv2.".format(idx) in k:
                kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(k, "perfectly matched!!")
            elif "model.{}.cv3.".format(idx) in k:
                kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(k, "perfectly matched!!")
            elif "model.{}.dfl.".format(idx) in k:
                kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(k, "perfectly matched!!")
    _ = model.eval()
    return model




def get_yolov9_converted_model(model_pt, cfg):
    device = torch.device("cpu")

    if not os.path.isfile(model_pt):
        logger.warning(f"Model Export: not found {model_pt}")
        return None

    if not os.path.isfile(cfg):
        logger.warning(f"Model Export: not found {cfg}")
        return ckpt
    
    model = Model(cfg, ch=3, nc=80, anchors=3).to(device) # create empty model
    _ = model.eval()

    ckpt = torch.load(model_pt, map_location='cpu')
    model.names = ckpt['model'].names
    model.nc = ckpt['model'].nc

    cfg_name = os.path.basename(cfg).lower()
    if '-t' in cfg_name or '-s' in cfg_name:
        model = convert_small_model(model, ckpt)    
    elif '-m' in cfg_name or '-c' in cfg_name:
        model = convert_medium_model(model, ckpt)
    else #if '-e' in cfg_name:
        model = convert_large_model(model, ckpt)
