#!/bin/sh
chmod 755 yolov5/predict.py
python yolov5/predict.py --task evaluate --data coco.yaml --img 640 --iou 0.65 --half