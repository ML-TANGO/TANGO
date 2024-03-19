#!/bin/sh
chmod 755 yolov7/predict.py
cd yolov7
python predict.py --data data/coco.yaml --img 640 --iou 0.65