# Pre_model_select Branch
Pre model select branch searches for model that showed the most efficient performance among yolov5 models.

|![model overview](images/model_comparison.png)|
|:--:|
| <b>Image Credits - Fig.1 - Yolov5 Pytorch Website</b>|
----
Yolov5 includes a family of networks for object detection with different complexity that was initially trained on the COCO dataset. Depending on the characteristics of the dataset, pre_model_select model chooses the best model that showed the best performance evaluated from the accuracy and inference time of each model.

Pre_model_select network uses docker environment to deploy(download) and execute the selection process. Currently, It randomly select 200 images in COCO 2017 dataset to determine the best model and finish the entire dataset with the chosen network. The instruction to run the network is explained [below](#docker-image-build-and-run).

## Docker Image build and run

### step 1
change current working directory into pre_model_select.
```bash
docker build -t yolo_docker .
```

### step 2
```bash
docker run --rm --ipc=host yolo_docker 
```
this process will download the coco2017 val dataset and run for the five models in yolov5(sample of 200 images), and choose the best model based on their efficiency, and run the entire dataset using the chosen best model.

