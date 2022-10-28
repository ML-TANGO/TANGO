# TANGO YAML file guide

**Table of Contents**
* [YAML format for project info](#project_info)
* [YAML format for AutoNN](#auto_nn)
  * [YAML format for dataset info](#dataset_info)
  * [YAML format for taget device info](#device_info)
  * [YAML format for base model infoi](#base_model_info)
* [YAML format for neural network deployment info](#nn_deploy_info)
* [YAML format for neural network exeuction info](#nn_exec_info)

----

## YAML format for project info <a name="project_info"></a>

```yaml
# Common Info
target_info : cloud/pc/Odroid-N1/Odroid-M1   # target device type, defualt cloud
cpu : x86/arm                                # cpu type, default  x86 
acc : cpu/cuda/opencl                        # accelerator type, default  cpu 
os : linux/windows                           # os type, default  linux 
engine  : pytorch/acl/rknn                   # deeplearning framework, default  pytorch 
target_ip : ipaddress                        # target device ip address, default  NULL (clould or target device IP address)
target_port : 8088                           # target device port info, default  NULL (clould or target device port)


# Info. on autonn
dataset : path/to/dataset.yaml               # YAML file path for dataset info
target: path/to/target.yaml                  # YAML file path for deploy target device info
basemodel: path/to/basemodel.yaml            # YAML file path for base model info 
deployment: path/to/deployment.yaml          # YAML file path for deploy info


# Info on deployment
lightweight_level : integer in [0..10]       # model lightness, default 0 -> no optimization
precision_level : interger in [0..10]        # model precision, default 10 -> no optimization
preprocessing_lib : numpy/opencv             # library info, default numpy
input_method : picture/moving_picture/camera # inference input method, default picture
input_data_location : /data                  # path fo inference input image files
output_method : text/graphic                 # inference output format 
user_editing : no/yes                        # allow or disallow of user editing of application, default no
```

----

## YAML format for AutoNN <a name="auto_nn"></a>

### YAML format for dataset info <a name="dataset_info"></a>

Following example YAML illustrates usage with COCO dataset.

For training/validation/test dataset info, you can use onel of the following three format
* directory path format: `path/to/imgs`
* file path format: `path/to/imgs.txt`
* list format : `[path/to/imgs1, path/to/imgs2, …]`

```YAML
path: /datasets_root/COCO
imgs: /datasets_root/COCO/JPEGImages
annos: /datasets_root/COCO/Annotations
train: /datasets_root/COCO/ImageSets/train.txt
val: /datasets_root/COCO/ImageSets/val.txt

# class (label) info
num_classes: 80 # number of class
names: [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
```

### YAML format for taget device info <a name="device_info"></a>

Following example YAML illustrates usage with ODROID N2+ target device.

In this example, we assume that target specific separated directories configured in advance and following example YAML file exists in the corresponding target specific directory.
```yaml
target: /target_root/odroidn2plus
device: gpu
acc: opencl
engine: armnn

# following constarint info is optional, 
# if omitted, the info. defined in autonn target is used.
min_latency: 7.5    # minimal latency, unit: fps
max_power: None     # maximul power allowed, unit: mW
max_model_size: 20  # maximum model size allowed, unit: MB
max_flops: None     # maximum FLOPs,  unit: flops
max_peak_memory:    # peak memory allowed, unit: MB 
```

### YAML format for base model info <a name="base_model_info"></a>

Following example YAML illustrates usage with YOLOv5 backbone model.

```yaml
backbone:
  [[-1, 1, Focus, [64, 3]],     # 0-P1/2
   [-1, 1, CBS, [128, 3, 2]],   # 1-P2/4
   [-1, 3, CSP, [128]],
   [-1, 1, CBS, [256, 3, 2]],   # 3-P3/8
   [-1, 9, CSP, [256]],         # 4 -> neck: cat

   [-1, 1, CBS, [512, 3, 2]],   # 5-P4/16
   [-1, 9, CSP, [512]],         # 6 -> neck: cat

   [-1, 1, CBS, [1024, 3, 2]],  # 7-P5/32 -> neck: spp or something else
  ]
```

---

## YAML format for neural network deployment info <a name="nn_deploy_info"></a>

```yaml
# 'neural_net_info.yaml'
# meta file from auto_nn

# NN Model
class_file: ['bestmodel.py', 'ops.py']
class_name: 'TheBestmodelClass()'
weight_file: bestmodel.pt

# usage ex.
# model = TheBestModelClass()
# model.load_state_dict(torch.load('bestmodel.pt'))
# model.eval()


# Label
nc: 80 # number of classes
label_info_file: labelmap.yaml

# labelmap.yaml ex.
# nc: 80 # number of classes
# names: [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
#          'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#          'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#          'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
#          'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#          'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#          'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
#          'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
#          'hair drier', 'toothbrush' ]
# task: detection



# Input
input_tensor_shape: [1, 3, 604, 604]
input_data_type: fp32           # fp32, fp16, int8, etc
# anchors: 3
anchors:
  - [10,13, 16,30, 33,23]       # P3
  - [30,61, 62,45, 59,119]      # P4
  - [116,90, 156,198, 373,326]  # P5

# Pre-processing
vision_lib: cv2 # OpenCV
norm: [255, 255, 255]           # 0 ~ 255 to 0.0 ~ 1.0 (need to divide by 255.0 on each channel)
mean: [0.0, 0.0, 0.0]

# pre-processing ex.
# img = img.to(device) # cpu or gpu
# img = img.float() # uint8 to fp32
# img = img / 255.0 # normalize 0~255 to 0.0~1.0
# Output
output_format_allow_list: True # mutiple detection per image
output_number: 3               # number of output layers (ex. 3-floor pyramid; P3, P4, P5)
output_size:                   # [batch_size, anchors, pred, height, width]
  [[1, 3, 85, 20, 20],
   [1, 3, 85, 40, 40],
   [1, 3, 85, 80, 80],
  ]
output_pred_format:            # 85 = 4(coordinate) + 1(confidence score) + 80(probablity of classes)
  ['x', 'y', 'w', 'h', 'confidence', 'probability_of_classes']

# Post-processing
conf_thres: 0.25   # for NMS
iou_thres: 0.45    # for NMS
need_nms: True     # need to add non-maximum suppression(NMS) codes
```

----

## YAML format for neural network exeuction info <a name="nn_exec_info"></a>

```yaml
cpu : x86/arm               #default  x86 
acc : cpu/cuda/opencl       #default  cpu 
os : linux/windows          #default  linux 
engine  : pytorch/acl/rknn  #default  pytorch 
libs : [python==3.9, torch>=1.1.0]       # libs to install
file_path : /test/test                   # target directory
run_file : run.py                        # exectuion file entry_point
run_parameters : [ -v, xxx, -path, /tmp] # run-time parameter
targer_location : 129.123.45.223         # target device ip address

#optional
nn_file : .py/.onnx
weight_file : .pt/null
label_info_file : file_name
```
