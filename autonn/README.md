# AutoNN

## Real-Time Object Detection NN
<div align="center">
  <p>
      <img width="100%" src=./../docs/media/object_detection_network.png></a>
  </p>
</div>

## Image Classification NN
<div align="center">
  <p>
      <img width="100%" src=./../docs/media/resnet18.png></a>
  </p>
</div>

## Directory Tree

```bash
autonn
├── autonn
│       ├── autonn_core
│       │       ├─ __init__.py
│       │       ├─ admin.py
│       │       ├─ apps.py
│       │       ├─ models.py
│       │       ├─ serializers.py
│       │       ├─ urls.py
│       │       ├─ views.py
│       │       ├─ migrations
│       │       │   └─ __init__.py
│       │       └─ tango
│       │           ├─ common
│       │           ├─ main
│       │           ├─ nas
│       │           ├─ hpo
│       │           ├─ utils
│       │           └─ viz
│       ├── config
│       │       ├─ __init__.py
│       │       ├─ settings.py
│       │       ├─ urls.py
│       │       ├─ asgi.py
│       │       └─ wsgi.py
│       └── visualization
│               ├─ public
│               ├─ src
│               └─ package.json
├── backbone_nas
│       ├── backend
│       └── bnas
├── neck_nas
│       ├── backend
│       ├── frontend
│       ├── neckNAS
│       ├── sample_data
│       └── sample_yaml
├── ResNet
│       ├── backend
│       ├── resnet_core
│       └── sample_yaml
└── YoloE
        ├── backend
        ├── sample_data
        ├── sample_yaml
        └── yoloe_core
```

## Docker Containers
|directory|docker name|port|note|
|--:|--|--|:--:|
|autonn|tango_autonn|8100|${\textsf{\color{green}currently in development}}$|
|backbone_nas|tango_autonn_bb|8087|${\textsf{\color{magenta}not active}}$|
|neck_nas|tango_autonn_nk|8089|${\textsf{\color{magenta}not active}}$|
|YoloE|tango_autonn_yoloe|8090|${\textsf{\color{blue}active}}$|
|ResNet|tango_autonn_resnet|8092|${\textsf{\color{blue}active}}$|

