README
---

# AutoNN 
## Directory Tree

```bash
autonn
├── backbone_nas
│   ├── backend
│   └── bnas
├── neck_nas
│   ├── backend
│   ├── frontend
│   ├── neckNAS
│   ├── sample_data
│   └── sample_yaml
├── ResNet
│   ├── backend
│   ├── resnet_core
│   └── sample_yaml
└── YoloE
    ├── backend
    ├── sample_data
    ├── sample_yaml
    └── yoloe_core
```

## Docker Containers
|directory|docker name|port|note|
|--:|--|--|:--:|
|backbone_nas|tango_autonn_bb|8087|${\textsf{\color{magenta}not active}}$|
|neck_nas|tango_autonn_nk|8089|${\textsf{\color{magenta}not active}}$|
|YoloE|tango_autonn_yoloe|8090|${\textsf{\color{blue}active}}$|
|ResNet|tango_autonn_resnet|8092|${\textsf{\color{blue}active}}$|
