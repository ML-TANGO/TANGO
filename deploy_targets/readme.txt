<deploy_targets>

□ Function of this folder
   o Deploys application code to the proper device (cloud, edge cloud or ondevice)

□ Folder structure
-----------------------------------------------------------------------------------------
|  Folder name        |   Functionality                                                 |
|  (Container name)   |                                                                 |
-----------------------------------------------------------------------------------------
|  cloud              |   Deploy docker image to cloud, edge cloud base                 |
-----------------------------------------------------------------------------------------
|  k8s                |   Deploy docker image to k8s supported targets                  |
-----------------------------------------------------------------------------------------
|  ondevice           |   Deploy neural network codes to PC or on-devices                |
-----------------------------------------------------------------------------------------

□ Operation Flow of this folder
   o Receives a yaml file and program code for deployment
   o If the target device is cloud or edge cloud, then cloud_deploy will be called
   o else ondevice_deploy will be called


□ Supported Targets
------------------------------------------------------------------------------------------------------------------------------
|  Type          |   CPU   |  Accelerator   |   OS             |  Neural Network Engine            |   Chip/Board/System Name |
------------------------------------------------------------------------------------------------------------------------------
|  Cloud         |   x64   |   CUDA         |   Linux          |   PyTorch                         |  Goolge Cloud Platform   |
------------------------------------------------------------------------------------------------------------------------------|
|  K8s           |   x64   |   CUDA         |   Linux          |   PyTorch                         |   PCs                    |
------------------------------------------------------------------------------------------------------------------------------
|  OnDevice      |   ARM   |   Mali GPU     |   Linux          |   ACL/ARMNN/PyARMNN               |  ODROID-N2+              |
|                |         |                |                  |                                   |  (Mali-G52 GPU)          |
|                 ------------------------------------------------------------------------------------------------------------|
|                |   ARM   |   CUDA         |   Linux          |   TensorRT/TVM/PyTorch            |  Jetson Nano             |
|                |         |                |                  |                                   |                          |
|                 ------------------------------------------------------------------------------------------------------------|
|                |   ARM   |   RKNN NPU     |   Linux          |   RKNN                            |  ODROID-M1               |
|                |         |                |                  |                                   |  Firefly-RK3399PRO       |
|                |         |                |                  |                                   |  Firefly-RK3399          |
------------------------------------------------------------------------------------------------------------------------------
* Anroid Smartphone will be added in the next release version.

