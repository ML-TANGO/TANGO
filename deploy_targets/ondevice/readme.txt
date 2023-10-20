<ondevice_deploy>

□ Functionality of this folder
   o Delivers application codes for ondevice to users

□ Operation Flow of this container
   o Receives a yaml file and program code for inference
   o Makes a zip file that contains program code, converted neural network model, etc.
   o Makes Project Manager send or copy the zip file to users


□ Supported Targets
------------------------------------------------------------------------------------------------------------------------------
|  Type          |   CPU   |  Accelerator   |   OS             |  Neural Network Engine            |  Chip/Board/System Name  |
------------------------------------------------------------------------------------------------------------------------------
|  OnDevice      |   ARM   |   Mali GPU     |   Linux          |   ACL/ARMNN/PyARMNN               |  ODROID-N2+              | 
|                |         |                |                  |                                   |  (Mali-G52 GPU)          |
|                 ------------------------------------------------------------------------------------------------------------|
|                |   ARM   |   CUDA         |   Linux          |   TensorRT/TVM/PyTorch            |  Jetson AGX Orin         | 
|                |         |                |                  |                                   |                          | 
|                 ------------------------------------------------------------------------------------------------------------|
|                |   ARM   |   NPU          |   Linux          |   TensorflowLite                  |  Galaxy S22              | 
|                 -------------------------------------------------------------------------------------------------------------
|                |   x64   |   CUDA         |   Linux          |   PyTorch                         |   PCs                    |
------------------------------------------------------------------------------------------------------------------------------


