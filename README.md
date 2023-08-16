# TANGO

> **Announcement**  
> * [2022  Fall TANGO Community Conference](https://github.com/ML-TANGO/TANGO/discussions/31)
> * [2022  Fall TANGO Pre-Release](https://github.com/ML-TANGO/TANGO/releases/tag/tango-22.11-pre1)
> * [2022  Fall TANGO Release](https://github.com/ML-TANGO/TANGO/releases/tag/tango-22.11)
> * [2023  June TANGO Release](https://github.com/ML-TANGO/TANGO/releases/tag/tango-23.06) 
----

## Introduction to TANGO <a name="intro"></a>

TANGO (**T**arget **A**ware  **N**o-code neural network **G**eneration and **O**peration framework) is code name of project for Integrated Machine Learning Framework.

It aims to develop automatic neural network generation and deployment framework that helps novice users to easily develop neural network applications with less or ideally no code efforts and deploy the neural network application onto the target device.

The users of TANGO just prepare their labelled datasets to train models and target devices. Then, TANGO analyzes the datasets and target devices characteristics, generates task-specific neural network based on user requirements, trains it with the datasets, creates Docker container images and deploys the container images onto target device.

TANGO uses container technology and MSA (Micro Service Architecture). Containers require less system resources than traditional or hardware virtual machine environments because they don't include operating system images. Applications running in containers can be deployed easily to multiple different operating systems and hardware platforms.

Each component of TANGO is self-contained service component implemented with container technology.
The component interacts with other component via REST APIs as depicted in the following image;

<img src="./docs/media/TANGO_structure_v2.png" alt="TANGO Project Overview" width="800px"/>

----

## Main Feature of TANGO

The TANGO framework aims to deploy and load ready-to-use deep learning models for the specific vision task (classification, object detection, or instance segmentation) onto the user's target devices by automatically constructing and training deep learning models without the help of experts or with minimal knowledge on usage on TANGO. To this end, data preparation, neural network model creation, and optimization for target device can be accomplished within TANGO framework.

<img src="./docs/media/TANGO_AutoML.png" alt="TANGO Auto ML" width="600px"/>

----

### Data Preparation

Data preparation consists of two main processes. First, you need to take or collect images (raw data) in various situations suitable for the given vision task. Then, collected row data should be annotated to be used for training of depp learning models. The latter process is very labor intensive and takes a lot of manpower and time. TANGO's labeling tool is a Web-based GUI tool that enables the users easily perform annotation on raw data. The user can load local raw data by just drag and drop style, perform class labeling, bounding box annotation, polygon annotation, etc. according to the specific task, and then save them.

<img src="./docs/media/TANGO_labeling.png" alt="TANGO labeling" width="600px"/>

### Basic Model Selection

Before applying automation techniques such as NAS and HPO, reducing the search space or limiting it to a specific neural network variation is very important in terms of the efficiency of automatic neural network generation. In the TANGO framework, it was named Base Model Selection and was assigned a role of recommending an appropriate neural network among existing well-known neural networks, so called SOTA model.

Model selection is typically done through an algorithm search process. The system may evaluate a set of candidate SOTA models using metrics such as accuracy, precision, recall, or F1 score, and select the model that performs best on a validation dataset. 

### Automatic Neural Network Generation

AutoML (Automated Machine Learning) refers to the process of automating the end-to-end machine learning process, including tasks such as data preprocessing, model selection, hyperparameter tuning, and model deployment. The goal of AutoML is to make it easier for non-experts to build and deploy machine learning models, by removing the need for extensive domain knowledge or manual intervention.

Neural network model generation is a key process in the TANGO framework. The TANGO framework provides guidelines for this kind of task automatically. First, a base neural network is extracted through a base model selector recommended by one of the existing SOTA, State of the Art, neural networks proved to work well. Afterwards, AutoNN uses automation techniques such as NAS ([Neural Architecture Search](https://en.wikipedia.org/wiki/Neural_architecture_search)) and HPO ([Hyper Parameter Optimization](https://en.wikipedia.org/wiki/Hyperparameter_optimization)) to find an appropriate neural network configuration. This includes retraining of the final model.

<img src="./docs/media/TANGO_BMS_AutoNN.png" alt="TANGO labeling" width="500px"/>


#### NAS: Neural Architecture Search

Neural Architecture Search (NAS) is a method for automatically searching for the best neural network architecture for a given task, rather than requiring the practitioner to manually design and fine-tune the network. The process of NAS involves searching the space of possible network architectures to find the one that performs best on a validation dataset. 

NAS has the potential to significantly reduce the time and effort required to design neural networks, especially for complex tasks where the optimal architecture is not clear. It also enables the discovery of novel network architectures that outperform hand-designed ones, leading to improved performance on a wide range of tasks. However, NAS can be computationally expensive and require large amounts of computing resources, making it challenging to apply in practice.

Finding the optimal neural network has long been the domain of AI experts. As the basic operations, layers, and blocks used in neural networks become matured, finding a better neural network architecture through combining them can be iterative work. Therefore, rather than finding a new operation, layer, or block, NAS technology uses the computer with the task of finding a better neural network structure from the possible combination of developed operations, layers, and blocks or changing their parameters. NAS in TANGO can be seen as a process of deriving an optimal user-customized neural network through NAS based on the base neural network recommended by BMS.


#### HPO: Hyper-Parameter Optimization

Even though neural network training is largely automated, there are still many variables that should be setup in advance or tunned during training. This kind of variables is called hyper-parameters. HPO stands for Hyperparameter Optimization, which is the process of tuning the hyper-parameters of a machine learning model to improve its performance on a validation dataset. Hyper-parameters are parameters that are set before training a model and control aspects of the training process such as the learning rate, the number of trees in a random forest, or the number of hidden units in a neural network. Therefore, HPO in TANGO can be seen as an iterative process of determining the optimal values of hyper-parameters to achieve the best performance for the optimal neural network derived from NAS.


### Deployment of Inference Engine on Target Device

During TANGO project configuration, users can specify theirs target device, which is used for inference with pre-trained neural network model. Due to different acceleration engines and available resources for each device, it may be difficult to immediately deploy/load the neural network model output from AutoNN. Therefore, in order to realize full-cycle AutoML, a tool is necessary to  help deploy/install on the target device and launch inference task. Depending on the environment of the target device, TANGO provides a method that can be distributed over the network through direct containerized image build, and a method that makes the executable code including essential libraries and pre/post-processing code into a compressed file and installs it on the the target device and unpacks it.

<img src="./docs/media/TANGO_AutoNN_Deployment.png" alt="Deployment in TANGO" width="500px"/>

### No Code

The TANGO framework aims to help users without specialized knowledge to create and use their own neural network models. To this end, it provides an environment that users can use without writing code, such as a project manager and a neural network visualization tool.

<img src="./docs/media/TANGO_No_Code.png" alt="ETRI SuperNeck" width="800px"/>

You can watch 2022 early version demo on Youtube. Click the below images.

<a href="https://youtu.be/T80YKRyIR3g">
<img src="./docs/media/TANGO_2022_Demo.png" alt="TANGO 2022 Demon on Youtube" width="300px"/>
</a>


----
## Source Tree Structure <a name="source_tree"></a>

The source tree is organized with the MSA principles: each subdirectory contains component container source code. Due to the separation of source directory, component container developers just only work on their own isolated subdirectory and publish minimal REST API to serve project manager container's service request.

```bash
TANGO
  ├─── docker-compose.yml
  ├─── project_manager
  │ 
  ├─── model_select
  │      ├─── device_based
  │      └─── data_feature_based
  │ 
  ├─── visualize
  │      └─── vis2code
  │ 
  ├─── auto_nn
  │      ├─── backbone_nas
  │      ├─── neck_nas
  │      └─── yolov7e
  │
  ├─── deploy_codegen 
  │      └─── optimize_codegen
  │
  └─── deploy_targets
         ├─── cloud
         ├─── k8s
         └─── ondevice

```

----


## How to build images and run containers <a name="img_build_container_run"></a>

refer to TANGO wiki [HowTo | TANGO Image Build and Run](https://github.com/ML-TANGO/TANGO/wiki/HowTo-%7C-TANGO-Image-Build-and-Run)


----

## Developer Guides and References<a name="dev_guides"></a>

* [TANGO Architecture Overview](https://github.com/ML-TANGO/TANGO/wiki/Guides-%7C-TANGO-Architecture)
* [TANGO Container Port Mapping guide](https://github.com/ML-TANGO/TANGO/wiki/Guides-%7C-Container-Port-Map)
* [Exchanging Data among Containers](https://github.com/ML-TANGO/TANGO/wiki/Guides-%7C-Exchanging-Data-among-Containers)
* [TANGO REST API Guide](https://github.com/ML-TANGO/TANGO/wiki/Guides-%7C-Rest-API)

----

## TANGO on media

YouTube Videos
* [ 성공하는 SW기업을 위한 AI, SW개발도구 'TANGO'](https://youtu.be/IwyHOl3WjWQ)
  * 2022 SW공학 TECHNICAL 세미나, 떠오르는 기업의 품질 확보 활동 
  * 2022년 11월 23일(수) 15:30 ~ 17:30 
  * 비앤디파트너스 삼성역점 M3 회의실

* [TANGO, 노코드 신경망 자동생성 통합개발 프레임워크의 품질 관리](https://youtu.be/jrJCXAPKJn8)
  * 2022 SW QUALITY INSIGHT CONFERENCE
  * 2022년 12월 7일(수) 15:30~17:30 
  * COEX 그랜드 컨퍼런스룸 402호


----

## Acknowledgement <a name="ack"></a>

This work was supported by [Institute of Information & communications Technology Planning & Evaluation (IITP)](https://www.iitp.kr/) grant funded by the Korea government(MSIT) (**No. 2021-0-00766**, _Development of Integrated Development Framework that supports Automatic Neural Network Generation and Deployment optimized for Runtime Environment_).
