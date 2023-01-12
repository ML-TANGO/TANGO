# TANGO

**Table of Contents**
* [Introduction to TANGO](#intro)
* [Source Tree Structure](#source_tree)
* [How to build images and run containers](#img_build_container_run)
  * [Docker and Docker-compose Installation](#docker_install)
  * [TANGO repository clone](#repo_clone)
  * [TANGO containers image build and run](#tango_run)
* [How to cleanup images and container instances outcomes](#clean_up)
* [How to run individual component containers](#per_container_run)
* [Developer Guides](#dev_guides)
* [Acknowledgement](#ack)

----

> **Announcement**  
> * [2022  Fall TANGO Community Conference](https://github.com/ML-TANGO/TANGO/discussions/31)
> * [2022  Fall TANGO Pre-Release](https://github.com/ML-TANGO/TANGO/releases/tag/tango-22.11-pre1)
> * [2022  Fall TANGO Release]()  - to be announced.
----

## Introduction to TANGO <a name="intro"></a>

TANGO (**T**arget **A**daptive  **N**o-code neural network **G**eneration and **O**peration framework) is code name of project for Integrated Machine Learning Framework.

It aims to develop automatic neural network generation and deployment framework that helps novice users to easily develop neural network applications with less or ideally no code efforts and deploy the neural network application onto the target device.

The users of TANGO just prepare their labelled datasets to train models and target devices. Then, TANGO analyzes the datasets and target devices characteristics, generates task-specific neural network based on user requirements, trains it with the datasets, creates Docker container images and deploys the container images onto target device.

TANGO uses container technology and MSA (Micro Service Architecture). Containers require less system resources than traditional or hardware virtual machine environments because they don't include operating system images. Applications running in containers can be deployed easily to multiple different operating systems and hardware platforms.

Each component of TANGO is self-contained service component implemented with container technology.
The component interacts with other component via REST APIs as depicted in the following image;

<img src="./docs/media/TANGO_structure_v1.png" alt="TANGO Project Overview" width="800px"/>

----

## Source Tree Structure <a name="source_tree"></a>

The source tree is organized with the MSA principles: each subdirectory contains component container source code. Due to the separation of work directory, component container developers just work on their own isolated subdirectory and publish minimal REST API to serve other component containers service request.

```bash
$ tree -d -L 2
.
├── project_manager            # front-end server for TANGO
│   ├── backend
│   ├── data
│   ├── tango
│   ├── frontend
│   └── static
│
├── labelling             # data labelling tool
│   ├── backend
│   └── labelling
│
├── base_model_select     # base model selection
│
├── autonn                # automatic neural network
│   ├── autonn
│   └── backend
│
├── target_image_build    # build neural network image to be deployed
│   ├── backend
│   └── target_image_build
│
├── target_deploy         # generated neural network deployment to target
│   ├── backend
│   └── target_deploy
│
├── visualization         # neural network model visualization
│
└── docs                  # project documentation

```

----


## How to build images and run containers <a name="img_build_container_run"></a>

If you have not installed the docker and docker-compose, please refer to following section.

### Docker and Docker-compose Installation <a name="docker_install"></a>

The descriptions in this sections are based on follow test environments:
* Linux Ubuntu 18.04 and 20.04 LTS

<details>
    <summary>System Prerequisite</summary>

```bash
sudo apt-get update

sudo apt-get install ca-certificates curl gnupg lsb-release

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
$(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```
</details>

<details>
    <summary>How to install docker engine</summary>

```bash
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
```

Check the installed `docker` version.

```bash
docker --version
```
</details>

<details>
    <summary>How to install NVIDIA container toolkit</summary>


TANGO can use GPU resources in some containers such as bms, autonn_nk, autonn_bb, etc.


You would consider installing NVIDIA container toolkit.


* Make sure you have installed the NVIDIA driver and Docker 20.10 for your linux machine.
* You do not need to install the CUDA toolkit on the host, but the driver need to be installed.
* With the release of Docker 19.03, usage of nvidia-docker2 packages is deprecated since NVIDIA GPUs are now natively supported as devices in the Docker runtime.

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -

curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

sudo systemctl restart docker
```
* you can check the latest version info at https://github.com/docker/compose/releases/

```bash
sudo chmod +x /usr/local/bin/docker-compose
```

Check the installed `docker-compose` version.
```bash
docker-compose --version
```
</details>

<details>
    <summary>How to install docker-compose</summary>

```bash
sudo curl -L "https://github.com/docker/compose/releases/download/v2.6.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
```
* you can check the latest version info at https://github.com/docker/compose/releases/

```bash
sudo chmod +x /usr/local/bin/docker-compose
```

Check the installed `docker-compose` version.
```bash
docker-compose --version
```
</details>

### TANGO repository clone <a name="repo_clone"></a>

Once youn installed docker and docker-compose in your local host system, you can clone the GitHub TANGO repository into local host

If you have registered your public key with your github ID, you can use following command
```bash
git clone git@github.com:ML-TANGO/TANGO.git
```

Please refer to  [How to Add SSH Keys to Your GitHub Account](https://www.inmotionhosting.com/support/server/ssh/how-to-add-ssh-keys-to-your-github-account/).


### TANGO containers image build and run<a name="tango_run"></a>

After cloning TANGO repository into your local host, change working directory into local TANGO repository.

```bash
cd TANGO
```

Build docker images and run the containers with `docker-compose` command.

```bash
docker-compose up -d --build
```
> Note 
> * run above command at directory where `docker-compose.yml` file is located.
> * `docker-compose up -d --build` requires a lot of times and sometimes it seems do nothing but it does something. **Be patient!!**

If you're in low bandwith Internet environment or using problematic DNS server, from time to time `docker-compose up -d --build` command would be interrupted by following errors(`Temporary failure in name resolution`):
```
failed to solve: rpc error: code = Unknown desc = failed to solve with frontend dockerfile.v0: 
failed to create LLB definition: 
failed to do request: Head "https://registry-1.docker.io/al tcp: lookup registry-1.docker.io: 
Temporary failure in name resolution
```
If this is your case, you should repeatedly run  `docker-compose up -d --build`  before to get the following message(**FINISHED**):
```
[+] Building 1430.5s (114/114) FINISHED
...
...
Use 'docker scan' to run Snyk tests against images to find vulnerabilities and learn how to fix them
[+] Running 9/10
...
```

Once previouse step completes successfule, following docker container images and containers can be found in your local host system.

**Example list of Docker images**

```bash
$ $ docker image ls
REPOSITORY                 TAG       IMAGE ID       CREATED              SIZE
tango_labelling            latest    08b7e0228997   About a minute ago   11.6GB
tango_viz2code             latest    0ba930ceb8e0   17 minutes ago       7.6GB
tango_autonn_nk            latest    ae9abca17942   32 minutes ago       10.9GB
tango_project_manager      latest    a1f70db5ce71   34 minutes ago       1.15GB
tango_target_deploy        latest    cc61506c133e   34 minutes ago       952MB
tango_target_image_build   latest    4e383c2f8344   34 minutes ago       957MB
postgres                   latest    901a82b310d3   7 days ago           377MB
mariadb                    10        14f1097913ec   2 weeks ago          384MB
```
* Note that the name of the docker images generated based on `docker-compose.yml` is prefixed by its folder name (e.g, `'tango_'`)

**Example list of Docker containers**
```bash
$ docker ps -a --format "table {{.Image}}\t{{.Names}}\t{{.Status}}\t{{.Command}}\t{{.Ports}}"
IMAGE                      NAMES                        STATUS          COMMAND                  PORTS
tango_project_manager      tango-project_manager-1      Up 51 seconds   "sh -c 'chmod 777 ./…"   0.0.0.0:8085->8085/tcp, :::8085->8085/tcp
tango_labelling            tango-labelling-1            Up 51 seconds   "./start.sh"             0.0.0.0:8086->80/tcp, :::8086->80/tcp, 0.0.0.0:8095->10236/tcp, :::8095->10236/tcp
tango_viz2code             tango-viz2code-1             Up 51 seconds   "sh -c 'cd ./visuali…"   0.0.0.0:8091->8091/tcp, :::8091->8091/tcp
postgres:latest            tango-postgresql-1           Up 52 seconds   "docker-entrypoint.s…"   5432/tcp
tango_target_image_build   tango-target_image_build-1   Up 52 seconds   "sh -c 'python manag…"   0.0.0.0:8088->8088/tcp, :::8088->8088/tcp
tango_autonn_nk            tango-autonn_nk-1            Created         "sh -c 'python manag…"
mariadb:10                 mariadb                      Up 51 seconds   "docker-entrypoint.s…"   0.0.0.0:3306->3306/tcp, :::3306->3306/tcp
tango_target_deploy        tango-target_deploy-1        Up 52 seconds   "sh -c 'python manag…"   0.0.0.0:8089->8089/tcp, :::8089->8089/tcp

```
* Note that the name of the docker containers genrated based on `docker-compose.yml` is prefixed by its folder name (e.g, `'tango_'`) and suffixed by the its instance ID (e.g, `'_1'`).

**TANGO in Web-browser**

Now you can launch web browser and open URL `http://localhost:8085` or `http://aaa.bbb.ccc.ddd:8085`.

* `aaa.bbb.ccc.ddd` is your host's DNS address or IP address.
* `8085` is published port from `TANGO_web_1` container, which acts as front-end server of TANGO.

Then you can see the login page of TANGO as follows:

<img src="./docs/media/TANGO_init_screen.png" alt="TANGO Screenshot" width="600px"/>

Once you can find the login page in the web browser, register new account and password and use the newly created account and password to login.


----

## How to cleanup docker images and container instances <a name="clean_up"></a>

When you want remove all the images and containers prviously built and run, you can use following commands;
```bash
# tear down all containers and remove all docker images created and volumes.
$ docker-compose down --rmi all --volumes

#or tear down all containers and remove all docker images created except for volumes.
$ docker-compose down --rmi all 

# remove all images in the local docker host for preventing cached image layers side effect
# when you are to build from the zero base.
docker system prune -a

# remove labelling dataset related folder if you want to start from the empty datasets
$ sudo rm -rf ./labelling/data/
$ sudo rm -rf ./labelling/datadb/
$ sudo rm -rf ./labelling/dataset/
```

> **Note**  
> * Run above command at project root directory (e.g `'TANGO'`) where `docker-compose.yml` file is.
> * Ater running of above commands, your account on project manager as well as datasets prepared with `labelling` tool are removed, due to  `--volumes` option.
> * Hence, you recreate account for project manager and dataset from the scratch.

----

## How to run individual component containers <a name="per_container_run"></a>

Currently we have following component containers;

* **labelling** : dataset labelling authoring tool: 
* **autonn**: automatic neural network creation: 
* **target_image_build**: target deployment image build
* **target_deploy**: image deployment to target: 

For testing or debugging of the individual component container, you want to run container individually.

First of all, check your current working branch is `main`.

```bash
$ git branch -a
* main
  remotes/origin/HEAD -> origin/main
  remotes/origin/main
  remotes/origin/sub
```


<details>
    <summary>labelling: container for labelling tool </summary>

Change current working directory into `labelling` and image build with `Dockerfile`
```bash
cd labelling
docker build -t labelling .
```

`labelling` container run
```bash
docker run -d --name labelling -p 8086:80 labelling:latest
```
</details>


<details>
    <summary> autonn: container for automatic neural network creationn </summary>

:warning:<span style='background-color:#dcffe4'>Currently, autonn consists of two different NAS modules, **neck-nas** and **backbone-nas**.</span>  
You should build both of them at each directory respectively.
***
For **backbone NAS**, change current working directory into `autonn/backbone-nas` and  image build with `Dockerfile`

```bash
cd ../autonn/backbone_nas
docker build -t autonn_bb .
```

`autonn_bb` container run  
Be careful not to get the port number wrong, `8087` is allocated for backbone-nas

```bash
docker run -d --name autonn_bb -p 8087:8087 -v autonn_bb:latest
```

If CUDA is available, you can use `--gpus=all` options

```bash
docker run -d --gpus=all --name autonn_bb -p 8087:8087 autonn_bb:latest
```

When you run into shared memory shortage, you should use `--ipc=host` options

```bash
docker run -d --gpus=all --ipc=host --name autonn_bb -p 8087:8087 autonn_bb:latest
```

***
Similary for **neck NAS**, change current working directory into `autonn/neck-nas` and  image build with `Dockerfile`

```bash
cd ../autonn/neck_nas
docker build -t autonn_nk .
```

`autonn_nk` container run  
Be careful not to get the port number wrong, `8089` is allocated for neck-nas

```bash
docker run -d --name autonn_nk -p 8089:8089 autonn_nk:latest
```

If CUDA is available, you can use `--gpus=all` options

```bash
docker run -d --gpus=all --name autonn_nk -p 8089:8089 autonn_nk:latest
```

When you run into shared memory shortage, you should use `--ipc=host` options

```bash
docker run -d --gpus=all --ipc=host --name autonn_nk -p 8089:8089 autonn_nk:latest
```
</details>


<details>
    <summary>target_image_build: container for target deployment image build</summary>

Change current working directory into `target_image_build` and image build with `Dockerfile`

```bash
cd ../target_image_build/
docker build -t target_image_build .
```

`target_image_build` container run

```bash
docker run -d --name target_image_build -p 8088:8088 target_image_build:latest
```
</details>


<details>
    <summary> target_deploy: container for image deployment to target</summary>

Change current working directory into `target_deploy` and image build with `Dockerfile`

```bash
cd ../target_deploy/
docker build -t target_deploy .
```

`target_deploy` container run
```bash
docker run -d --name target_deploy -p 8089:8089 target_deploy:latest
```
</details>

----

## Developer Guides and References<a name="dev_guides"></a>

* [TANGO Architecture Overview](https://github.com/ML-TANGO/TANGO/wiki/Guides-%7C-TANGO-Architecture)
* [TANGO Container Port Mapping guide](https://github.com/ML-TANGO/TANGO/wiki/Guides-%7C-Container-Port-Map)
* [Exchanging Data among Containers](https://github.com/ML-TANGO/TANGO/wiki/Guides-%7C-Exchanging-Data-among-Containers)
* [TANGO REST API Guide](https://github.com/ML-TANGO/TANGO/wiki/Guides-%7C-Rest-API)

----

## Acknowledgement <a name="ack"></a>

This work was supported by [Institute of Information & communications Technology Planning & Evaluation (IITP)](https://www.iitp.kr/) grant funded by the Korea government(MSIT) (**No. 2021-0-00766**, _Development of Integrated Development Framework that supports Automatic Neural Network Generation and Deployment optimized for Runtime Environment_).
