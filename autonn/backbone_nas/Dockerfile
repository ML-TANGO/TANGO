# FROM node:14.14.0-alpine3.12
# # install apk
# RUN apk update &&\
#     apk add python2 build-base &&\
#     apk add bash

# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 as base

# # environment
# ENV DEBIAN_FRONTEND=noninteractive
# RUN echo 'export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}' >> ~/.bashrc
# RUN echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
# RUN echo 'export PATH=/usr/local/cuda/bin:/$PATH' >> ~/.bashrc
# RUN echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc


# link cuda
# RUN VER=`nvidia-smi | grep NVIDIA | awk '{print$3}'`
# RUN ln -s lib/cuda/libcuda.so.${VER} /usr/local/cuda-11.8/lib64/libcuda.so.1
# ENV PATH=/usr/local/cuda/bin${PATH:+:${PATH}}


# RUN apt-get update \
#     && apt-get install -y sudo
# RUN adduser --disabled-password --gecos '' docker
# RUN adduser docker sudo
# RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
# USER docker

FROM python:3.10

# install nodejs, db
RUN apt-get update
# RUN apt-get install curl -y
# RUN curl -sL https://deb.nodesource.com/setup_14.x | bash -
# RUN apt-get install nodejs -y
RUN apt-get install -y postgresql postgresql-contrib
RUN apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx libglib2.0-0

# install NVIDA Driver
# RUN ubuntu-drivers autoinstall
# RUN curl -fSsl -O https://us.download.nvidia.com/tesla/450.80.02/NVIDIA-Linux-x86_64-450.80.02.run | bash -
# RUN chmod +x NVIDIA-Linux-x86_64-450.80.02.run
# RUN ./NVIDIA-Linux-x86_64-450.80.02.run
# RUN apt-get install linux-headers-$(uname -r)
# RUN chmod +x NVIDIA-Linux-x86_64-440.44.run 
# RUN ./NVIDIA-Linux-x86_64-440.44.run

# install python
# RUN apt-get install python3.8-dev -y
# RUN apt-get install python3-pip -y
# RUN python -m pip install --upgrade pip


WORKDIR /source
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
COPY . .

# WORKDIR /envset
# ADD install/lib/cuda  /source/app/package/cuda
# RUN ["chmod", "+x", "/envset/env_install.sh"]
# RUN ./envset/env_install.sh





# ENV PATH=/usr/bin/python
# ENV PATH=/usr/bin/python/Scripts

# COPY requirements.txt ./
# RUN pip install --upgrade pip
# RUN pip install -r requirements.txt
# COPY . .

# RUN apt-get install -y nodejs
# RUN apt-get -y install libgl1-mesa-glx
# RUN apt-get install -y postgresql postgresql-contrib
# RUN npm
# RUN npm install pnpm -g
# RUN npm install
# WORKDIR /souce/frontend
# RUN npm run build

# WORKDIR /source

CMD ["bash"]
EXPOSE 8087

