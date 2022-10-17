# FROM node:14.14.0-alpine3.12
# # install apk
# RUN apk update &&\
#     apk add python2 build-base &&\
#     apk add bash

FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

# environment
ENV DEBIAN_FRONTEND=noninteractive
RUN echo 'export PATH=/usr/local/cuda-11.3/bin${PATH:+:${PATH}}' >> ~/.bashrc
RUN echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
RUN echo 'export PATH=/usr/local/cuda/bin:/$PATH' >> ~/.bashrc
RUN echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc


# link cuda
# RUN VER=`nvidia-smi | grep NVIDIA | awk '{print$3}'`
# RUN ln -s lib/cuda/libcuda.so.${VER} /usr/local/cuda/lib64/libcuda.so.1
# ENV PATH=/usr/local/cuda/bin${PATH:+:${PATH}}

# install nodejs, db
RUN apt-get update
RUN apt-get install curl -y
RUN curl -sL https://deb.nodesource.com/setup_14.x | bash -
RUN apt-get install nodejs -y
RUN apt-get install -y postgresql postgresql-contrib

# install NVIDA Driver
RUN curl -fSsl -O https://us.download.nvidia.com/tesla/450.80.02/NVIDIA-Linux-x86_64-450.80.02.run
RUN chmod +x NVIDIA-Linux-x86_64-450.80.02.run
RUN ./NVIDIA-Linux-x86_64-450.80.02.run
# RUN apt-get install linux-headers-$(uname -r)
# RUN chmod +x NVIDIA-Linux-x86_64-440.44.run 
# RUN ./NVIDIA-Linux-x86_64-440.44.run

# install python
RUN apt-get install python3.8-dev -y
#RUN rm -f ./usr/bin/python
#RUN ln -s ./usr/bin/python3.8 /usr/bin/python
RUN apt-get install python3-pip -y
RUN python3 -m pip install --upgrade pip
RUN apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx libglib2.0-0

WORKDIR /source
COPY requirements.txt ./
RUN pip install -r ./requirements.txt
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
# FROM python:3.10
# WORKDIR /source
# EXPOSE 8087
# CMD ["python3", "manage.py", "runserver", "0.0.0.0:8087"]

