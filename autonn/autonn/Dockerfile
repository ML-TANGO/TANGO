# --------- Frontend : base-NodeJS, React ----------
FROM node:20-alpine AS builder

WORKDIR /frontend

COPY ./visualization/package.json ./
COPY ./visualization .
RUN npm install pnpm -g
RUN npm install --force
RUN npm run build

# ---------- Backend : base-NNI, Django ------------
# FROM python:3.8
FROM msranni/nni
# FROM nvcr.io/nvidia/pytorch:23.05-py3

# ENV TF_ENABLE_ONEDNN_OPTS=0
# ENV TF_CPP_MIN_LOG_LEVEL=3
ENV PIP_DEFAULT_TIMEOUT=1000
# ENV CUDA_LAUNCH_BLOCKING=1

# RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
# RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get -y install libgl1-mesa-glx
# RUN apt-get install libusb-1.0-0
# RUN apt-get install edgetpu-compiler
# RUN curl -fsSL https://ollama.com/install.sh | sh


WORKDIR /source
COPY requirements.txt ./
RUN pip install --upgrade pip
# RUN pip install -r requirements.txt

RUN pip install asgiref==3.5.0
RUN pip install blobfile==3.0.0
RUN pip install certifi==2021.10.8
RUN pip install charset-normalizer==2.0.12
RUN pip install Django==3.2.25
RUN pip install django-cors-headers==3.11.0
RUN pip install djangorestframework==3.15.0
RUN pip install djangorestframework-jwt==1.11.0
RUN pip install easydict==1.10
RUN pip install idna==3.3
RUN pip install matplotlib>=3.5.1
RUN pip install opencv-python==4.7.0.72
RUN pip install pandas==1.4.2
RUN pip install Pillow>=8.4.0
RUN pip install protobuf==3.20.3
RUN pip install psycopg2-binary==2.9.3
RUN pip install pycocotools
RUN pip install pytorch-lightning==0.8.3
RUN pip install pytz==2022.1
RUN pip install PyYAML==6.0
RUN pip install requests==2.28.0
RUN pip install seaborn==0.11.2
RUN pip install setuptools==59.5.0
RUN pip install sqlparse==0.4.2
RUN pip install tensorboardX==2.6
RUN pip install thop>=0.1.1
RUN pip install torch==1.13.1
RUN pip install torchvision==0.14.1
RUN pip install tqdm==4.65.0
RUN pip install typing_extensions
RUN pip install urllib3==1.26.7

# llm --------------------------------------------------------------------------
# RUN pip install beautifulsoup4
# RUN pip install chromadb==0.3.29
# RUN pip install git-lfs==1.6
# RUN pip install huggingface_hub
# RUN pip install langchain==0.2.17
# RUN pip install langchain-community==0.2.19
# RUN pip install ollama
# RUN pip install streamlit>=1.31.0

# model export -----------------------------------------------------------------
# RUN pip install distributed
# RUN pip install flatbuffers==24.3.25
# RUN pip install keras
# RUN pip install nvidia-tensorrt==8.4.1.5
RUN pip install onnx==1.17.0
# RUN pip install onnx_graphsurgeon>=0.3.26
# RUN pip install onnx2tf==1.22.3
# RUN pip install onnxruntime==1.14.1
RUN pip install onnxsim==0.4.33
RUN pip install onnxslim==0.1.39
# RUN pip install sng4onnx>=1.0.1
# RUN pip install tensorflow==2.16.1
# RUN pip install tf_keras==2.16.0
# RUN pip install tflite_support

RUN pip install albumentations==1.0.3 --no-binary qudida,albumentations
RUN pip install numpy==1.23.5

COPY . .

COPY --from=builder /frontend /frontend
EXPOSE 8100
