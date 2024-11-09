# Dockerfile for ResNet(autonn_resnet)
FROM python:3.10
FROM msranni/nni

ENV TF_ENABLE_ONEDNN_OPTS=0
ENV PIP_DEFAULT_TIMEOUT=1000

RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx

# GitHub 미러로부터 사전학습 모델 다운로드
# - https://github.com/ML-TANGO/TANGO/releases/tag/Model_Mirror
RUN mkdir /pretrained
WORKDIR /pretrained
RUN wget https://github.com/ML-TANGO/TANGO/releases/download/Model_Mirror/kagglecxr_resnet152_normalize.pt --retry-connrefused

WORKDIR /source
COPY requirements.txt ./
RUN pip install --force-reinstall pip==24.0.0
RUN pip install -r requirements.txt
COPY . .

EXPOSE 8092
