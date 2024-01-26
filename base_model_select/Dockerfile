FROM python:3.10

ENV TF_ENABLE_ONEDNN_OPTS=0
ENV PIP_DEFAULT_TIMEOUT=1000

RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx

WORKDIR /source
COPY requirements.txt ./
RUN pip install -U pip
RUN pip install -r requirements.txt
COPY . ./

# docker build -t yolo_docker . --progress=plain
# docker run --rm --ipc=host yolo_docker 


EXPOSE 8081