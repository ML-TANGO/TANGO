# FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04
FROM python:3.10

RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y postgresql postgresql-contrib

WORKDIR /souce
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
# RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
COPY . .

CMD ["python3", "./manage.py", "runserver", "0.0.0.0:8087"]
EXPOSE 8087
