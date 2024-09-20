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
# FROM nvcr.io/nvidia/pytorch:21.11-py3

ENV TF_ENABLE_ONEDNN_OPTS=0
ENV PIP_DEFAULT_TIMEOUT=1000

RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx
# RUN apt-get install -y nodejs npm

WORKDIR /source
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

COPY --from=builder /frontend /frontend
EXPOSE 8100
