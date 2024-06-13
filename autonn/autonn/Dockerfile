# --------- Frontend : base-NodeJS, React ----------
FROM node:16-alpine AS builder

WORKDIR /frontend

COPY ./visualization/package.json ./
COPY ./visualization .
RUN npm install
RUN npm run build

# ---------- Backend : base-NNI, Django ------------
FROM python:3.10
FROM msranni/nni

ENV TF_ENABLE_ONEDNN_OPTS=0
ENV PIP_DEFAULT_TIMEOUT=1000

RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx

WORKDIR /source
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . .

COPY --from=builder /frontend /frontend
EXPOSE 8100
