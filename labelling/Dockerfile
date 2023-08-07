# -------------------------------------------------------------------
# BUILD
# -------------------------------------------------------------------
# build base
FROM node:14.14.0-alpine3.12 AS weda_build

# install apk
RUN apk update &&\
    apk add python2 build-base &&\
    apk add bash


# git install
RUN apk add git

# copy soruce
COPY Client /var/client
COPY Server /var/server
COPY Model /var/model
COPY Bin /var/bin

RUN npm install pnpm -g
RUN ls /var
RUN rm -rf /var/client/node_modules
RUN rm -rf /var/client/package-lock.json
RUN rm -rf /var/server/node_modules
RUN rm -rf /var/server/package-lock.json
WORKDIR /var/server
RUN npm install
RUN npm run build
WORKDIR /var/client
RUN npm cache clean --force
RUN npm install
RUN npm rebuild node-sass
RUN npm run build:dev
RUN mkdir -p /var/release/Server
RUN mkdir -p /var/release/Client
RUN mkdir -p /var/release/Model
WORKDIR /var/release/Server
RUN cp -R /var/server/dist ./
RUN cp -R /var/server/node_modules ./
RUN find ./dist/config ! -name "server-release.json" -type f -delete
RUN mv ./dist/config/server-release.json ./dist/config/server.json
RUN cp -R /var/client/dist/* /var/release/Client
RUN cp -R /var/model/* /var/release/Model



# -------------------------------------------------------------------
# BUILD
# -------------------------------------------------------------------
# base
# FROM node:14.14.0-alpine3.12 AS weda_bluai
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04 AS weda_bluai

# environment
ENV DEBIAN_FRONTEND=noninteractive

# db configuration
ADD Deployment/db/my.cnf /etc/my.cnf
ADD Deployment/install/package_install.sh \
    Deployment/config/default.conf \
    Deployment/config/nginx.conf \
    Deployment/config/requirement.txt \
    /var/app/config/
ADD Deployment/lib/cuda  /var/app/package/cuda

RUN /var/app/config/package_install.sh
# RUN /var/app/config/package_install.sh &&\
#     rm -rf /var/app/config


# ENV ORACLE_HOME=/var/app/package/oracle
ENV PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
# ENV LD_LIBRARY_PATH=${ORACLE_HOME}:${ORACLE_HOME}/lib:${LD_LIBRARY_PATH}:/usr/local/cuda/lib64:/usr/local/cuda/compat:/usr/lib/x86_64-linux-gnu


# copy oracle client
# COPY Deployment/lib/oracle /var/app/package/oracle

# copy SQL Query
COPY Deployment/db/*.sql /docker-entrypoint-initdb.d/

# copy server
COPY --from=weda_build /var/release /var/app/

# copy bin
COPY Deployment/bin /var/app/bin

RUN pip install pysimdjson

WORKDIR /var/app/bin

# copy entrypoint
ENTRYPOINT ["./start.sh"]
