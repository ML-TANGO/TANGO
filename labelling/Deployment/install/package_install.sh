#!/bin/bash

chmod 644 /etc/my.cnf
#chmod 644 /docker-entrypoint-initdb.d/*.sql

# link cuda
VER=`nvidia-smi | grep NVIDIA | awk '{print$3}'`
ln -s /var/app/package/cuda/libcuda.so.${VER} /usr/local/cuda/lib64/libcuda.so.1

# install database
apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
apt-get update
apt-get install tzdata
apt-get -o Dpkg::Options::="--force-confnew" install -y mariadb-server
apt-get install -y gosu
rm -rf /var/lib/mysql/mysql
rm -f /var/lib/mysql/ibdata1
rm -f /var/lib/mysql/ib_logfile101
rm -f /var/lib/mysql/ib_logfile1
rm -f /var/lib/mysql/ib_logfile0
mkdir /var/run/mysqld
chown mysql:root /var/run/mysqld

# install nodejs
apt-get install curl -y
curl -sL https://deb.nodesource.com/setup_14.x | bash -
apt-get install nodejs -y

# install wget
apt-get install -y wget

# install nginx
apt-get install nginx -y
cp /var/app/config/default.conf /etc/nginx/conf.d/default.conf
cp /var/app/config/nginx.conf /etc/nginx/nginx.conf

# install pg
apt-get install libpq-dev -y


# install python
apt-get install python3.8-dev -y
rm -f /usr/bin/python
ln -s /usr/bin/python3.8 /usr/bin/python
apt-get install python3-pip -y
python -m pip install --upgrade pip
apt-get install git -y
apt-get install -y musl-dev
ln -s /usr/lib/x86_64-linux-musl/libc.so /lib/libc.musl-x86_64.so.1
apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx libglib2.0-0

cat /var/app/config/requirement.txt | while read line
do
  pip install $line
done
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
pip install MarkupSafe==2.0.1
