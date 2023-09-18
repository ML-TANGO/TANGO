FROM python:3.6
WORKDIR /app
COPY . /app
RUN apt-get update
RUN apt-get install -y git
RUN git clone  https://github.com/parkjb12/lib.git /app/lib
RUN cp /app/lib/* /app
RUN cp /app/libprotobuf.so.10 /lib/x86_64-linux-gnu
COPY db/. /app/db
COPY db/yolov3.body /app/db
COPY db/yolov3.head /app/db
COPY db/yolo_v3_tiny_darknet_fp32.tflite /app/db
COPY db/odroid-m1-manual.txt /app/db
COPY db/odroid-n2-manual.txt /app/db
COPY db/yolov3/. /app/db/yolov3
ADD  https://github.com/giantlkh/git-test/releases/download/yolov7.onnx/yolov7.onnx  /app/db
ADD  https://github.com/giantlkh/git-test/releases/download/k8syolov5/k8syolov5.db  /app/db
ADD  https://s3.ap-northeast-2.amazonaws.com/backend.ai/offline-installers/yolov3.pt /app/db/yolov3
# RUN apt-get install -y python3 python3-dev  gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
# RUN apt-get install -y gcc-11 
RUN apt-get install -y libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
# RUN apt-get remove -y python3-pip
# RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install pyyaml requests
RUN pip3 install tvm
RUN pip3 install rknn_toolkit2-1.2.0_f7bb160f-2-cp36-cp36m-linux_x86_64.whl
RUN pip3 install torch
RUN pip3 install tensorflow
# RUN pip3 install onnx<=1.12.0
RUN pip3 install onnx
RUN pip3 install seaborn pandas torchvision
RUN pip3 install --upgrade attrs psutil
RUN apt-get install -y libgl1-mesa-glx 
RUN apt-get install -y vim sed 
# RUN apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
WORKDIR /app
RUN git clone --recursive https://github.com/apache/tvm tvm
ADD https://github.com/llvm/llvm-project/releases/download/llvmorg-11.0.0/clang+llvm-11.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz /app
RUN tar xvf clang+llvm-11.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz
RUN mv clang+llvm-11.0.0-x86_64-linux-gnu-ubuntu-20.04 llvm11
WORKDIR /app/tvm
RUN mkdir build
# RUN cp cmake/config.cmake build
RUN sed 's/set(USE_LLVM\ OFF)/set\(USE_LLVM\ \"\/app\/llvm11\/bin\/llvm-config\ --link-static\"\)\nset(HIDE_PRIVATE_SYMBOLS\ ON)'/g cmake/config.cmake > build/config.cmake
WORKDIR /app/tvm/build
ENV TVM_LIBRARY_PATH=/app/tvm/build \
PYTHONPATH=/app/tvm/python:${PYTHONPATH} 
# PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
# fot test RUN cmake ..
# for test RUN make -j4

# for android

WORKDIR /app
CMD ["/bin/bash"]
