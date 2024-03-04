FROM python:3.6

ENV CMAKE_ARGS="-DONNX_USE_PROTOBUF_SHARED_LIBS=ON"

RUN apt-get update
RUN pip3 install --upgrade pip
# RUN apt-get install -y build-essential cmake git libprotoc-dev protobuf-compiler 
# RUN apt-get install -y build-essential git libprotoc-dev protobuf-compiler 
RUN git init
RUN git update-ref -d HEAD
RUN pip3 install "pybind11[golbal]"
RUN pip3 install onnx
RUN pip3 install onnx-simplifier
RUN pip3 install tvm
RUN pip3 install decorator
RUN apt-get install -y libtinfo-dev libncurses5

WORKDIR /app
COPY . /app
COPY db/. /app/db
COPY db/yolov3.body /app/db
COPY db/yolov3.head /app/db
COPY db/yolo_v3_tiny_darknet_fp32.tflite /app/db
COPY db/odroid-m1-manual.txt /app/db
COPY db/odroid-n2-manual.txt /app/db
COPY db/yolov3/. /app/db/yolov3
ADD  https://s3.ap-northeast-2.amazonaws.com/backend.ai/offline-installers/yolov3.pt /app/db/yolov3
RUN apt-get install -y zlib1g-dev libedit-dev libxml2-dev
RUN pip3 install pyyaml requests
RUN pip3 install tvm
RUN pip3 install torch==1.7.0
RUN pip3 install tensorflow==2.4.0
RUN pip3 install tensorflow-datasets
RUN pip3 install seaborn pandas torchvision
RUN pip3 install --upgrade attrs psutil
RUN apt-get install -y libgl1-mesa-glx 
RUN apt-get install -y vim sed 
RUN pip3 install cmake==3.22.1
WORKDIR /app
RUN git init
RUN git clone --recursive https://github.com/apache/tvm tvm
ADD https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.0/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz  /app
RUN tar xvf clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
RUN mv clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04 llvm10
WORKDIR /app/tvm
RUN mkdir build
RUN sed 's/set(USE_LLVM\ OFF)/set\(USE_LLVM\ \"\/app\/llvm10\/bin\/llvm-config\ --link-static\"\)\nset(HIDE_PRIVATE_SYMBOLS\ ON)'/g cmake/config.cmake > build/config.cmake
WORKDIR /app/tvm/build
ENV TVM_LIBRARY_PATH=/app/tvm/build \
PYTHONPATH=/app/tvm/python:${PYTHONPATH} 
ENV PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN cmake ..
RUN make -j4

# for android
WORKDIR /app
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y  \
    wget \
    cpio \
    unzip \
    sudo \
    lsb-release

# install openvino
ENV DOWNLOAD_LINK=https://registrationcenter-download.intel.com/akdlm/irc_nas/17662/l_openvino_toolkit_p_2021.3.394.tgz
ENV INSTALL_DIR=/opt/intel/computer_vision_sdk
ENV TEMP_DIR=/tmp/openvino_installer
RUN mkdir -p $TEMP_DIR && cd $TEMP_DIR && \
    wget  https://registrationcenter-download.intel.com/akdlm/irc_nas/17662/l_openvino_toolkit_p_2021.3.394.tgz && \
    tar xf l_openvino_toolkit*.tgz && \
    cd l_openvino_toolkit* && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh -s silent.cfg && \
    rm -rf $TEMP_DIR
RUN pip3 install openvino2tensorflow==1.17.2

# android sdk
RUN echo 'y\n\
y\n\
y\n\
y\n\
y\n\
y\n\
y\n\
y\n\
y\n\
y\n\
y\n\ 
'> /app/yes.txt

RUN echo '\n\
\n\
\n\
\n\
\n\
\n\
\n\
\n\
\n\
\n\
\n\
' > /app/enter.txt

#install android sdk
ENV FORMAT=float32
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64/
ENV ANDROID_SDK_ROOT=/app/android_sdk
ENV ANDROID_HOME=/app/android_sdk
ENV GRADLE_HOME=/app/android_sdk/gradle-8.3

RUN pip3 install openvino

ENV PATH=${GRADLE_HOME}/bin:${PATH}
ENV PYTHONPATH=/opt/intel/openvino_2021/python/python3.6/:$PYTHONPATH

RUN pip3 install networkx
RUN pip3 install opencv-python
RUN pip3 install defusedxml
RUN pip3 install matplotlib

RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:openjdk-r/ppa
RUN apt-get install -y openjdk-17-jdk
WORKDIR /app
RUN mkdir android_sdk
WORKDIR /app/android_sdk
RUN mkdir cmdline-tools
WORKDIR /app/android_sdk/cmdline-tools
RUN wget https://dl.google.com/android/repository/commandlinetools-linux-10406996_latest.zip
RUN unzip commandlinetools-linux-10406996_latest.zip
RUN mv cmdline-tools latest

RUN /app/android_sdk/cmdline-tools/latest/bin/sdkmanager --licenses < /app/yes.txt
RUN wget https://services.gradle.org/distributions/gradle-8.3-bin.zip
RUN unzip gradle-8.3-bin.zip
WORKDIR ${GRADLE_HOME}/bin
RUN wget https://github.com/ninja-build/ninja/releases/download/v1.11.1/ninja-linux.zip
RUN unzip ninja-linux.zip


# down load android project
WORKDIR /app
RUN git clone https://github.com/lhy0718/tflite_yolov7_test

WORKDIR /app/db
RUN wget https://github.com/parkjb12/yoloe/raw/main/yolov7.zip
RUN mv yolov7.zip k8syolov7.zip
RUN unzip -o k8syolov7.zip
RUN mv yolov7/yoloe_core .
RUN /bin/rm -r yolov7

RUN git clone https://github.com/WongKinYiu/yolov7.git
RUN mv yoloe_core yolov7

WORKDIR /app
CMD ["/bin/bash"]
