#FROM python:3.8
FROM python:3.9

ENV CMAKE_ARGS="-DONNX_USE_PROTOBUF_SHARED_LIBS=ON"

RUN apt-get update
RUN pip3 install --upgrade pip
# RUN apt-get install -y build-essential cmake git libprotoc-dev protobuf-compiler 
# RUN apt-get install -y build-essential git libprotoc-dev protobuf-compiler 
RUN git init
RUN git update-ref -d HEAD
RUN pip3 install "pybind11[golbal]"
RUN pip3 install tensorflow tf_keras  tensorflow-datasets
RUN pip3 install onnx2tf==1.20
RUN pip3 install onnx onnx-simplifier  onnx_graphsurgeon
RUN pip3 install sng4onnx numpy
RUN pip3 install tvm
RUN pip3 install decorator
RUN pip3 install ninja
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
# ADD  https://s3.ap-northeast-2.amazonaws.com/backend.ai/offline-installers/yolov3.pt /app/db/yolov3
RUN apt-get install -y zlib1g-dev libedit-dev libxml2-dev
RUN pip3 install pyyaml requests
# RUN pip3 install torch==1.7.0
RUN pip3 install torch
# RUN pip3 install tensorflow==2.8.0
# RUN pip3 install tensorflow==2.13.1
RUN pip3 install seaborn pandas torchvision
RUN pip3 install --upgrade attrs psutil
RUN apt-get install -y libgl1-mesa-glx 
RUN apt-get install -y vim sed 
RUN apt-get install -y clang 
RUN apt-get install -y unzip 
RUN pip3 install cmake==3.22.1
WORKDIR /app
RUN git init
ADD https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-18.1.8.zip /app
RUN unzip llvmorg-18.1.8.zip
RUN mv llvm-project-llvmorg-18.1.8 llvm-project
RUN rm llvmorg-18.1.8.zip
WORKDIR /app/llvm-project
RUN cmake -S llvm -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
RUN ninja -C build install

WORKDIR /app
RUN git clone --recursive https://github.com/apache/tvm tvm
WORKDIR /app/tvm
RUN rm -rf build && mkdir build 
WORKDIR /app/tvm/build
RUN cp ../cmake/config.cmake .
RUN echo "set(CMAKE_BUILD_TYPE RelWithDebInfo)" >> config.cmake
RUN echo "set(USE_LLVM \"llvm-config --ignore-libllvm --link-static\")" >> config.cmake
RUN echo "set(HIDE_PRIVATE_SYMBOLS ON)" >> config.cmake
RUN echo "set(USE_CUDA   OFF)" >> config.cmake
RUN echo "set(USE_METAL  OFF)" >> config.cmake
RUN echo "set(USE_VULKAN OFF)" >> config.cmake
RUN echo "set(USE_OPENCL OFF)" >> config.cmake
RUN echo "set(USE_CUBLAS OFF)" >> config.cmake
RUN echo "set(USE_CUDNN  OFF)" >> config.cmake
RUN echo "set(USE_CUTLASS OFF)" >> config.cmake
RUN cmake .. && cmake --build . --parallel $(nproc)
ENV TVM_HOME=/app/tvm 
ENV PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
ENV TVM_LIBRARY_PATH=$TVM_HOME/build
RUN pip3 install -e $TVM_HOME/python


# for android
WORKDIR /app
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y  \
    wget \
    cpio \
    sudo \
    lsb-release

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


ENV PATH=${GRADLE_HOME}/bin:${PATH}
# ENV PYTHONPATH=/opt/intel/openvino_2021/python/python3.6/:$PYTHONPATH

RUN pip3 install networkx
RUN pip3 install opencv-python
RUN pip3 install defusedxml
RUN pip3 install matplotlib

RUN apt-get install software-properties-common -y
# added by khlee to fix build error
RUN apt-get install python3-launchpadlib -y
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
# RUN wget https://github.com/ninja-build/ninja/releases/download/v1.11.1/ninja-linux.zip
# RUN unzip ninja-linux.zip


# down load android project
WORKDIR /app
RUN git clone https://github.com/lhy0718/YOLOv9-Object-Detector-Android-Tflite /app/tflite_yolov9_test

WORKDIR /app/db

WORKDIR /app
CMD ["/bin/bash"]
