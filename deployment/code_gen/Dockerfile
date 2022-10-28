FROM python:3.6
WORKDIR /app
COPY . /app
COPY lib/. /app
COPY lib/libprotobuf.so.10 /lib/x86_64-linux-gnu
COPY db/. /app/db
COPY db/yolov3.body /app/db
COPY db/yolov3.head /app/db
COPY db/yolo_v3_tiny_darknet_fp32.tflite /app/db
COPY db/odroid-m1-manual.txt /app/db
COPY db/odroid-n2-manual.txt /app/db
COPY db/yolov3/. /app/db/yolov3
ADD  https://s3.ap-northeast-2.amazonaws.com/backend.ai/offline-installers/yolov3.pt /app/db/yolov3
RUN pip3 install --upgrade pip
RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx vim gedit
RUN pip3 install rknn_toolkit2-1.2.0_f7bb160f-2-cp36-cp36m-linux_x86_64.whl
RUN pip3 install pyyaml requests
CMD ["/bin/bash"]
