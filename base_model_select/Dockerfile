FROM python:3.9

# creating a working directory
WORKDIR /home

# copy src [dest]
COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install libgl1 -y

# copying all the required materials
COPY . ./
RUN ls yolov7/ -la

# RUN echo $(tree -1)
# RUN echo $(ls -1)

# download the coco dataset (val only)
RUN wget -q https://ultralytics.com/assets/coco2017val.zip
RUN unzip -o -q coco2017val.zip -d /home/yolov7 && rm coco2017val.zip 

# python yolov5/predict.py --task evaluate --data coco.yaml --img 640 --iou 0.65 --half

# DOCKER run command
CMD ["./run_predict.sh"]

# docker build -t yolo_docker . --progress=plain
# docker run --rm --ipc=host yolo_docker 


EXPOSE 8081