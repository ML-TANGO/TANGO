FROM python:3.9

# creating a working directory
WORKDIR /yolov5

# copy src [dest]
COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN apt-get update
#RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install libgl1 -y

# copying all the required materials
COPY . ./
RUN ls -la

# download the coco dataset (val only)
#RUN wget -q https://ultralytics.com/assets/coco2017val.zip
#RUN unzip -q coco2017val.zip -d datasets && rm coco2017val.zip 
#RUN mv evaluate2017.txt datasets/coco/evaluate2017.txt
RUN python generate_txt_file.py

# DOCKER run command
#CMD ["python", "yolov5/predict.py", "--task", "evaluate", "--data", "coco.yaml", "--img", "640", "--iou", "0.65", "--half", "runserver", "0.0.0.0:8081"]
CMD ["python", "yolov5/predict.py", "--task", "evaluate", "--data", "coco.yaml", "--img", "640", "--iou", "0.65", "--half"]
#CMD ["python", "val.py", "--weights", "yolov5s.pt", "--data", "coco.yaml", "--img", "640", "--iou", "0.65", "--half"]

EXPOSE 8081