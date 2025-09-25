FROM python:3.10

ENV PIP_DEFAULT_TIMEOUT=1000

FROM msranni/nni

RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx
#RUN apt-get install -y postgresql postgresql-contrib

WORKDIR /source
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . .

ENV TF_ENABLE_ONEDNN_OPTS=0

#CMD ["python3", "./manage.py", "runserver", "0.0.0.0:8087"]
CMD ["bash"]
EXPOSE 8089
