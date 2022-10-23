FROM python:3.10

RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx
RUN apt-get install -y postgresql postgresql-contrib

WORKDIR /souce
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . .

CMD ["python3", "./manage.py", "runserver", "0.0.0.0:8087"]
EXPOSE 8087