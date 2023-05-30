FROM python:3.7

WORKDIR /app
COPY . /app

RUN mkdir /home/root && pip3 install kubernetes

RUN apt-get update
RUN apt-get install -y vim

# EXPOSE 8089
# CMD ["python", "manage.py", "runserver", "0.0.0.0:8089"]
