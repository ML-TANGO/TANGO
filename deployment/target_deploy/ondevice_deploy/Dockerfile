FROM python:3.6
WORKDIR /app
COPY . /app
RUN pip3 install --upgrade pip
RUN apt-get update
RUN apt-get install -y vim 
RUN pip3 install pyyaml requests
CMD ["/bin/bash"]
