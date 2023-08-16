FROM python:3.7


WORKDIR /app
COPY . /app

RUN mkdir /home/root && pip3 install kubernetes jinja2
RUN apt-get update && \
     apt-get -y install apt-transport-https \
     ca-certificates \
     curl \
     gnupg2 \
     zip \
     unzip \
     vim \
     sudo\
     software-properties-common
    
# RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
# RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -

# RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
RUN add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"


RUN apt-get update 
RUN apt-get install docker.io -y 
# EXPOSE 8089
# CMD ["python", "manage.py", "runserver", "0.0.0.0:8089"]
