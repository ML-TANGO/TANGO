FROM python:3.10

ENV PIP_DEFAULT_TIMEOUT=1000

RUN apt-get update
RUN apt-get install -y postgresql postgresql-contrib
RUN apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx libglib2.0-0

WORKDIR /source
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
COPY . .

# RUN export PATH="$PATH:../usr/local/lib/python3.10/site-packages/graphviz"

CMD ["bash"]
EXPOSE 8087

