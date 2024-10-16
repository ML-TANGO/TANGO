FROM python:3.8
WORKDIR /app
RUN pip3 install --upgrade pip

RUN apt-get update
RUN apt-get install -y vim vlc 

RUN pip install --upgrade pip
RUN pip install cython
RUN pip install "numpy<17"
RUN pip install imutils
RUN pip install flask
RUN pip install opencv-python
RUN pip install opencv-contrib-python
RUN pip install imageio imageio-ffmpeg
RUN pip install pyyaml matplotlib
RUN pip install torch torchvision 
RUN pip install pandas tqdm 
RUN pip install seaborn  
RUN pip install requests  

CMD ["/bin/bash"]
