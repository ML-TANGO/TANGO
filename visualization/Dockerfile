FROM python:3.10
USER root
WORKDIR /souce
COPY requirements.txt ./
RUN python3 --version
RUN pip install anyio==3.5.0 asgiref==3.5.0
RUN pip install certifi==2021.10.8 charset-normalizer==2.0.12
RUN pip install click==8.0.4 colorama==0.4.4
RUN pip install fastapi==0.74.0 h11==0.13.0
RUN pip install idna==3.3 numpy==1.22.3
RUN pip install Pillow==9.0.1 pydantic==1.9.0
RUN pip install pytz==2021.3 requests==2.27.1
RUN pip install sniffio==1.2.0 sqlparse==0.4.2
RUN pip install starlette==0.17.1 typing-extensions==4.1.1
RUN pip install urllib3==1.26.9 uvicorn==0.17.5
RUN pip install Django==3.2
RUN pip install django-cors-headers==3.11.0 django-rest-framework==0.1.0 djangorestframework==3.13.1 psycopg2 six
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
COPY . .
RUN pwd
RUN ls
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y nodejs \
    npm                       # note this one
RUN apt-get install -y postgresql postgresql-contrib
RUN npm install pnpm -g
RUN npm install
WORKDIR /souce/frontend
RUN npm run build
WORKDIR /souce/
CMD ["python3", "./manage.py", "runserver", "0.0.0.0:8091"]
EXPOSE 8091
