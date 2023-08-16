
FROM python:3.7

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /code
COPY requirements.txt ./
COPY base_model_data.json ./
RUN pip install -r requirements.txt

COPY . .