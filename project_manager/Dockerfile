
# FROM node:16-alpine

# WORKDIR /code/frontend
# COPY ./frontend .

# RUN npm i
# # RUN npm run build
# # RUN npm run serve
# CMD ["npm", "run", "build"]
# # ===================================================


FROM python:3.7

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /code
COPY requirements.txt ./
COPY base_model_data.json ./
RUN pip install -r requirements.txt

COPY . .
# CMD ["python", "manage.py", "runserver", "0.0.0.0:8085"]