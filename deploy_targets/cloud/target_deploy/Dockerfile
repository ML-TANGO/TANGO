FROM python:3.7

WORKDIR /source
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .

# EXPOSE 8089
# CMD ["python", "manage.py", "runserver", "0.0.0.0:8089"]