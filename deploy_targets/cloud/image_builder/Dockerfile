FROM python:3.9

WORKDIR /source
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .

# EXPOSE 7007
# CMD ["python", "manage.py", "runserver", "0.0.0.0:7007"]
