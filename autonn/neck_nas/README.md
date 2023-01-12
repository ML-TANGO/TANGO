README
---

# autonn/neck_nas

**TL;DR**
```bash
docker rm -f $(docker ps -aq)
docker rmi autonn_nk:lastest
cd {tango_root}/autonn/neck_nas
docker build -t autonn_nk .
docker run -it --gpu=all -p 8089:8089 --name=autonn_nk autonn_nk:lastest
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver 0.0.0.0:8089
curl -v http://localhost:8089/start?user_id=root\&project_id=20230118
curl -v http://localhost:8089/status_request?user_id=root\&project_id=20230118
curl -v http://localhost:8089/stop?user_id=root\&project_id=20230118
```

## TANGO 신경망 자동 생성 모듈 / 'Neck' Network Architeture Search 컨테이너
#### 도커 서비스 이름
    autonn_nk

### REST APIs
#### GET /start?userid=<userid>&project_id=<project_id>
    parameter : userid, project_id
    return: 200 OK
    return content: "starting" / "error"
    return content-type: text/plain

#### GET /stop?userid=<userid>&project_id=<project_id>
    parameter : userid, project_id
    return: 200 OK
    return content: "finished" / "error"
    return content-type: text/plain

#### GET /status_request?userid=<userid>&project_id=<project_id>
    parameter : userid, project_id
    return: 200 OK
    return content: "ready" / started" / "runnung" / "stopped" / "failed" / "completed"
    return content-type: text/plain

### 사용 PORT
    8089

**Note**
> URL for neck architeture search is temporary assigned for simple testing of container behavior.
Now you can launch web browser and open URL `http://localhost:8089/`
And run using CURL 'curl -v http://localhost:8089/start?userid=root\&project_id=2022xxxx'


---
### PORT 번호 변경시
####  docker-compose.yaml 파일 수정
    'autonn_nk' 항목의 'command' 명령어 수정 ( 기존 8089 PORT 번호 변경 )
    'autonn_nk' 항목의 'ports' 수정         ( 기존 8089 PORT 번호 변경 )

#### Project Manager 소스코드 수정 필요
    ~/project_manager/frontend/src/service/restDummyApi.js 파일 수정

    requestCreateNeural_Dummy 함수내의 기존 8089 PORT 번호 변경

---
### Docker volumne
    ./autonn/neck_nas --> /source
    ./shared          --> shared

---
### DB
    ~/autonn/neck_nas/backend/setting.py 파일 수정
    Django 기본 DB인 sqlite3 사용 (./db.sqlite3/)

