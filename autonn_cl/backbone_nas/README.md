README
---

# autonn/backbone_nas

## TANGO 신경망 자동 생성 모듈 / 'Backbone' Network Architeture Search 컨테이너
#### 도커 서비스 이름
    autonn_bb

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
    8087

**Note**
Now you can launch web browser and open URL `http://localhost:8087/`
And run using CURL 'curl -v http://localhost:8087/start?userid=root\&project_id=2022xxxx'