README
---

# autonn 

## 신경망 생성 서버
#### DeepFramework 프로젝트 생성 (신경망 생성 및 탑재)시 신경망 생성 요청

### REST API
#### Request [ DeepFramework -> autonn ]
    parameter : 데이터셋 yaml 파일 경로, 타겟 yaml 파일 경로

    const param = {
        'data_yaml_path': dataYamlPath,
        'target_yaml_path': targetYamlPath
    }

#### Response [ autonn -> DeepFramework ]
    parameter : 신경망 모델 저장 경로, 신경망 모델 이름

    const param = {
        'neural_model_path': neural_model_path, 
        'neural_model_name': neural_model_name
    }

---
### 사용 PORT
    8087

**Note**
> URL for backbone search are temporary assigned for simple testing of container behavior.
Now you can launch web browser and open URL `http://localhost:8087/backbone`

---
### PORT 번호 변경시
####  docker-compose.yaml 파일 수정
    'autonn' 항목의 'command' 명령어 수정 ( 기존 8087 PORT 번호 변경 )
    'autonn' 항목의 'ports' 수정         ( 기존 8087 PORT 번호 변경 )

#### DeepFramework( ui_manager ) 소스코드 수정 필요
    ~/ui_manager/frontend/src/service/restDummyApi.js 파일 수정

    requestCreateNeural_Dummy 함수내의 기존 8087 PORT 번호 변경

---
### Docker volumne
    사용 안함

---
### DB
    사용 안함



