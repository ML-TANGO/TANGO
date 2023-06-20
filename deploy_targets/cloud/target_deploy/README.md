README
---

# target_deploy 

## 실행 이미지 탑재 서버
#### DeepFramework 프로젝트 생성 (신경망 생성 및 탑재)시 실행 이미지 탑재

### REST API
#### Request [ DeepFramework -> target_deploy ]
    parameter : 타겟 이미지 저장 경로, 타겟 URL, 실행 명령어

    const param = {
        'target_image_save_path': target_image_path,
        'target_url': '타겟 URL',
        'startup_command' : './rk3399pro-onnx-detector --model=best_detnn-coco-rk3399pro.onnx'
    }

#### Response [ target_deploy -> DeepFramework ]
    HTPP STATUS CODE 200 응답

---
### 사용 PORT
    8089

---
### PORT 번호 변경시
####  docker-compose.yaml 파일 수정
    'target_deploy' 항목의 'command' 명령어 수정 ( 기존 8089 PORT 번호 변경 )
    'target_deploy' 항목의 'ports' 수정         ( 기존 8089 PORT 번호 변경 )

#### DeepFramework( ui_manager ) 소스코드 수정 필요
    ~/ui_manager/frontend/src/service/restDummyApi.js 파일 수정

    requestDeployImage_Dummy 함수내의 기존 8089 PORT 번호 변경

---
### Docker volumne
    사용 안함

---
### DB
    사용 안함



