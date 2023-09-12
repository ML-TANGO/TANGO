README
---

# target_image_build

## 타겟에서 실행할 추론 엔진 이미지 생성

#### DeepFramework 프로젝트 생성 (신경망 생성 및 탑재)시 실행 이미지 생성

### REST API
#### Request [ DeepFramework -> target_image_build ] - 파라미터 추가 협의 필요
    parameter : 소스 이미지 경로, 타겟 이미지 파일 저장 경로, 선택 타겟, 타겟 상세 정보, 신경망 모델 저장 경로, 신경망 실행 app 저장 경로

    const param = {
        'source_image_path': '소스 이미지 경로',
        'target_image_save_path': '타겟 이미지 저장 경로',
        'target_name': target,
        'target_os' : target_sw_info['os'],
        'target_engine' : target_sw_info['accel'],
        'target_ml_lib' : target_sw_info['mllib'],
        'target_module' : target_sw_info['dep'],
        'neural_model_save_path' : neural_model_path,
        'neural_run_app_path' : '신경망 실행 app 저장 경로',
    }

#### Response [ target_image_build -> DeepFramework ]
    parameter : 타겟 이미지 저장 경로

    const param = {
        'target_image_save_path': target_image_save_path,
    }

---
### 사용 PORT
    7007

---
### PORT 번호 변경시
####  docker-compose.yaml 파일 수정
    'target_image_build' 항목의 'command' 명령어 수정 ( 기존 7007 PORT 번호 변경 )
    'target_image_build' 항목의 'ports' 수정         ( 기존 7007 PORT 번호 변경 )

#### DeepFramework( ui_manager ) 소스코드 수정 필요
    ~/ui_manager/frontend/src/service/restDummyApi.js 파일 수정

    requestCreateImage_Dummy 함수내의 기존 7007 PORT 번호 변경

---
### Docker volumne
    사용 안함

---
### DB
    사용 안함



