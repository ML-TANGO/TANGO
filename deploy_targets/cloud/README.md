# Cloud Deploy

클라우드 환경에 서비스를 배포하기 위한 모듈의 모음.

요청시 함께 전달되는 패키지 및 의존성 정보를 통해 이미지를 생성하고, 해당 이미지를 여러 클라우드 타겟 환경에 배포하는 과정을 포함한다.

## 공통 설정에 관한 설명

TANGO는 여러 주체가 개발하는 다양한 서비스로 구성되어 있으므로, 각 서브 모듈을 다음과 같은 공통 API 인터페이스를 사용하여야 한다.
모든 요청은 `user_id`, `project_id`를 쿼리 스트링(query string) 기반의 `GET` 요청이므로 그에 대응하는 서브
모듈 핸들러를 개발하여야 한다.\
https://github.com/ML-TANGO/TANGO/wiki/Guides-%7C-Rest-API

모든 서브 모듈은 최상위 저장소의
[docker-compose.yml](https://github.com/ML-TANGO/TANGO/blob/main/docker-compose.yml)
파일을 기반으로 실행되는 구조이다. 또한, 통합 서비스가 설치된 로컬 디스크 또는 사설망 내 공용 스토리지 시스템 상에 존재하는 데이터가
컨테이너 내부의 공통 위치에 마운트되어 모든 서브 모듈이 공유하게 된다.\
https://github.com/ML-TANGO/TANGO/wiki/Guides-%7C-Exchanging-Data-among-Containers#volume-for-exchanging-data

기타 배포 탑재에 필요한 상세 스펙은 다음과 같은 `deployment.yaml` 파일로 정의된다. 현재, 컨테이너 내의 파일 위치는
`/shared/common/<user-id>/<project-id>/deployment.yaml` 이다.\
https://github.com/ML-TANGO/TANGO/wiki/Guides-%7C-Exchanging-Data-among-Containers#deploymentyaml

개발할 때 모든 통합 서버 구성 요소를 구동하는 것은 불필요하게 시간과 자원을 낭비하게 되므로, 개발용으로 클라우드 배포 관련 모듈만 띄우는
방법은 다음과 같다.
- `deploy_targets/cloud/docker-compose.yml`` 파일을 사용하여 컨테이너 서비스를 구동.
- 공용 폴더는 `deploy_targets/cloud/shared` 디렉토리를 생성하여 사용. `.gitignore` 파일에 해당 디렉토리를 추가하여 원격 저장소에 추가되지 않음.
- `deploy_targets/cloud/samples/` 아래에 모아둔 예제 파일 중 하나를 `shared/common/<user-id>/<project-id>/deployment.yaml` 파일로 복사하여 테스트용으로 사용. `<user-id>`와 `<project-id>`는 임의의 값으로 대체하여 사용할 수 있음.

### 사용 포트

현재, 클라우드 배포 모듈에서는 다음과 같은 포트를 사용한다.  사용하는 포트 변호 변경이 필요한 경우, 저장소 관리자와 사전 협의 후
[docker-compose.yml](https://github.com/ML-TANGO/TANGO/blob/main/docker-compose.yml) 파일의 `cloud_deploy` 항목을 변경한다.

Port | Description
-----|------------
8890 | 클라우드 배포 매니저 REST API 서버 포트
7007 | 이미지 빌더 서버 포트(클라우드 배포 매니저에서만 통신하면 되므로 실서버에서는 노출하지 않아도 무출)
8080 | 이미지 빌더 GUI 서버 포트(전용 GUI는 통합 매니저에서 사용하지 않으므로 실서버에서는 노출하지 않아도 무방)


### REST API 스펙

현재, 클라우드 배포 모듈에서는 다음과 같은 API를 제공한다.

Method | Path | Name | Request parameters | Success status code | Response content
-------|------|------|--------------------|----------------------|-----------------
GET | / | Hello | No parameters | 200 | No response body
GET | /start | Start service | `user_id`, `project_id` | 200 | `started` / `error` / ...
GET | /stop | Stop service | `user_id`, `project_id` | 200 | `finished` / `error` / ...
GET | /status_request | Service status | `user_id`, `project_id` | 200 | `running` / `stopped` / ...

보다 상세한 최신 API 스펙은 클라우드 배포 모듈을 구동한 후 `/docs` 경로로 접근하면 Swagger UI를 통해 확인할 수 있다. (예: http://localhost:8890/docs)


### 환경 변수 설정

클라우드 배포 모듈은 다양한 클라우드 서비스의 인증 정보 및 설정을 환경 변수로 관리한다.

**`.env` 파일 사용 (권장)**

매번 `export` 명령으로 환경 변수를 설정할 필요 없이, 프로젝트 루트에 `.env` 파일을 생성하여 환경 변수를 관리할 수 있다.

1. `.env.example` 파일을 복사하여 `.env` 파일 생성:
   ```bash
   cp .env.example .env
   ```

2. `.env` 파일을 편집하여 실제 인증 정보 입력:
   ```bash
   # Compute Session 타입 배포 설정
   CS_ENDPOINT=https://api.example.ai
   CS_ACCESS_KEY=실제_액세스_키
   CS_SECRET_KEY=실제_시크릿_키

   # GCP 설정
   GCP_PROJECT_ID=실제_프로젝트_ID

   # AWS 설정
   AWS_ACCESS_KEY_ID=실제_액세스_키
   AWS_SECRET_ACCESS_KEY=실제_시크릿_키
   ```

3. `docker-compose.yml`이 자동으로 `.env` 파일을 읽어 컨테이너에 환경 변수 전달

**참고**: `.env` 파일은 `.gitignore`에 포함되어 있어 Git 저장소에 커밋되지 않는다.


## 대상 타겟 장치

현재, 클라우드 배포 모듈에서는 다음과 같은 클라우드 타겟 장치를 지원한다.

### Google Cloud Platform

Google Cloud Platfomr (GCP)를 통해 배포하기 위해서는 최소한 다음과 같은 사전 준비가 필요하다.
- Google Cloud Project 계정 및 프로젝트 생성
- 사용할 프로젝트에 대한 결제 설정
- 배포시 권한 문제를 최소화 하기 위해 계정의 역할을 "소유자"로 설정하는 것이 좋음
    - GCP의 "IAM 및 관리자" 페이지 이동
    - IAM 메뉴로 가서 서비스 계정 이메일과 역할 확인
    - 역할이 "소유자"가 아닌 경우 "소유자"로 변경
        - 우측에 있는 편집 버튼("주 구성원 수정")을 클릭
    <img width="804" alt="image" src="https://github.com/ML-TANGO/TANGO/assets/7539358/f350688a-72b0-4ca7-b3ec-4de070c48b30">

계정 생성 및 권한 설정이 완료되었으면, 클라우드 배포 모듈 실행시 다음과 같은 환경 변수를 설정해야 한다(구체적인 값은 예시).
```shell
# 서비스 계정 키는 GCP 상 "IAM 및 관리자" - "서비스 계정" - "키" 페이지에서 발급 가능.
GOOGLE_APPLICATION_CREDENTIALS=/source/cloud_manager/service-account-file.json
# 사용할 GCP 리전: 서울은 asia-northeast3
GCP_REGION=asia-northeast3
# 사용할 GCP 프로젝트 ID
GCP_PROJECT_ID=tango-testbed
```

#### Image upload to Google Cloud Platform

```shell
gcloud init
gcloud auth configure-docker asia-northeast3-docker.pkg.dev
docker push ...
# asia-northeast3-docker.pkg.dev/backendai-testbed/tango/pytorch:21.08-py3
```

#### Google Cloud Run

보통 클라우드 서비스는 컨테이너 기반 배포를 다양한 방식으로 지원한다. 현재, TANGO의 클라우드 배포 매니저는 Google Cloud
Platform의 서버리스 컨테이너 관리형 솔루션인 [Google Cloud Run](https://cloud.google.com/run)를
타겟 장치로 통합하여 서비스를 배포할 수 있다.

`deploy_targets/cloud/samples/deployment_gcp_cloudrun.yaml` 파일을 참고하여 YAML 스펙을
정의해서 실행해야 한다. 특히, `deploy.type`의 값을 `gcp-cloudrun`으로 설정한다.


### AWS Elastic Container Service (ECS)

AWS 상의 ECS 서비스를 통해 배포하기 위해서는 최소한 다음과 같은 사전 준비가 필요하다.
- AWS 계정 생성
- 계정에 API/SDK 접속해서 사용하기 위한 Access / Secret 키페어 생성

AWS 접속 정보 및 키페어는 `.env` 파일 등에 다음과 같이 설정하여 컨테이너 내
서비스가 읽을 수 있도록 하여야 함.

```shell
AWS_REGION=ap-northeast-2
AWS_ACCESS_KEY_ID=AKIA****
AWS_SECRET_ACCESS_KEY=Bq1N****
```

`deploy_targets/cloud/samples/deployment_aws_ecs.yaml` 파일을 참고하여 YAML 스펙을
정의해서 실행해야 한다. 특히, `deploy.type`의 값을 `aws-ecs`으로 설정한다.


### Compute Session

다양한 이기종 가속기(GPU, NPU, TPU)를 지원하는 Sokovan 스케줄러를 이용하는 방식.
TANGO 클라우드 배포 매니저는 Compute Session(연산 세션) 기능을 통해 AI/ML 추론
서비스를 배포할 수 있다.

**주요 지원 가속기**
- GPU (NVIDIA, AMD)
- NPU (Rebellion Atom)
- CPU

**사전 준비**
- Sokovan 계정 및 접속 정보 확보
- Access Key 및 Secret Key 발급

**환경 변수 설정**

`.env` 파일에 다음 정보를 설정:

```shell
CS_ENDPOINT=https://api.example.ai
CS_ACCESS_KEY=AKIA...
CS_SECRET_KEY=실제_시크릿_키
CS_DEFAULT_DOMAIN=default
CS_DEFAULT_GROUP=default
```

**배포 예시**

다음 샘플 파일들을 참고하여 YAML 스펙을 정의:

1. **GPU 기반 추론 서비스**: `deploy_targets/cloud/samples/deployment_compute_session.yaml`
2. **Rebellion Atom NPU 추론 서비스**: `deploy_targets/cloud/samples/deployment_rebellion_atom.yaml`

`deploy.type`의 값을 `compute-session`으로 설정한다.

**가속기 스펙 설정**

Build 섹션에서 가속기를 다음과 같이 지정:

```yaml
build:
  accelerator:
    type: gpu          # cpu, gpu, npu, tpu
    vendor: nvidia     # nvidia, amd, rebellion, google
    count: 1           # 가속기 개수
```

또는 간단한 문자열 형식:

```yaml
build:
  accelerator: "npu"  # 또는 "npu"
deploy:
  type: compute-session
  service_name: Qwen25-14b
  resources:
    cpu: 4
    memory: 65536
    atom-max.device: 1
  model_service:
    model: Qwen25-14b
    scaling_group: default
```
