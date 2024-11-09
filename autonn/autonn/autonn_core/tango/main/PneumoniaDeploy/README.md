# 폐렴 진단 모델 진단 도구

## 실행 방법

현재 윈도우즈 환경에서 venv 환경 혹은 conda 환경으로 실행할 수 있습니다.

### Windows에서 venv 가상환경으로 실행

```ps1
# 환경 설치 스크립트 실행
./install_windows_venv.ps1

# 가상환경에서 실행
./run_windows_venv.ps1
```

> Windows PowerShell에서 스크립트 실행 권한이 없을 경우: PowerShell을 관리자 권한으로 실행 후 `Set-ExecutionPolicy RemoteSigned` 명령어를 실행합니다.

### Windows에서 Conda 및 Miniconda로 실행

```ps1
# 환경 설치 스크립트 실행
./install_windows_conda.ps1

# 가상환경에서 실행
./run_windows_conda.ps1
```

> **주의**: MiniConda를 사용하는 경우 conda 도구 경로가 Path에 추가되지 않을 수 있습니다. 이 경우 `Anaconda Powershell Prompt (miniconda3)`를 실행한 뒤 명령어를 실행해주세요.

## 평가 데이터셋

제공하는 평가 데이터셋은 `dataset` 디렉토리에 있습니다. 이미지 파일을 재귀적으로 수집하여 분석 대상 이미지로 등록합니다. 각 데이터의 정답(Ground Truth)은 부모 폴더의 이름(`NORMAL`, `PNEUMONIA`)으로 정의합니다. 레이블은 대소문자를 구분하지 않습니다.

## 모델 파일

`pretained_model` 디렉토리의 모델 파일을 사용합니다. 현재 모델 파일 변경은 `src/DiagThread.py` 파일의 `run()` 함수에서 직접 변경할 수 있습니다.
