import os
from pathlib import Path
import requests

COMMON_ROOT = Path("/shared/common")
SOURCE_ROOT = Path("/source")
MODEL_URL = "https://github.com/ML-TANGO/TANGO/releases/download/Model_Mirror/kagglecxr_resnet152_normalize.pt"

def deploy_resnet152(userid, project_id, data):
    source_dir: Path = SOURCE_ROOT / "autonn_core/tango/main/PneumoniaDeploy"
    save_dir: Path = COMMON_ROOT / userid / project_id / "nn_model"
    # 저장 디렉토리가 존재하면 삭제
    if save_dir.exists():
        os.system(f"rm -rf {save_dir}")
    # 소스 디렉토리에 있는 배포 코드를 재귀 복사
    for src in source_dir.rglob("*"):
        if src.is_file():
            dest = save_dir / src.relative_to(source_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(src.read_bytes())
    # 데이터셋 복사
    data_dir = save_dir / "dataset"
    data_dir.mkdir(parents=True, exist_ok=True)
    for key, value in data.items():
        if key in ["train", "val"]:
            dest = data_dir / key
            dest.mkdir(parents=True, exist_ok=True)
            for src in Path(value).rglob("*"):
                if src.is_file():
                    dest_file = dest / src.relative_to(value)
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    dest_file.write_bytes(src.read_bytes())
    # 모델 다운로드
    model_file = save_dir / "pretained_model/kagglecxr_resnet152_normalize.pt"
    model_file.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(MODEL_URL, stream=True) as r:
        r.raise_for_status()
        with model_file.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return
