import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import json
from pathlib import Path
import requests
from django.apps import apps

COMMON_ROOT = Path("/shared/common")
DATASET_ROOT = Path("/shared/datasets")
MODEL_ROOT = Path("/shared/models")
CORE_DIR = Path(__file__).resolve().parent.parent.parent # /source/autonn_core
CFG_PATH = CORE_DIR / 'tango' / 'common' / 'cfg'

DEBUG = False

def is_distributed() -> bool:
    return os.environ.get("WORLD_SIZE", "1") not in ("1", "", None)

def is_rank0() -> bool:
    return str(os.environ.get("RANK", "0")) == "0"

def _ensure_django_ready():
    """
    Django 앱 레지스트리가 준비되었는지 보장
        - 이미 준비되었으면 True
        - 준비되지 않았고 settings 모듈을 알면 setup() 시도
        - 실패하면 False
    """
    if apps.ready:
        return True
    try:
        import django
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "autonn_core.settings")
        django.setup()
        return True
    except Exception as e:
        print(f"Django setup error: {e}")
        return False

def get_model(model_name: str):
    """
    모델 지연(lazy) 획득, 준비 안되었으면 None 변환
    """
    if not _ensure_django_ready():
        return None
    try:
        return apps.get_model('autonn_core', model_name)
    except Exception as e:
        print(f"Django model error: {e}")
        return None

def status_update(userid, project_id, update_id=None, update_content=None):
    """
    Update AutoNN status for P.M. to visualize the progress on their dashboard
    """
    if is_distributed() and not is_rank0():
        return
    try:
        url = 'http://projectmanager:8085/status_update'
        headers = {
            'Content-Type' : 'application/json'
        }
        payload = {
            'container_id' : "autonn",
            'user_id' : userid,
            'project_id' : project_id,
            'update_id' : update_id,
            'update_content' : json.dumps(update_content),
        }
        response = requests.post(url, headers=headers, 
                data=json.dumps(payload), timeout=3)
        # temp printing
        if DEBUG:
            import pprint
            print(f"_________POST /status_update [ {update_id} ]_________")
            pprint.pprint(update_content, indent=2, depth=3, compact=False)
    except Exception as e:
        print(f"[AutoNN status_update] exception: {e}")

__all__ = [
    "COMMON_ROOT", "DATASET_ROOT", "MODEL_ROOT", "CORE_DIR", "CFG_PATH",
    "status_update", "get_model",
]
