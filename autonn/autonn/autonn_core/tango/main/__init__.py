import sys
from pathlib import Path

COMMON_ROOT = Path("/shared/common")
DATASET_ROOT = Path("/shared/datasets")
CORE_DIR = Path(__file__).resolve().parent.parent.parent # /source/autonn_core
sys.path.append(CORE_DIR)

print(f"TANGO module is initialized...// appending path CORE_DIR = {CORE_DIR}")
# from autonn_core.models import Info
from django.apps import apps
Info = apps.get_model('autonn_core', 'Info')

import requests

def status_update(userid, project_id, update_id=None, update_content=None):
    """
        Update AutoNN status for P.M. to visualize the progress on their dashboard
    """
    try:
        url = 'http://projectmanager:8085/status_update'
        headers = {
            'Content-Type' : 'text/json'
        }
        payload = {
            'container_id' : "autonn",
            'user_id' : userid,
            'project_id' : project_id,
            'update_id' : update_id,
            'update_content' : update_content,
        }
        response = requests.get(url, headers=headers, params=payload)
        # temp printing
        print("GET /status_update")
        print(f"-------------{update_id}------------")
        print(f"{update_content}")

        info = models.Info.objects.get(userid=userid, project_id=project_id)
        if update_id in ['basemodel', 'model', 'model_summary',
                            'train_dataset', 'val_dataset', 'anchor']:
            info.progress = "setting"
        elif update_id in ['train_start', 'train_loss', 'val_accuracy', 'train_end']:
            info.progress = "training"
        elif update_id in ['nas_start', 'evolution_search', 'nas_end',
                              'fintune_start', 'finetue_loss', 'finetue_acc', 'finetune_end']:
            info.progress = "nas"
        else:
            info.progress = "unknown"
        info.save()
    except Exception as e:
        print(f"[AutoNN status_update] exception: {e}")
