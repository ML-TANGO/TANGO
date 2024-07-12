import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from django.apps import apps
Info = apps.get_model('autonn_core', 'Info')
Node = apps.get_model('autonn_core', 'Node')
Edge = apps.get_model('autonn_core', 'Edge')
Pth  = apps.get_model('autonn_core', 'Pth' )

import requests
import json
from pathlib import Path
COMMON_ROOT = Path("/shared/common")
DATASET_ROOT = Path("/shared/datasets")
CORE_DIR = Path(__file__).resolve().parent.parent.parent # /source/autonn_core
CFG_PATH = CORE_DIR / 'tango' / 'common' / 'cfg'

DEBUG = False

def status_update(userid, project_id, update_id=None, update_content=None):
    """
    Update AutoNN status for P.M. to visualize the progress on their dashboard
    """
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
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        # temp printing
        if DEBUG:
            import pprint
            print(f"_________POST /status_update [ {update_id} ]_________")
            pprint.pprint(update_content, indent=2, depth=3, compact=False)

        info = Info.objects.get(userid=userid, project_id=project_id)
        if update_id in ['project_info', 'hyperparameter', 'arguments', 'system',
                         'basemodel', 'batchsize', 'model', 'model_summary',
                         'train_dataset', 'val_dataset', 'anchors']:
            info.progress = "setting"
        elif update_id in ['train_start', 'train_loss', 'val_accuracy', 'epoch_summary', 'train_end']:
            info.progress = "training"
        elif update_id in ['nas_start', 'evolution_search', 'nas_end',
                           'fintune_start', 'finetune_loss', 'finetune_acc', 'finetune_end']:
            info.progress = "nas"
        else:
            print(f"[AutoNN status_update] unknown update ID {update_id}")
            info.progress = "unknown"
        info.save()

    except Exception as e:
        print(f"[AutoNN status_update] exception: {e}")


# __all__ = ['status_update']
