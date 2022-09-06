'''
main.py
'''

import os
from pathlib import Path

from .search import arch_search

# Path on Current abs. dir.
PATHA = os.path.dirname(os.path.abspath(__file__))
# Path on backboneNAS
BASEPATH = Path(PATHA).parent

# if str(basePath) not in sys.path:
#    sys.path.append(str(basePath))

# ROOT = ROOT.parent.parent


def run_nas(
        data_path='',
        # base_model = ROOT / 'temp_ui/yolov5s.yaml',
        # base_model_weights = ROOT / 'temp_ui/yolov5s.pt',
        # save_path = ROOT / 'temp_ui/best_det_model.pt',
        device=0,
        batch_size=16
):
    '''
    main func
    '''
    data_path = str(BASEPATH) + data_path
    base_model_weights = os.path.dirname(data_path) + '/model/yolov5s.pt'
    best_det_model = arch_search(
        data_path, base_model_weights, batch_size, device)
    # torch.save(best_det_model.state_dict(), save_path)
    return best_det_model
