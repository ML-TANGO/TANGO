import sys
from pathlib import Path

# COMMON_ROOT = Path("/shared/common")
# DATASET_ROOT = Path("/shared/datasets")
# CORE_DIR = Path(__file__).resolve().parent.parent.parent # /source/autonn_core
# sys.path.append(CORE_DIR)

import yaml, json

TASK_TO_MODEL_TABLE = {
    "detection": "yolov7",
    "classification": "resnet"
}
MODEL_TO_SIZE_TABLE = {
    "yolov7": {
        "cloud": "-tiny",
        "k8s": "-tiny",
        "k8sjetsonnano": "-tiny",
        "pcweb": "-tiny",
        "pc": "-tiny",
        "jetsonagxorin": "-tiny",
        "jetsonagxxavier": "-tiny",
        "jetsonnano": "-tiny",
        "galaxys22": "-tiny",
        "odroidn2": "-tiny",
    },
    "resnet": {
        "cloud": "20",
        "k8s": "20",
        "k8sjetsonnano": "20",
        "pcweb": "20",
        "pc": "50",
        "jetsonagxorin": "20",
        "jetsonagxxavier": "20",
        "jetsonnano": "20",
        "galaxys22": "20",
        "odroidn2": "20",
    },
}

def get_user_requirements(userid, projid):
    """
        Get user requirements(dataset, project, hyperparameters, arguments)
    """
    # ----------------------------- from P.M. ----------------------------------
    # project_info.yaml
    proj_path = COMMON_ROOT / userid / projid
    proj_yaml_path = proj_path / "project_info.yaml"
    with open(proj_yaml_path, "r") as f:
        proj_info = yaml.safe_load(f)
    proj_info_json = json.dumps(proj_info)
    status_update(userid, project_id, update_id="project_info", update_content=proj_info_json)

    # dataset.yaml
    dataset_on_proj = proj_info["dataset"]
    if os.path.isdir(str(DATASET_ROOT) + dataset_on_proj):
        dataset_yaml_path = DATASET_ROOT / dataset_on_proj / "dataset.yaml"
    else:
        print(f"There is no {DATASET_ROOT}/{dataset_on_proj}. Instead COCO dataset will be used.")
        dataset_yaml_path = DATASET_ROOT / "coco" / "dataset.yaml"

    # ---------------------------- from internal sources -----------------------
    # args.yaml
    config_path = CORE_DIR / 'tango' / 'common' / 'cfg'
    with open(config_path / 'args.yaml', encoding='utf-8') as f:
        opt = yaml.safe_load(f)
    opt_json = json.dumps(opt)
    status_update(userid, project_id, update_id="argument", update_content=proj_info_json)


    # basemodel.yaml
    basemode_yaml_path = base_model_select(userid, project_id, proj_info)
    with open(proj_path / 'basemodel.yaml') as f:
        basemodel_yaml = yaml.load(f, Loader=yaml.SafeLoader)

    # hyp.yaml
    hyp_yaml_path = config_path / f"hyp.scratch.{basemodel_yaml['hyp']}.yaml"
    with open(hyp_yaml_path) as f:
        hyp = yaml.safe_load(f)



def base_model_select(userid, project_id, proj_info, manual_select=False):
    # read condition
    task = proj_info['task_type']
    target = proj_info['target_info'].replace('-', '').replace('_', '').lower()

    # look up table
    if not manual_select:
        model = task_to_model_table[task]
        size = model_to_size_table[model][target]
    else:
        model = manual_select['model']
        size = manual_select['size']

    # save internally
    info = Info.objects.get(userid=userid, project_id=project_id)
    info.model_type = model
    info.model_size = size
    info.save()

    # store basemodel.yaml
    proj_path = COMMON_ROOT / userid / projid
    config_path = CORE_DIR / 'tango' / 'common' / 'cfg'
    source_path = f'{config_path}/{model}/{model}{model_size}.yaml'
    target_path = f'{proj_path}/basemodel.yaml'
    shutil.copy(source_path, target_path)

    # update status (P.M)
    model_p = model.upper()
    size_p = size.replace('-', '').replace('_', '').upper()
    content = {
        'model_name': model_p,
        'model_size': size_p,
    }
    status_update(userid, project_id, update_id="basemodel", update_content=content)

    return target_path


def run_autonn(userid, project_id, viz2code=False, nas=False, hpo=False):
    # Load settings
    get_user_requirements(userid, project_id)

    # temp
    import time
    time.sleep(15)
    return None
