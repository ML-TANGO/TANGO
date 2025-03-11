import json
import math

from django.db.models import Model

from ..models import Project, AutonnStatus, TrainLossLastStep, ValAccuracyLastStep, EpochSummary
from ..enums import autonn_update_ids, autonn_process

def create_last_step(model_class, update_data, project_id):
    try:
        data = json.loads(update_data)
        if int(data["step"]) == (int(data["total_step"])):
            project_info = Project.objects.get(id = project_id)
            if model_class == TrainLossLastStep:
                data["project_id"] = project_id
                data["project_version"] = project_info.version
                
                train_loss_last_step = TrainLossLastStep()
                for key, value in data.items():
                    setattr(train_loss_last_step, key, value)
                train_loss_last_step.save()
            elif model_class== ValAccuracyLastStep:
                data["project_id"] = project_id
                data["project_version"] = project_info.version

                val_accuracy_laststep = ValAccuracyLastStep()
                for key, value in data.items():
                    setattr(val_accuracy_laststep, key, value)
                val_accuracy_laststep.save()

    except Exception as error:
        print("autonn_status 업데이트 오류.............")
        print(error)

def create_epoch_summary(_data, project_id):
    try:
        epoch_summary_info = json.loads(_data)
        project_info = Project.objects.get(id = project_id)
    
        epoch_summay = EpochSummary()
        epoch_summay.project_id = project_id
        epoch_summay.project_version = project_info.version
        
        for key, value in epoch_summary_info.items():
            setattr(epoch_summay, key, value)
    
        epoch_summay.save()

        # EPOCH 완료 시 autonn_retry_count 초기화 
        # (동일 EPOCH에서 failed 되었을때만 autonn을 다시 시도 하기 때문)
        project_info.autonn_retry_count = 0
        project_info.save()
    except Exception as error:
        print("create epoch summary error")
        print(error)
    
def update_instance_with_dict(model_class, update_data):
    try:
        data = json.loads(update_data)
        
        for key, value in data.items():
            setattr(model_class, key, value)

        model_class.save()
    except Exception as error:
        print("autonn_status 업데이트 오류.............")
        print(error)

def update_instance_by_system(model_class, update_data):
    try:
        data = json.loads(update_data)

        gpus = []

        for key in data.keys():
            if str(key).isdigit():
                gpus.append(data[key])

        model_class.torch = data['torch']
        model_class.cuda = data['cuda']
        model_class.cudnn = data['cudnn']
        model_class.gpus = json.dumps(gpus)
        
        # for key, value in data.items():
        #     setattr(model_class, key, value)

        model_class.save()
    except Exception as error:
        print("autonn_status - system 업데이트 오류.............")
        print(error)

def update_autonn_status(body):
    print("-----------------------------------------------")
    print(body)

    autonn_status_info = None

    try:
        autonn_status_info = AutonnStatus.objects.get(project = body["project_id"])
    except Exception:
        print("AutonnStatus를 찾을 수 없음")
        return
    
    update_id_mapping = {
        autonn_update_ids["hyperparameter"] : ("hyperparameter", autonn_status_info.hyperparameter),
        autonn_update_ids["arguments"] : ("arguments", autonn_status_info.arguments),
        autonn_update_ids["system"] : ("system", autonn_status_info.system),
        autonn_update_ids["basemodel"] : ("basemodel", autonn_status_info.basemodel),
        # autonn_update_ids["model"] : ("model", autonn_status_info.mo),
        autonn_update_ids["model_summary"] : ("model_summary", autonn_status_info.model_summary),
        autonn_update_ids["batchsize"] : ("batchsize", autonn_status_info.batch_size),
        autonn_update_ids["train_dataset"] : ("train_dataset", autonn_status_info.train_dataset),
        autonn_update_ids["val_dataset"] : ("val_dataset", autonn_status_info.val_dataset),
        autonn_update_ids["anchors"] : ("anchors", autonn_status_info.anchor),
        autonn_update_ids["train_start"] : ("train_start", autonn_status_info.train_start),
        autonn_update_ids["train_loss"] : ("train_loss", autonn_status_info.train_loss_latest),
        autonn_update_ids["val_accuracy"] : ("val_accuracy", autonn_status_info.val_accuracy_latest),
        autonn_update_ids["epoch_summary"] : ("epoch_summary", None),
        # autonn_update_ids["train_end"] : ("train_end", autonn_status_info.tr),
        # autonn_update_ids["nas_start"] : ("nas_start", autonn_status_info.),
        # autonn_update_ids["evolution_search"] : ("evolution_search", autonn_status_info.),
        # autonn_update_ids["nas_end"] : ("nas_end", autonn_status_info.),
        # autonn_update_ids["fintune_start"] : ("fintune_start", autonn_status_info.),
        # autonn_update_ids["finetue_loss"] : ("finetue_loss", autonn_status_info.),
        # autonn_update_ids["finetue_acc"] : ("finetue_acc", autonn_status_info.),
        # autonn_update_ids["finetune_end"] : ("finetune_end", autonn_status_info.),
    }

    update_id = body["update_id"]

    if update_id in autonn_process:
        autonn_status_info.progress = autonn_process[update_id]
        autonn_status_info.save()

    if update_id not in  update_id_mapping:
        print(str(update_id) + "는 허용되지 않은 Update Id 입니다.")
        return
    
    action_name, update_instance = update_id_mapping[update_id]

    if update_id == autonn_update_ids["system"]:
        update_instance_by_system(update_instance, body['update_content'])
    elif update_id == autonn_update_ids["model_summary"]:
        data = json.loads(body['update_content'])
        if 'FLOPS' in data:
            data["flops"] = data["FLOPS"]
            data.pop("FLOPS")
        update_instance_with_dict(update_instance, json.dumps(data))
    elif update_id == autonn_update_ids["epoch_summary"]:
        data = json.loads(body['update_content'])
        if 'total_time' in data:
            data['total_time'] = float(data['total_time']) * 3600
        if 'train_loss_box' in data and math.isinf(data['train_loss_box']):
            previous_epoch =  int(data['current_epoch'])-1
            previous_epoch_summary = None
            try:
                previous_epoch_summary = EpochSummary.objects.get(project_id = body['project_id'], current_epoch = previous_epoch)
                data['train_loss_box']  = float(previous_epoch_summary.train_loss_box)
                data['train_loss_total'] = float(previous_epoch_summary.train_loss_box) + float(previous_epoch_summary.train_loss_obj) + float(previous_epoch_summary.train_loss_cls)
            except EpochSummary.DoesNotExist:
                print(f"이전 Epoch_Summary를 찾을 수 없습니다. project_id = {body['project_id']}, 현재 Epoch = {int(data['current_epoch'])}")
                data['train_loss_box'] = 0
                data['train_loss_total'] = 0
        create_epoch_summary(json.dumps(data), body['project_id'])
    elif update_id == autonn_update_ids["val_accuracy"]:
        data = json.loads(body['update_content'])
        if 'class' in data:
            data['class_type'] = data['class']
            data.pop('class')
        if 'mAP50-95' in data:
            data['mAP50_95'] = data['mAP50-95']
            data.pop('mAP50-95')
        update_instance_with_dict(update_instance, json.dumps(data))
        create_last_step(ValAccuracyLastStep, json.dumps(data), body['project_id'])
    else:
        update_instance_with_dict(update_instance, body['update_content'])
        if update_id == autonn_update_ids["train_loss"]:
            create_last_step(TrainLossLastStep, body['update_content'], body['project_id'])


