import json

from django.db.models import Model

from ..models import Project, Hyperparameter, AutonnStatus, TrainLossLastStep, ValAccuracyLastStep
from ..enums import autonn_update_ids, autonn_process

def create_last_step(model_class, update_data, project_id):
    try:
        data = json.loads(update_data)
        if int(data["step"]) == (int(data["total_step"]) - 1):
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

# def update_autonn_status(body):
#     print("-----------------------------------------------")
#     print(body)

#     autonn_status_info = None

#     try:
#         autonn_status_info = AutonnStatus.objects.get(project = body["project_id"])
#     except Exception:
#         print("AutonnStatus를 찾을 수 없음")
#         return

#     if body["update_id"] == autonn_update_ids["project_info"]:
#         print("project_info")
#     elif body["update_id"] == autonn_update_ids["hyperparameter"]:
#         print("hyperparameter")
#         update_instance_with_dict(autonn_status_info.hyperparameter, body['update_content'])
#     elif body["update_id"] == autonn_update_ids["arguments"]:
#         print("arguments")
#         update_instance_with_dict(autonn_status_info.arguments, body['update_content'])
#     elif body["update_id"] == autonn_update_ids["system"]:
#         print("system")
#         update_instance_by_system(autonn_status_info.system, body['update_content'])
#     elif body["update_id"] == autonn_update_ids["basemodel"]:
#         print("basemodel")
#         update_instance_with_dict(autonn_status_info.basemodel, body['update_content'])
#     elif body["update_id"] == autonn_update_ids["model"]:
#         print("model")
#         # update_instance_with_dict(autonn_status_info.mo, body['update_content'])
#     elif body["update_id"] == autonn_update_ids["model_summary"]:
#         print("model_summary")
#         update_instance_with_dict(autonn_status_info.model_summary, body['update_content'])
#     elif body["update_id"] == autonn_update_ids["batchsize"]:
#         print("batchsize")
#         update_instance_with_dict(autonn_status_info.batch_size, body['update_content'])
#     elif str(body["update_id"]) == autonn_update_ids["train_dataset"]:
#         print("train_dataset")
#         update_instance_with_dict(autonn_status_info.train_dataset, body['update_content'])
#     elif body["update_id"] == autonn_update_ids["val_dataset"]:
#         print("val_dataset")
#         update_instance_with_dict(autonn_status_info.val_dataset, body['update_content'])
#     elif body["update_id"] == autonn_update_ids["anchors"]:
#         print("anchors")
#         update_instance_with_dict(autonn_status_info.anchor, body['update_content'])
#     elif body["update_id"] == autonn_update_ids["train_start"]:
#         print("train_start")
#         update_instance_with_dict(autonn_status_info.train_start, body['update_content'])
#     elif body["update_id"] == autonn_update_ids["train_loss"]:
#         print("train_loss")
#         update_instance_with_dict(autonn_status_info.train_loss_latest, body['update_content'])
#         create_last_step(TrainLossLastStep, body['update_content'], body['project_id'])
#     elif body["update_id"] == autonn_update_ids["val_accuracy"]:
#         print("val_accuracy")
#         # class라는 key는 python에서 사용할 수 없으므로...
#         data = json.loads(body['update_content'])
#         if 'class' in data:
#             data['class_type'] = data['class']
#             data.pop('class')
#         update_instance_with_dict(autonn_status_info.val_accuracy_latest, json.dumps(data))
#         create_last_step(ValAccuracyLastStep, json.dumps(data), body['project_id'])
#     elif body["update_id"] == autonn_update_ids["train_end"]:
#         print("train_end")
#     elif body["update_id"] == autonn_update_ids["nas_start"]:
#         print("nas_start")
#     elif body["update_id"] == autonn_update_ids["evolution_search"]:
#         print("evolution_search")
#     elif body["update_id"] == autonn_update_ids["nas_end"]:
#         print("nas_end")
#     elif body["update_id"] == autonn_update_ids["fintune_start"]:
#         print("fintune_start")
#     elif body["update_id"] == autonn_update_ids["finetue_loss"]:
#         print("finetue_loss")
#     elif body["update_id"] == autonn_update_ids["finetue_acc"]:
#         print("finetue_acc")
#     elif body["update_id"] == autonn_update_ids["finetune_end"]:
#         print("finetune_end")

#     print('\n\n')







