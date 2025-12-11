from enum import Enum

class ContainerId(str, Enum):
    autonn = "autonn"
    autonn_cl = "autonn_cl"  # Continual Learning을 위한 AutoNN_CL 컨테이너 추가
    codeGen = "code_gen"
    imagedeploy = "imagedeploy"

class ContainerStatus(str, Enum):
    READY = "ready"
    STARTED = "started"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"

class LearningType(str, Enum):
    NORMAL = "normal"
    INCREMENTAL = "incremental"
    TRANSFER = "transfer"
    HPO="HPO"
    CONTINUAL_LEARNING = "continual_learning"  # Segmentation을 위한 Continual Learning Type 추가

class TaskType(str, Enum):
    DETECTION = "detection"
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"  # Segmentation 기능을 위한 Task Type 추가

autonn_update_ids={
    "project_info":"project_info",
    "hyperparameter":"hyperparameter",
    "arguments":"arguments",
    "system":"system",
    "basemodel":"basemodel",
    "model":"model",
    "model_summary":"model_summary",
    "batchsize":"batchsize",
    "train_dataset":"train_dataset",
    "val_dataset":"val_dataset",
    "anchors":"anchors",
    "train_start":"train_start",
    "train_loss":"train_loss",
    "val_accuracy":"val_accuracy",
    "epoch_summary":"epoch_summary",
    "train_end":"train_end",
    "nas_start":"nas_start",
    "evolution_search":"evolution_search",
    "nas_end":"nas_end",
    "fintune_start":"fintune_start",
    "finetue_loss":"finetue_loss",
    "finetue_acc":"finetue_acc",
    "finetune_end":"finetune_end",
}

autonn_process={
    autonn_update_ids["project_info"]: 0,
    autonn_update_ids["system"]: 1,
    autonn_update_ids["model_summary"]: 1.5,
    autonn_update_ids["train_dataset"]: 2,
    autonn_update_ids["anchors"]: 3,
    autonn_update_ids["train_end"]: 4,
    autonn_update_ids["finetune_end"]: 5,
}
