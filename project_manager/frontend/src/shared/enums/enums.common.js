export const TaskType = Object.freeze({
  DETECTION: "detection",
  CLASSIFICATION: "classification",
  CHAT: "chat",
  SEGMENTATION: "segmentation" // Segmentation 기능을 위한 Task Type 추가
});

// 사용안함
export const ContainerPort = Object.freeze({
  bms: "8081",
  vis2code: "8091",
  autonn_bb: "8087",
  autonn_nk: "8089",
  code_gen: "8888",
  cloud_deployment: "8088",
  ondevice_deployment: "8891",
  yolo_e: "8090",
  lablilng: "8095",
  autonn: "8100"
});

export const DataType = {
  I: "Image",
  V: "Video"
};

export const ObjectType = {
  C: "Classification",
  D: "Detection",
  S: "Segmentation"
};

export const ProjectRequiredColumn = [
  // "dataset",
  "target_id",
  "target_id",
  "task_type",
  "nas_type",
  "deploy_weight_level",
  "deploy_precision_level",
  "deploy_user_edit",
  "deploy_output_method"
];

export const ContainerName = {
  BMS: "bms",
  AUTO_NN: "autonn",
  AUTO_NN_CL: "autonn_cl",  // AutoNN_CL 추가
  AUTO_NN_YOLOE: "yoloe",
  AUTO_NN_RESNET: "autonn-resnet",
  CODE_GEN: "code_gen",
  IMAGE_DEPLOY: "imagedeploy",
  USER_EDITING: "user_edit",
  VISUALIZATION: "viz2code"
};

export const DisplayName = {
  [ContainerName.BMS]: "BMS",
  [ContainerName.AUTO_NN]: "Auto NN",
  [ContainerName.AUTO_NN_CL]: "AutoNN CL",  // AutoNN_CL 표시명 추가
  [ContainerName.AUTO_NN_YOLOE]: "Auto NN",
  [ContainerName.AUTO_NN_RESNET]: "Auto NN",
  [ContainerName.CODE_GEN]: "Code Gen",
  [ContainerName.IMAGE_DEPLOY]: "Image Deploy",
  [ContainerName.USER_EDITING]: "User Editing",
  [ContainerName.VISUALIZATION]: "VISUALIZATION"
};

export const EngineValues = {
  ACL: "acl",
  PYTORCH: "pytorch",
  TVM: "tvm",
  TENSORRT: "tensorrt",
  TFLITE: "tflite",
  RKNN: "rknn",
  ONNX: "onnx",
  OPEN_VINO: "OpenVINO"
};

export const EngineLabel = {
  [EngineValues.ACL]: "ACL",
  [EngineValues.PYTORCH]: "Pytorch",
  [EngineValues.TVM]: "TVM",
  [EngineValues.TENSORRT]: "Tensorrt",
  [EngineValues.TFLITE]: "TFLite",
  [EngineValues.RKNN]: "RKNN"
};

export const TargetInfoList = [
  {
    key: "Cloud",
    value: "Cloud",
    allowedEngine: [EngineValues.PYTORCH, EngineValues.TENSORRT],
    requiredFields: ["target_hostip", "target_hostport"]
  },
  {
    key: "K8S",
    value: "K8S",
    allowedEngine: [EngineValues.PYTORCH, EngineValues.TENSORRT],
    requiredFields: ["target_hostip", "target_hostport", "nfs_ip", "nfs_path"]
  },
  {
    key: "K8S_Jetson_Nano",
    value: "K8S_Jetson_Nano",
    allowedEngine: [EngineValues.PYTORCH, EngineValues.TENSORRT],
    requiredFields: ["target_hostip", "target_hostport", "nfs_ip", "nfs_path"]
  },
  {
    key: "PC_Web",
    value: "PC_Web",
    allowedEngine: [EngineValues.PYTORCH, EngineValues.TENSORRT],
    requiredFields: ["target_hostip", "target_hostport"]
  },
  {
    key: "PC",
    value: "PC",
    allowedEngine: [EngineValues.PYTORCH, EngineValues.TENSORRT],
    requiredFields: []
  },
  {
    key: "Jetson_AGX_Orin",
    value: "Jetson_AGX_Orin",
    allowedEngine: [EngineValues.PYTORCH, EngineValues.TENSORRT],
    requiredFields: []
  },
  {
    key: "Jetson_AGX_Xavier",
    value: "Jetson_AGX_Xavier",
    allowedEngine: [EngineValues.PYTORCH, EngineValues.TENSORRT],
    requiredFields: []
  },
  {
    key: "Jetson_Nano",
    value: "Jetson_Nano",
    allowedEngine: [EngineValues.PYTORCH, EngineValues.TENSORRT],
    requiredFields: []
  },
  {
    key: "Galaxy_S22",
    value: "Galaxy_S22",
    allowedEngine: [EngineValues.TFLITE],
    requiredFields: []
  },
  {
    key: "Odroid_N2",
    value: "Odroid_N2",
    allowedEngine: [EngineValues.TVM, EngineValues.ACL],
    requiredFields: []
  },
  {
    key: "Odroid_M1",
    value: "Odroid_M1",
    allowedEngine: [EngineValues.RKNN],
    requiredFields: []
  },
  {
    key: "Galaxy_S23",
    value: "Galaxy_S23",
    allowedEngine: [EngineValues.TFLITE],
    requiredFields: []
  },
  {
    key: "Rasberry_Pi5",
    value: "Rasberry_Pi5",
    allowedEngine: [EngineValues.TFLITE],
    requiredFields: []
  },
  {
    key: "Comma_3X",
    value: "Comma_3X",
    allowedEngine: [EngineValues.PYTORCH, EngineValues.ONNX],
    requiredFields: []
  },
  {
    key: "KT_cloud",
    value: "KT_cloud",
    allowedEngine: [EngineValues.PYTORCH, EngineValues.TENSORRT],
    requiredFields: ["target_hostip", "target_hostport"]
  },
  {
    key: "GCP",
    value: "GCP",
    allowedEngine: [EngineValues.PYTORCH, EngineValues.TENSORRT],
    requiredFields: ["target_hostip", "target_hostport"]
  },
  {
    key: "AWS",
    value: "AWS",
    allowedEngine: [EngineValues.PYTORCH, EngineValues.TENSORRT],
    requiredFields: ["target_hostip", "target_hostport"]
  }
];

export const CommonDatasetName = Object.freeze({
  IMAGE_NET: "imagenet",
  CHESTXRAY: "ChestXRay",
  VOC: "VOC",
  COCO: "coco"
});

export const DatasetStatus = Object.freeze({
  NONE: 1,
  DOWNLOADING: 2,
  COMPLETE: 3
});

export const ProjectStatus = Object.freeze({
  PREPARING: "preparing",
  READY: "ready",
  STARTED: "started",
  FAILED: "failed",
  RUNNING: "running",
  COMPLETED: "completed",
  STOPPED: "stopped"
});

export const ViewerMode = Object.freeze({
  TEXT: "text",
  CHART: "chart",
  MODEL_VIEW: "model_view",
  CHAT: "chat"
});

export const AutonnStatus = Object.freeze({
  PROJECT_INFO: 0,
  SYSTEM: 1,
  MODEL: 2,
  DATASET: 3
});

export const AutonnLogTitle = Object.freeze({
  [TaskType.CLASSIFICATION]: {
    train: { left: "Image", center: "Correct", right: "Accuracy", result: "Loss" },
    val: { left: "Images", center: "Correct", right: "Loss", result: "Accuracy" }
  },
  [TaskType.DETECTION]: {
    train: { left: "Box", center: "OBJECTNESS", right: "CLASS", result: "TOTAL" },
    val: { left: "Precision", center: "Recall", right: "mAP50", result: "mAP" }
  }
});

export const LearningType = Object.freeze({
  NORMAL: "normal",
  INCREMENTAL: "incremental",
  TRANSFER: "transfer",
  HPO: "HPO",
  CONTINUAL_LEARNING: "continual_learning" // Segmentation을 위한 Continual Learning Type 추가
});
