export const TaskType = Object.freeze({
  DETECTION: "detection",
  CLASSIFICATION: "classification"
});

export const ContainerPort = Object.freeze({
  bms: "8081",
  vis2code: "8091",
  autonn_bb: "8087",
  autonn_nk: "8089",
  code_gen: "8888",
  cloud_deployment: "8088",
  ondevice_deployment: "8891",
  yolo_e: "8090",
  lablilng: "8095"
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

export const ContainerName = {
  BMS: "bms",
  AUTO_NN: "yoloe",
  AUTO_NN_RESNET: "autonn-resnet",
  CODE_GEN: "codeGen",
  IMAGE_DEPLOY: "imagedeploy",
  USER_EDITING: "user_edit",
  VISUALIZATION: "viz2code"
};

export const DisplayName = {
  [ContainerName.BMS]: "BMS",
  [ContainerName.AUTO_NN]: "Auto NN",
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
  TFLITE: "tflite"
};

export const EngineLabel = {
  [EngineValues.ACL]: "ACL",
  [EngineValues.PYTORCH]: "Pytorch",
  [EngineValues.TVM]: "TVM",
  [EngineValues.TENSORRT]: "Tensorrt",
  [EngineValues.TFLITE]: "TFLite"
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
    allowedEngine: [EngineValues.TENSORRT],
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
  }
];
