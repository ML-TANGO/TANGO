function define(name, value) {
  Object.defineProperty(exports, name, {
    value: value,
    enumerable: true
  })
}

define("MODEL", {
  BASE_D: "/Users/upload/models/detection/yolo/yolov3_checkpoints/",
  BASE_S: "/Users/upload/models/segmentation/deeplab/"
})

define("Q", {
  DATASET: "DataSet",
  TABDATASET: "TabDataSet",
  AIPRJ: "AiProject",
  TABAIPRJ: "TabAiProject",
  IMG_ANNO: "ImageAnno",
  VDO_ANNO: "VideoAnno",
  BIN: "Binary",
  SYS: "System",
  IS: "InputSource",
  QP: "QiProject",
  HS: "HwSetting",
  RPT: "Report",
  AUTH: "Auth"
})

define("MAPPER", {
  DATASET: "DataSet",
  TABDATASET: "TabDataSet",
  AIPRJ: "AiProject",
  IMG_ANNO: "ImageAnno",
  VDO_ANNO: "VideoAnno",
  BIN: "Binary",
  SYS: "System",
  IS: "InputSource",
  QP: "QiProject",
  HS: "HwSetting",
  RPT: "Report",
  AUTH: "Auth",
  TANGO: "Tango"
})

define("BIN", {
  miniPredictor: "../Model/predictor/miniPredictor.py",
  thumbNail: "../Model/preprocessing/makeThumbnail.py",
  videoInfo: "../Model/preprocessing/getVideoInfo.py",
  trainCLF: "../Model/trainer/classificationTrain.py",
  trainDETEC: "../Model/trainer/detectionTrain.py",
  trainSEG: "../Model/trainer/segmentationTrain.py",
  trackOBJECT: "../Model/preprocessing/tracker.py",
  getHeader: "../Model/Dataset/GetHeader.py",
  runTrain: "../Model/Train/TrainManager.py",
  runService: "../Model/Service/service.py",
  runAnalysis: "../Model/Analytics/Analytics.py",
  runEvaluation: "../Model/Evaluation/Evaluation.py",
  importData: "../Model/Dataset/DatasetImporter.py"  
})

define("URI", {
  setMasterConfig: "/setMastConfig",
  miniPredictor: "/miniPredictor",
  modelLoad: "/modelLoad",
  aiTrain: "/startTrain",
  activePredictor: "/procList",
  killProcess: "/killProc",
  autoLabeling: "/autoLabeling",
  staticPredict: "/staticPredict",
  makeThumnail: "/makeThumbnail",
  videoInfo: "/getFps",
  gpuInfo: "/getGpuStatus",
  getUsableGpu: "/getUsableGpu",
  jet_setting: "/jetsonSetting",
  jet_runProc: "/startChild",
  qi_runProc: "/runRealtimePredictor"
})
