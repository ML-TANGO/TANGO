import axios from "axios"

// AI 프로젝트 리스트 전체 호출
export function _getAiPrjList(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/aiproject/getAiPrjList", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

// AI 프로젝트 생성
export function _setAiPrj(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/aiproject/setAiPrj", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

// AI 프로젝트 설정 변경
export function _updateAiPrjDataSet(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/aiproject/updateAiPrjDataSet", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

// AI 프로젝트에 등록된 트레인 데이터셋 리스트 호출
export function _getAiSetUpData(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/aiproject/getAiSetUpData", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

// AI에 사용되는 클래스 정보 설정
export function _setAiClassInfo(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/aiproject/setAiClassInfo", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

// AI 학습 시작
export function _setTrainAi(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/aiproject/startTrainAi", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

// AI 학습 종료
export function _setStopTrainAi(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/aiproject/stopTrainAi", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

// DATASet_CD를 던져 TAG 정보들을 호출
export function _getTagListByDatasetCD(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/aiproject/getTagListByDatasetCD", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

// AI 정보창에서 AI가 포함하고 있는 데이터셋 리스트를 호출
// param: AI_CD
export function _getDataSetListByAICD(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/aiproject/getDataSetListByAICD", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

// AI 정보창에서 AI가 포함하고 있는 모델 정보를 호출
// param: AI_CD
export function _getModelDataByAICD(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/aiproject/getModelDataByAICD", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

// AI 정보창에서 AI가 포함하고 있는 클래스들의 리스트를 호출
// param: AI_CD
export function _getClassListByAICD(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/aiproject/getClassListByAICD", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

// AI 학습 결과 조회
export function _getTrainResult(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/aiproject/getTrainResult", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

// AI 테스트 Predict 조회
//AI_CD, DATASET_CD, EPOCH_NO, OBJECT_TYPE, {DATA_CD}
//DATA_CD까지 넘기면 한개, 안넘기면 전체
export function _getTestAi(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/aiproject/getTestAi", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _removeAiModel(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/aiproject/removeAiModel", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export async function _getDownloadAi(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/aiproject/getDownloadAi", param, {
        timeout: 300000,
        responseType: "arraybuffer"
      })
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

//전문가 모드에 뿌려질 코드리스트
export function _getExpertCode(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/aiproject/getExpertCode", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

//전문가 모드에 뿌려질 바닐라 모델 리스트
export function _getExpertBaseModels(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/aiproject/getExpertBaseModels", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

//전문가 모드에 뿌려질 내가만든 모델 리스트
export function _getExpertMyModels(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/aiproject/getExpertMyModels", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

//AI 정보 변경 이름/ 설명
export function _updateAiInfo(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/aiproject/updateAiInfo", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

//활성화된 AI 정지
//param {PID: ddd}
export function _stopActiveModel(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/aiproject/stopActiveModel", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

//AI 소켓 정보
export function _getTrainingInfo(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/aiproject/getTrainingInfo", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

//GPU 사용 정보
export function _getUsableGpu(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/aiproject/getUsableGpu", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

//현재 학습중인 AI 개수 전달
export function _getActiveTrain(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/aiproject/getActiveTrain", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

// 모델 리스트 가져오기 (베스트모델 선택 위해)
// param
// AI_CD, DATA_TYPE
export function _getTrainedModelList(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/aiproject/gettrainedmodel", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

// 모델 리스트 가져오기 (베스트모델 선택 위해)
// param
// AI_CD
// DATA_TYPE
// OBJECT_TYPE
// EPOCH
// MDL_IDX
export function _setTrainedModelList(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/aiproject/settrainedmodel", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

// 모델 리스트 가져오기 (베스트모델)
// param
// AI_CD
// DATA_TYPE
// OBJECT_TYPE
// EPOCH
// MDL_IDX
export function _getSelectedModelList(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/aiproject/getselectedModel", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

// Deploy Model Delete
export function _removeSelectedModel(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/aiproject/removeSelectedModel", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

//param AI_CD
//학습중인 모델 정보 조회
export function _getTrainModelInfo(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/aiproject/gettrainmodelinfo", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}
