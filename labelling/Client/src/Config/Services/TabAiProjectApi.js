import axios from "axios"

export function _getmodellist(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/tab/aiproject/getmodellist", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _createAiProject(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/tab/aiproject/createaiproject", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

//ai 1개의 상세 정보 조회
export function _getAiProjectData(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/tab/aiproject/getaiprojectdata", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

//param AI_CD
//      MDL_IDX_LIST: [{MDL_IDX: 0 }, {MDL_IDX: 1 }...]
//      "GRAPH_TYPE": "Distribution_Plot",
export function _getModelResult(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/tab/aiproject/getmodelresult", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _startTrain(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/tab/aiproject/starttrain", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

//param AI_CD
export function _getModelSummary(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/tab/aiproject/getmodelsummary", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

//param AI_CD
export function _stopTrain(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/tab/aiproject/stoptrain", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

//param AI_CD
export function _updateAiProject(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/tab/aiproject/updateaiproject", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}
