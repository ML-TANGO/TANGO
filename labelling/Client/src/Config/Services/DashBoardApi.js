import axios from "axios"

// 조건에 따른 리소스 기록 가져오기
export function _getSystemInfo(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/system/getSystemInfo", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _getGpuInfo(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/system/getGpuInfo", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _getCurGpuInfo(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/system/getCurGpuInfo", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _getAiInfo(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/system/getAiInfo", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _getSourceTreeMap(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/system/getSourceTreeMap", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}
