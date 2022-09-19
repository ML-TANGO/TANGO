import axios from "axios"

export function getCode(postId) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/code/" + postId)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

//시스템 정보 호출
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

export function _getPretrainedModel(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/auth/getPretrainedModel", param, {
        timeout: 300000
      })
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _getSidebarInfo(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/system/getSidebarInfo", param, {
        timeout: 300000
      })
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}
