import axios from "axios"

// 라이센스 발급
/**
 * PARAM
 "BUILD": "CE",           
 "EMAIL":"dmshin@weda.kr",
 "NAME": "신동밍",
 "COMPANY": "WEDA",
 "JOB_POSITION": "연구원",
 "JOB_RANK": "선임"
 */
export function _setNewLicense(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/auth/setNewLicense", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

// 라이센스 인증
/**
 * PARAM
 "LICENSE_CODE": "sdfsdfsdfsdf",           
 */
export function _setAuthentication(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/auth/setAuthentication", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

// 라이센스 인증
/**
 * PARAM
 */
export function _getAuthCheck(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/auth/getAuthCheck", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

// 프리트레인모델 다운
/**
 * PARAM
 * MDK_KIND:
 *      detection
 *      segmentation
 *      classification
 *   값은 소문자
 */
export function _getPretrainedModel(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/auth/getPretrainedModel", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _login(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/auth/login", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _getUsers(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/auth/getusers", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _setUser(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/auth/setuser", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _checkUser(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/auth/checkuser", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _updateUser(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/auth/updateuser", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _deleteUser(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/auth/deleteuser", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}
