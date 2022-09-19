import axios from "axios"

export function _createDataset(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/tab/dataset/createdataset", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _getFiledata(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/tab/dataset/getfiledata", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _setUpdateDataset(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/tab/dataset/setUpdateDataset", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _setDupDataset(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/tab/dataset/setdupdataset", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _getfeatures(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/tab/dataset/getfeatures", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _getFileList(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/tab/dataset/getFileList", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _getAnalysis(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/tab/dataset/getAnalysis", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

// DB 접속 확인 및 쿼리 API
//   "IS_TEST" : true // 트루일 경우 테스트 커넥션만 동작 False일 경우 Query 실행 결ㄹ과
//   "CLIENT": "oracledb",   //mysql, mysql2, pg
//   "ADDRESS": "106.251.247.178",
//   "PORT": 9154,
//   "DBNAME": "XE",
//   "USER": "CTMSPLUS",
//   "PASSWORD": "HKTIRE_CTMS"
//   "QUERY": "SELECT ...."
//   (OPTIONAL)  LIMIT : 제한 갯수 설정 없으면 10개
//   (OPTIONAL)  TIMEOUT: 제한시간 설정

export function _getDbConnectionInfo(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/tab/dataset/dbconnection", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

// param
// DATASET_CD
export function _getDBInfo(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/tab/dataset/getdbinfo", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}
