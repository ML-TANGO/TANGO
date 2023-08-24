import axios from "axios"

export function _getDataSetList(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/dataset/getDataSetList", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _getDataSetImportList(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/dataset/getDataSetImportList", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _removeTempFiles(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/dataset/removeTempFiles", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

//dataset 파일 업로드
export function _setFileupload(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/dataset/upload", param, {
        headers: {
          "Content-Type": "multipart/form-data"
        }
      })
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _setNewDataSets(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/dataset/setNewDataSets", param, { timeout: 0 })
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
      .post("/api/dataset/setUpdateDataset", param)
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
      .post("/api/dataset/getFileList", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _removeDataSet(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/dataset/removeDataSet", param)
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
      .post("/api/dataset/setDupDataset", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _autoLabeling(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/dataset/autolabeling", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _getBaseModel(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/dataset/getBaseModel", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _getModelEpochs(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/dataset/getModelEpochs", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _getTagInfo(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/dataset/getTagInfo", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _checkDirExist(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/dataset/checkDirExist", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _deployDataSet(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/dataset/deployDataSet", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _removeDeployedDataSet(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/dataset/removeDeployedDataSet", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}
