import axios from "axios"

export function _getVideoList(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/videoanno/getVideoList", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _getVideo(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/videoanno/getVideo", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _setDataTag(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/videoanno/setDataTag", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _getDataTags(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/videoanno/getDataTags", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _removeDataTag(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/videoanno/removeDataTag", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _getCategory(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/videoanno/getCategory", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

/*
PARAM: {
      "DATASET_CD" : "CI200021",
      "DATA_CD" : "00000000",
      "CLASS_CD" : 247, // TAG 정보 안에 있음
      "OBJECT_TYPE" : "D"
}
*/
export function _getVideoPredict(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/videoanno/getImagePredict", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

/*
PARAM: {
      "DATASET_CD" : "CI200021",
      "DATA_CD" : "00000000",
      "ANNO_DATA" [그 배열]
}
*/
export function _setVideoAnnotation(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/videoanno/setVideoAnnotation", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _getTrackedObject(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/videoanno/getTrackedObject", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _getTrackResult(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/videoanno/getTrackResult", param, { timeout: 300000 })
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _getPredictResult(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/videoanno/getPredictResult", param, { timeout: 300000 })
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _getActivePredictor(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/imageanno/getActivePredictor", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}
