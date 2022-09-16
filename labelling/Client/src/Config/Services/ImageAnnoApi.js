import axios from "axios"

export function _getImageList(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/imageanno/getImageList", param)
      .then(response => {
        resolve(response.data)
      })
      .catch(error => {
        reject(error.response)
      })
  })
}

export function _getImage(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/imageanno/getImage", param)
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
      .post("/api/imageanno/setDataTag", param)
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
      .post("/api/imageanno/getDataTags", param)
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
      .post("/api/imageanno/removeDataTag", param)
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
      .post("/api/imageanno/getCategory", param)
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
export function _getImagePredict(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/imageanno/getImagePredict", param)
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
export function _setImageAnnotation(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/imageanno/setImageAnnotation", param)
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
      TAG_CD,
      COLOR
      NAME
      DESC_TXT
      있으면 {CLASS_CD}
}
*/
export function _updateDataTag(param) {
  return new Promise((resolve, reject) => {
    axios
      .post("/api/imageanno/updateDataTag", param)
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
