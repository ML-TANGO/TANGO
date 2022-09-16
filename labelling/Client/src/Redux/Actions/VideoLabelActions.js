export const SET_CUR_VIDEO = "SET_CUR_VIDEO"
export const NEXT_VIDEO = "NEXT_VIDEO"
export const PREV_VIDEO = "PREV_VIDEO"
export const INIT_LABEL = "INIT_LABEL"
export const SET_CUR_TAG = "SET_CUR_TAG"
export const SET_BTN_STS = "SET_BTN_STS"
export const SET_VIDEOLIST_LENGTH = "SET_VIDEOLIST_LENGTH"
export const OPEN_TAG = "OPEN_TAG"
export const SET_VIDEO_STATUS = "SET_VIDEO_STATUS"
export const SET_PRE_STATUS = "SET_PRE_STATUS"
export const SET_OBJECTLIST = "SET_OBJECTLIST"
export const SET_BRUSHLIST = "SET_BRUSHLIST"
export const SET_TAGLIST = "SET_TAGLIST"
export const SAVE_VIDEO = "SAVE_VIDEO"
export const SET_LOAD_STATUS = "SET_LOAD_STATUS"
export const SET_MODALCHECK = "SET_MODALCHECK"
export const SET_CURFRAME = "SET_CURFRAME"
export const SET_FRAMEBOUND = "SET_FRAMEBOUND"
export const STATUS_MODAL = "STATUS_MODAL"
export const CHECK_MODEL = "CHECK_MODEL"
export const iSDRAW_ACTION = "ISDRAW_ACTION"

export function _setLoadStatus(data) {
  return {
    label: "V",
    type: SET_LOAD_STATUS,
    data
  }
}

export function _setCurVideo(index, data) {
  return {
    label: "V",
    type: SET_CUR_VIDEO,
    index,
    data
  }
}

export function _setVideoListLength(data) {
  return {
    label: "V",
    type: SET_VIDEOLIST_LENGTH,
    data
  }
}

export function _nextVideo(data) {
  return {
    label: "V",
    type: NEXT_VIDEO,
    data
  }
}

export function _prevVideo(data) {
  return {
    label: "V",
    type: PREV_VIDEO,
    data
  }
}

export function _initVideoLabel() {
  return {
    label: "V",
    type: INIT_LABEL
  }
}

export function _setCurTag(data) {
  return {
    label: "V",
    type: SET_CUR_TAG,
    data
  }
}

export function _setBtnSts(data) {
  return {
    label: "V",
    type: SET_BTN_STS,
    data
  }
}

export function _openTag() {
  return {
    label: "V",
    type: OPEN_TAG
  }
}

export function _setVideoStatus(data) {
  return {
    label: "V",
    type: SET_VIDEO_STATUS,
    data
  }
}

export function _setPreStatus(data) {
  return {
    label: "V",
    type: SET_PRE_STATUS,
    data
  }
}

export function _setObjectList(data) {
  return {
    label: "V",
    type: SET_OBJECTLIST,
    data
  }
}

export function _setBrushList(data) {
  return {
    label: "V",
    type: SET_BRUSHLIST,
    data
  }
}

export function _setTagList(data) {
  return {
    label: "V",
    type: SET_TAGLIST,
    data
  }
}

export function _saveVideo(data) {
  return {
    label: "V",
    type: SAVE_VIDEO,
    data
  }
}

export function _setModalCheck(data) {
  return {
    label: "V",
    type: SET_MODALCHECK,
    data
  }
}

export function _setCurFrame(data) {
  return {
    label: "V",
    type: SET_CURFRAME,
    data
  }
}

export function _setFrameBound(data) {
  return {
    label: "V",
    type: SET_FRAMEBOUND,
    data
  }
}

export function _statusModal(data) {
  return {
    label: "V",
    type: STATUS_MODAL,
    data
  }
}

export function _checkModel(data) {
  return {
    label: "V",
    type: CHECK_MODEL,
    data
  }
}

export function _isDrawAction(data) {
  return {
    label: "V",
    type: iSDRAW_ACTION,
    data
  }
}
