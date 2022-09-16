export const SET_CUR_IMAGE = "SET_CUR_IMAGE"
export const NEXT_IMAGE = "NEXT_IMAGE"
export const PREV_IMAGE = "PREV_IMAGE"
export const INIT_LABEL = "INIT_LABEL"
export const SET_CUR_TAG = "SET_CUR_TAG"
export const SET_BTN_STS = "SET_BTN_STS"
export const SET_IMAGELIST_LENGTH = "SET_IMAGELIST_LENGTH"
export const OPEN_TAG = "OPEN_TAG"
export const SET_IMAGE_STATUS = "SET_IMAGE_STATUS"
export const SET_PRE_STATUS = "SET_PRE_STATUS"
export const SET_OBJECTLIST = "SET_OBJECTLIST"
export const SET_BRUSHLIST = "SET_BRUSHLIST"
export const SET_TAGLIST = "SET_TAGLIST"
export const SET_BRUSHSIZE = "SET_BRUSHSIZE"
export const SET_OPACITYVALUE = "SET_OPACITYVALUE"
export const SAVE_IMAGE = "SAVE_IMAGE"
export const SET_MASK_IMG = "SET_MASK_IMG"
export const SET_MODALCHECK = "SET_MODALCHECK"
export const STATUS_MODAL = "STATUS_MODAL"
export const CHECK_MODEL = "CHECK_MODEL"
export const iSDRAW_ACTION = "ISDRAW_ACTION"
export const ISZOOMFIX = "ISZOOMFIX"
export const ISRESET_IMAGE_SIZE = "ISRESET_IMAGE_SIZE"

export function _setCurImage(index, data) {
  return {
    label: "I",
    type: SET_CUR_IMAGE,
    index,
    data
  }
}

export function _setImageListLength(data) {
  return {
    label: "I",
    type: SET_IMAGELIST_LENGTH,
    data
  }
}

export function _nextImage(data) {
  return {
    label: "I",
    type: NEXT_IMAGE,
    data
  }
}

export function _prevImage(data) {
  return {
    label: "I",
    type: PREV_IMAGE,
    data
  }
}

export function _initLabel() {
  return {
    label: "I",
    type: INIT_LABEL
  }
}

export function _setCurTag(data) {
  return {
    label: "I",
    type: SET_CUR_TAG,
    data
  }
}

export function _setBtnSts(data) {
  return {
    label: "I",
    type: SET_BTN_STS,
    data
  }
}

export function _openTag() {
  return {
    label: "I",
    type: OPEN_TAG
  }
}

export function _setImageStatus(data) {
  return {
    label: "I",
    type: SET_IMAGE_STATUS,
    data
  }
}

export function _setPreStatus(data) {
  return {
    label: "I",
    type: SET_PRE_STATUS,
    data
  }
}

export function _setObjectList(data) {
  return {
    label: "I",
    type: SET_OBJECTLIST,
    data
  }
}

export function _setBrushList(data) {
  return {
    label: "I",
    type: SET_BRUSHLIST,
    data
  }
}

export function _setTagList(data) {
  return {
    label: "I",
    type: SET_TAGLIST,
    data
  }
}

export function _setBrushSize(data) {
  return {
    label: "I",
    type: SET_BRUSHSIZE,
    data
  }
}

export function _setOpacityValue(data) {
  return {
    label: "I",
    type: SET_OPACITYVALUE,
    data
  }
}

export function _saveImage(data) {
  return {
    label: "I",
    type: SAVE_IMAGE,
    data
  }
}

export function _setMaskImg(data) {
  return {
    label: "I",
    type: SET_MASK_IMG,
    data
  }
}

export function _setModalCheck(data) {
  return {
    label: "I",
    type: SET_MODALCHECK,
    data
  }
}

export function _statusModal(data) {
  return {
    label: "I",
    type: STATUS_MODAL,
    data
  }
}

export function _checkModel(data) {
  return {
    label: "I",
    type: CHECK_MODEL,
    data
  }
}

export function _isDrawAction(data) {
  return {
    label: "I",
    type: iSDRAW_ACTION,
    data
  }
}

export function _isZoomFix(data) {
  return {
    label: "I",
    type: ISZOOMFIX,
    data
  }
}

export function _isResetImageSize(data) {
  return {
    label: "I",
    type: ISRESET_IMAGE_SIZE,
    data
  }
}
