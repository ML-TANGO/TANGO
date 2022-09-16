import update from "immutability-helper"

import {
  SET_CUR_IMAGE,
  NEXT_IMAGE,
  PREV_IMAGE,
  INIT_LABEL,
  SET_CUR_TAG,
  SET_BTN_STS,
  SET_IMAGELIST_LENGTH,
  OPEN_TAG,
  SET_IMAGE_STATUS,
  SET_BRUSHLIST,
  SET_PRE_STATUS,
  SET_OBJECTLIST,
  SET_TAGLIST,
  SET_BRUSHSIZE,
  SET_OPACITYVALUE,
  SAVE_IMAGE,
  SET_MASK_IMG,
  SET_MODALCHECK,
  STATUS_MODAL,
  CHECK_MODEL,
  iSDRAW_ACTION,
  ISZOOMFIX,
  ISRESET_IMAGE_SIZE
} from "./../Actions/ImageLabelActions"

const initialState = {
  curImage: {},
  curTag: {},
  curIndex: 0,
  imageStatus: false,
  preStatus: false,
  totalCount: 0,
  btnSts: "none",
  openTag: false,
  objectList: [],
  brushList: [],
  tagList: [],
  brushSize: 15,
  opacityValue: 50,
  saveImage: false,
  maskImg: [],
  modalCheck: false,
  statusModal: false,
  checkModel: false,
  isDrawAction: false,
  isZoomFix: false,
  isResetImageSize: false
}

export default function imageLabel(state = initialState, action) {
  if (action?.label !== "I") return state
  switch (action.type) {
    case SET_CUR_IMAGE:
      return update(state, {
        curImage: { $set: action.data },
        curIndex: { $set: action.index }
      })
    case SET_IMAGELIST_LENGTH:
      return update(state, {
        totalCount: { $set: action.data }
      })
    case NEXT_IMAGE:
      if (state.totalCount <= state.curIndex + action.data) {
        return update(state, {
          curIndex: { $set: state.totalCount - 1 },
          imageStatus: { $set: false }
        })
      } else {
        return update(state, {
          curIndex: { $set: state.curIndex + action.data }
        })
      }
    case PREV_IMAGE:
      if (state.curIndex - action.data < 0) {
        return update(state, {
          curIndex: { $set: 0 },
          imageStatus: { $set: false }
        })
      } else {
        return update(state, {
          curIndex: { $set: state.curIndex - action.data }
        })
      }
    case INIT_LABEL:
      return update(state, {
        $set: initialState
      })
    case SET_CUR_TAG:
      return update(state, {
        curTag: { $set: action.data }
      })
    case SET_BTN_STS:
      return update(state, {
        btnSts: { $set: action.data }
      })
    case OPEN_TAG:
      return update(state, {
        openTag: { $set: !state.openTag }
      })
    case SET_IMAGE_STATUS:
      return update(state, {
        imageStatus: { $set: action.data }
      })
    case SET_PRE_STATUS:
      return update(state, {
        preStatus: { $set: action.data }
      })
    case SET_OBJECTLIST:
      return update(state, {
        objectList: { $set: action.data }
      })
    case SET_BRUSHLIST:
      return update(state, {
        brushList: { $set: action.data }
      })
    case SET_TAGLIST:
      return update(state, {
        tagList: { $set: action.data }
      })
    case SET_BRUSHSIZE:
      return update(state, {
        brushSize: { $set: action.data }
      })
    case SET_OPACITYVALUE:
      return update(state, {
        opacityValue: { $set: action.data }
      })
    case SAVE_IMAGE:
      return update(state, {
        saveImage: { $set: action.data }
      })
    case SET_MASK_IMG:
      return update(state, {
        maskImg: { $set: action.data }
      })
    case SET_MODALCHECK:
      return update(state, {
        modalCheck: { $set: action.data }
      })
    case STATUS_MODAL:
      return update(state, {
        statusModal: { $set: action.data }
      })
    case CHECK_MODEL:
      return update(state, {
        checkModel: { $set: action.data }
      })
    case iSDRAW_ACTION:
      return update(state, {
        isDrawAction: { $set: action.data }
      })
    case ISZOOMFIX:
      return update(state, {
        isZoomFix: { $set: action.data }
      })
    case ISRESET_IMAGE_SIZE:
      return update(state, {
        isResetImageSize: { $set: action.data }
      })
    default:
      return state
  }
}
