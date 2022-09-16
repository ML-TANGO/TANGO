import update from "immutability-helper"

import {
  SET_CUR_VIDEO,
  NEXT_VIDEO,
  PREV_VIDEO,
  INIT_LABEL,
  SET_CUR_TAG,
  SET_BTN_STS,
  SET_VIDEOLIST_LENGTH,
  OPEN_TAG,
  SET_VIDEO_STATUS,
  SET_PRE_STATUS,
  SET_OBJECTLIST,
  SET_BRUSHLIST,
  SET_LOAD_STATUS,
  SET_TAGLIST,
  SAVE_VIDEO,
  SET_MODALCHECK,
  SET_CURFRAME,
  SET_FRAMEBOUND,
  STATUS_MODAL,
  CHECK_MODEL,
  iSDRAW_ACTION
} from "../Actions/VideoLabelActions"

const initialState = {
  curVideo: [],
  curTag: {},
  curIndex: 0,
  videoStatus: false,
  preStatus: false,
  totalCount: 0,
  btnSts: "none",
  openTag: false,
  objectList: [],
  brushList: [],
  tagList: [],
  isLoaded: false,
  saveVideo: false,
  modalCheck: false,
  curFrame: 0,
  frameBound: 0,
  statusModal: false,
  checkModel: false,
  isDrawAction: false
}

export default function videoLabel(state = initialState, action) {
  if (action?.label !== "V") return state
  switch (action.type) {
    case SET_LOAD_STATUS:
      return update(state, {
        isLoaded: { $set: action.data }
      })
    case SET_CUR_VIDEO:
      return update(state, {
        curVideo: { $set: action.data },
        curIndex: { $set: action.index }
      })
    case SET_VIDEOLIST_LENGTH:
      return update(state, {
        totalCount: { $set: action.data }
      })
    case NEXT_VIDEO:
      if (state.totalCount <= state.curIndex + action.data) {
        return update(state, {
          curIndex: { $set: state.totalCount - 1 },
          videoStatus: { $set: false }
        })
      } else {
        return update(state, {
          curIndex: { $set: state.curIndex + action.data }
        })
      }
    case PREV_VIDEO:
      if (state.curIndex - action.data < 0) {
        return update(state, {
          curIndex: { $set: 0 },
          videoStatus: { $set: false }
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
    case SET_VIDEO_STATUS:
      return update(state, {
        videoStatus: { $set: action.data }
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
    case SAVE_VIDEO:
      return update(state, {
        saveVideo: { $set: action.data }
      })
    case SET_MODALCHECK:
      return update(state, {
        modalCheck: { $set: action.data }
      })
    case SET_CURFRAME:
      return update(state, {
        curFrame: { $set: action.data }
      })
    case SET_FRAMEBOUND:
      return update(state, {
        frameBound: { $set: action.data }
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
    default:
      return state
  }
}
