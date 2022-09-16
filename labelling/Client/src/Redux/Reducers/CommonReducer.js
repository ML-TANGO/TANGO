import update from "immutability-helper"

import { VIEW_TOPBAR, COLLAPSE_SIDEBAR } from "./../Actions/CommonActions"

const initialState = {
  viewTopbar: true,
  collapse: false
}

export default function common(state = initialState, action) {
  switch (action.type) {
    case VIEW_TOPBAR:
      return update(state, { viewTopbar: { $set: !state.viewTopbar } })
    case COLLAPSE_SIDEBAR:
      return update(state, { collapse: { $set: action.data } })
    default:
      return state
  }
}
