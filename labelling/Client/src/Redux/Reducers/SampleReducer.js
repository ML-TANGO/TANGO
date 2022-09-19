import update from "immutability-helper"

import { SAMPLE } from "Redux/Actions/SampleActions"

const initialState = {
  sample: {}
}

export default function user(state = initialState, action) {
  switch (action.type) {
    case SAMPLE:
      return update(state, {
        sample: { $set: action.data }
      })
    default:
      return state
  }
}
