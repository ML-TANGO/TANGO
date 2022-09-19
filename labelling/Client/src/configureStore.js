import { combineReducers, createStore, applyMiddleware, compose } from "redux"
import thunk from "redux-thunk"

import { ImageLabelReducer, VideoLabelReducer, CommonReducer } from "./Redux/Reducers/index"

const configureStore = () => {
  const reducer = combineReducers({
    imageLabel: ImageLabelReducer,
    videoLabel: VideoLabelReducer,
    common: CommonReducer
  })

  const middleware = [thunk]
  let composeEnhancers
  if (process.env.NODE_ENV !== "production") {
    composeEnhancers = window.__REDUX_DEVTOOLS_EXTENSION_COMPOSE__ || compose
    //composeEnhancers = compose
  } else {
    composeEnhancers = compose
  }

  // eslint-disable-next-line no-underscore-dangle
  return createStore(reducer, composeEnhancers(applyMiddleware(...middleware)))
}

export default configureStore
