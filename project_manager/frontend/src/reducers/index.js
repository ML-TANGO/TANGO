import setUserID from "./userId";

import { combineReducers } from "redux";

const rootReducer = combineReducers({
    userID : setUserID
});

export default rootReducer;


/*
import projectNameReducer from "./projectName";
import projectIdReducer from "./projectId";

const rootReducer = combineReducers({
  projectName : projectNameReducer,
  projectId : projectIdReducer,
});

export default rootReducer;
*/


