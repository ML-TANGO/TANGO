/**
 @license
 Copyright (c) 2015-2022 Lablup Inc. All rights reserved.
 */

import {UPDATE_PAGE} from '../backend-ai-app.js';

const INITIAL_STATE = {
  page: '',
  params: {},
};

const app = (state = INITIAL_STATE, action: any) => {
  switch (action.type) {
  case UPDATE_PAGE:
    return {
      ...state,
      page: action.page,
      params: action.params
    };
  default:
    return state;
  }
};

export default app;
