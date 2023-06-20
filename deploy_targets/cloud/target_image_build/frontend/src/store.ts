/**
 @license
 Copyright (c) 2015-2022 Lablup Inc. All rights reserved.
 */

import {applyMiddleware, combineReducers, compose, createStore} from 'redux';
import thunk from 'redux-thunk';
import {lazyReducerEnhancer} from 'pwa-helpers/lazy-reducer-enhancer';

import app from './reducers/app';

 declare global {
   interface Window {
     __REDUX_DEVTOOLS_EXTENSION_COMPOSE__: any;
   }
 }

// Sets up a Chrome extension for time travel debugging.
// See https://github.com/zalmoxisus/redux-devtools-extension for more information.
const devCompose = window.__REDUX_DEVTOOLS_EXTENSION_COMPOSE__ || compose;

// Initializes the Redux store with a lazyReducerEnhancer (so that you can
// lazily add reducers after the store has been created) and redux-thunk (so
// that you can dispatch async actions). See the "Redux and state management"
// section of the wiki for more details:
// https://github.com/Polymer/pwa-starter-kit/wiki/4.-Redux-and-state-management
export const store: any = createStore(
  (state) => state,
  devCompose(
    lazyReducerEnhancer(combineReducers),
    applyMiddleware(thunk))
);

// Initially loaded reducers.
store.addReducers({
  app
});
