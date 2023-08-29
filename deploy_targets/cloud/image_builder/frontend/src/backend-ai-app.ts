/**
 @license
 Copyright (c) 2015-2022 Lablup Inc. All rights reserved.
 */
export const UPDATE_PAGE = 'UPDATE_PAGE';
export const UPDATE_OFFLINE = 'UPDATE_OFFLINE';
export const UPDATE_DRAWER_STATE = 'UPDATE_DRAWER_STATE';
export const OPEN_SNACKBAR = 'OPEN_SNACKBAR';
export const CLOSE_SNACKBAR = 'CLOSE_SNACKBAR';

export const navigate = (path: any, params: Record<string, unknown> = {}) => (dispatch: any) => {
  // Extract the page name from path.
  if (['/login', '/summary', '/builder', '/environments', '/logs'].includes(path) !== true) { // Fallback for Electron Shell/Windows OS
    const fragments = path.split(/[/]+/);
    if (fragments.length > 1 && fragments[0] === '') {
      path = fragments[1];
      params['requestURL'] = fragments.slice(2).join('/');
    }
  }
  params['queryString'] = window.location.search;
  if (path === 'index.html' || path === '') {
    path = '/';
  }
  let page;
  if (['/', 'build', '/build', 'app', '/app'].includes(path)) {
    page = 'summary';
  } else if (path[0] === '/') {
    page = path.slice(1);
  } else {
    page = path;
  }

  dispatch(loadPage(page, params));

  // Close the drawer - in case the *path* change came from a link in the drawer.
  dispatch(updateDrawerState(false));
};

const loadPage = (page: string, params: Record<string, unknown> = {}) => (dispatch: any) => {
  switch (page) {
  case 'login':
    import('./components/forklift-login-view');
    break;
  case 'summary':
    import('./components/forklift-summary-view');
    break;
  case 'builder':
    import('./components/forklift-build-view.js');
    break;
  case 'environments':
    import('./components/forklift-environment-view');
    break;
  case 'logs':
    import('./components/forklift-tasks-view');
    break;
  case 'error':
  default:
    import('./components/forklift-error-view').then((module) => {
      // TODO: after page changing?
      return;
    });
    break;
  }
  dispatch(updatePage(page, params));
};

export const updatePage = (page: any, params: any) => {
  return {
    type: UPDATE_PAGE,
    page,
    params
  };
};

let offlineTimer;

export const showOffline = () => (dispatch) => {
  dispatch({
    type: OPEN_SNACKBAR
  });
  window.clearTimeout(offlineTimer);
  offlineTimer = window.setTimeout(() =>
    dispatch({type: CLOSE_SNACKBAR}), 3000);
};

export const updateOffline = (offline) => (dispatch, getState) => {
  // Show the snackbar only if offline status changes.
  if (offline !== getState().app.offline) {
    dispatch(showOffline());
  }
  dispatch({
    type: UPDATE_OFFLINE,
    offline
  });
};

export const updateDrawerState = (opened: boolean) => {
  return {
    type: UPDATE_DRAWER_STATE,
    opened
  };
};
