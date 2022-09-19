export const VIEW_TOPBAR = "VIEW_TOPBAR"
export const COLLAPSE_SIDEBAR = "COLLAPSE_SIDEBAR"

export function _viewTopBar() {
  return {
    type: VIEW_TOPBAR
  }
}

export function _collapseSidebar(data) {
  return {
    type: COLLAPSE_SIDEBAR,
    data
  }
}
