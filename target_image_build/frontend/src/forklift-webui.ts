import {LitElement, html, CSSResultGroup} from 'lit';
import {customElement, property} from 'lit/decorators.js';

import {ForkliftStyles} from './components/forklift-styles';
import {
  IronFlex,
  IronFlexAlignment,
  IronFlexFactors,
  IronPositioning
} from './plastics/layout/iron-flex-layout-classes';

// PWA components
import {connect} from 'pwa-helpers/connect-mixin';
import {installOfflineWatcher} from 'pwa-helpers/network';
import {installRouter} from 'pwa-helpers/router';
import {store} from './store';

import {navigate, updateOffline} from './backend-ai-app';
import {ForkliftUtils} from './components/forklift-utils';

import './plastics/mwc/mwc-top-app-bar-fixed';
import {TopAppBarFixed} from './plastics/mwc/mwc-top-app-bar-fixed';
import '@material/mwc-icon-button/mwc-icon-button';
import '@material/mwc-menu/mwc-menu';
import '@material/mwc-list/mwc-list';
import '@material/mwc-list/mwc-list-item';
import '@material/mwc-circular-progress/mwc-circular-progress';
import {Drawer} from '@material/mwc-drawer';

import 'weightless/popover';
import 'weightless/popover-card';

import './components/backend-ai-offline-indicator';
import './components/backend-ai-sidepanel-notification';
import './components/backend-ai-sidepanel-task';
import './components/forklift-login-view';
import './components/forklift-summary-view';
import './components/forklift-build-view';
import './components/forklift-environment-view';
import './components/forklift-tasks-view';
import './components/forklift-error-view';
import './components/forklift-notification';
import BackendAiSettingsStore from './components/backend-ai-settings-store';
import BackendAiTasker from './components/backend-ai-tasker';

globalThis.backendaioptions = new BackendAiSettingsStore;
globalThis.tasker = new BackendAiTasker;

@customElement('forklift-webui')
export default class ForkliftWebUI extends connect(store)(LitElement) {
  @property({type: String}) menuTitle = '';
  @property({type: String}) _page = '';
  @property({type: String}) _lazyPage = '';
  @property({type: String}) _sidepanel = '';
  @property({type: String}) user_id = 'DISCONNECTED';
  @property({type: String}) full_name = 'DISCONNECTED';
  @property({type: Boolean}) _drawerOpened = false;
  @property({type: Boolean}) _offlineIndicatorOpened = false;
  @property({type: Boolean}) _offline = false;
  @property({type: Boolean}) mini_ui = false;
  @property({type: Boolean}) auto_logout = false;
  @property({type: Boolean}) is_connected = false;
  @property({type: Boolean}) isUserInfoMaskEnabled;
  @property({type: Object}) _pageParams = {};
  @property({type: Object}) config = Object();
  @property({type: Object}) loginPanel = Object();
  @property({type: Object}) appPage;
  @property({type: Object}) appBody;
  @property({type: Object}) contentBody;
  @property({type: Object}) mainToolbar;
  @property({type: Object}) sidebarMenu;
  @property({type: Number}) minibarWidth = 88;
  @property({type: Number}) sidebarWidth = 250;
  @property({type: Number}) sidepanelWidth = 250;
  @property({type: Array}) supportLanguageCodes = ['en', 'ko'];
  @property({type: Array}) availablePages = ['login', 'summary', 'builder', 'environments', 'tasks'];


  static get styles(): CSSResultGroup {
    return [
      ForkliftStyles,
      IronFlex,
      IronFlexAlignment,
      IronFlexFactors,
      IronPositioning
    ];
  }

  constructor() {
    super();
  }

  firstUpdated() {
    globalThis.lablupNotification = this.shadowRoot?.querySelector('#notification');
    globalThis.currentPage = this._page;
    globalThis.currentPageParams = this._pageParams;
    this.appBody = this.shadowRoot?.querySelector('#app-body');
    this.appPage = this.shadowRoot?.querySelector('#app-page');
    this.contentBody = this.shadowRoot?.querySelector('#content-body');
    this.contentBody.type = 'dismissible';
    this.mainToolbar = this.shadowRoot?.querySelector('#main-toolbar');
    this.sidebarMenu = this.shadowRoot?.querySelector('#sidebar-menu');
    this.user_id = this.full_name = globalThis.backendaioptions.get('email') || 'DISCONNECTED';
    installRouter((location) => store.dispatch(navigate(decodeURIComponent(location.pathname))));
    installOfflineWatcher((offline) => store.dispatch(updateOffline(offline)));
    this._changeDrawerLayout(document.body.clientWidth, document.body.clientHeight);
    const configPath = '../configs/forklift.toml';
    document.addEventListener('backend-ai-logout', () => this.logout());
    document.addEventListener('show-menu', () => this.showMenu());
    globalThis.addEventListener('beforeunload', () => {
      globalThis.backendaioptions.set('last_window_close_time', new Date().getTime() / 1000);
    });
    ForkliftUtils._parseConfig(configPath).then((config) => {
      this.config = config;
      this.loadConfig(this.config);
      // If disconnected
      if (!ForkliftUtils._checkLogin()) {
        this.refreshPage();
        ForkliftUtils._moveTo('/login');
      }
      if (this._page === 'login') {
        this.hideMenu();
      }
    });
    this.addToolTips();
    globalThis.addEventListener('resize', () => {
      this._changeDrawerLayout(document.body.clientWidth, document.body.clientHeight);
      // Tricks to close expansion if window size changes
      document.body.dispatchEvent(new Event('click'));
    });
  }

  async connectedCallback() {
    super.connectedCallback();
    document.addEventListener('backend-ai-connected', () => this.refreshPage());
  }

  disconnectedCallback() {
    document.removeEventListener('backend-ai-connected', () => this.refreshPage());
    super.disconnectedCallback();
  }

  /**
   * Refresh the user information panel.
   */
  _refreshUserInfoPanel(): void {
    this.user_id = this.full_name = globalThis.backendaioptions.get('email') || 'DISCONNECTED';
  }

  refreshPage() {
    this._refreshUserInfoPanel();
    document.body.style.backgroundImage = 'none';
    this.appBody.style.visibility = 'visible';
    const curtain = this.shadowRoot?.getElementById('loading-curtain');
    curtain?.classList.add('visuallyhidden');
    curtain?.addEventListener('transitionend', () => {
      curtain?.classList.add('hidden');
      this.is_connected = true;
    }, {
      capture: false,
      once: true,
      passive: false
    });
  }

  loadConfig(config): void {
    if ((typeof config.general !== 'undefined' && 'maskUserInfo' in config.general)) {
      this.isUserInfoMaskEnabled = config.general.maskUserInfo;
    }
    const refreshWithConfig = new CustomEvent('refresh-login-panel-with-config', {detail: {config: config}});
    document.dispatchEvent(refreshWithConfig);
  }

  updated(changedProps: any) {
    if (changedProps.has('_page')) {
      let view: string = this._page;
      // load data for view
      if (this.availablePages.includes(view) !== true) { // Fallback for Windows OS
        const modified_view: (string | undefined) = view.split(/[/]+/).pop();
        if (typeof modified_view != 'undefined') {
          view = modified_view;
        }
      }
      this._page = view;
      this._updateSidebar(view);
    }
  }

  /**
   * Update the sidebar menu title according to view.
   *
   * @param {string} view - Sidebar menu title string.
   */
  _updateSidebar(view) {
    switch (view) {
    case 'login':
      this.menuTitle = 'Login';
      break;
    case 'summary':
      this.menuTitle = 'Summary';
      break;
    case 'builder':
      this.menuTitle = 'Builder';
      break;
    case 'environments':
      this.menuTitle = 'Environments';
      break;
    case 'tasks':
      this.menuTitle = 'Tasks';
      break;
    default:
      if (this._page !== 'error') {
        this._lazyPage = this._page;
      }
      this._page = 'error';
      this.menuTitle = 'NOTFOUND';
    }
  }

  /**
   * Change the state.
   *
   * @param {object} state
   */
  stateChanged(state: any) {
    this._page = state.app.page;
    this._pageParams = state.app.params;
    this._offline = state.app.offline;
    this._offlineIndicatorOpened = state.app.offlineIndicatorOpened;
    this._drawerOpened = state.app.drawerOpened;
    globalThis.currentPage = this._page;
    globalThis.currentPageParams = this._pageParams;
  }

  /**
   * Create a popover.
   *
   * @param {string} anchor
   * @param {string} title
   */
  _createPopover(anchor: string, title: string) {
    const popover = document.createElement('wl-popover');
    popover.anchor = anchor;
    popover.setAttribute('fixed', '');
    popover.setAttribute('disablefocustrap', '');
    popover.setAttribute('anchororiginx', 'right');
    popover.setAttribute('anchororiginy', 'center');
    popover.setAttribute('transformoriginx', 'left');
    popover.setAttribute('transformoriginy', 'center');
    popover.anchorOpenEvents = ['mouseover'];
    popover.anchorCloseEvents = ['mouseout'];
    const card = document.createElement('wl-popover-card');
    const carddiv = document.createElement('div');
    carddiv.style.padding = '5px';
    carddiv.innerText = title;
    card.appendChild(carddiv);
    popover.appendChild(card);
    const tooltipBox = this.shadowRoot?.querySelector('#mini-tooltips');
    tooltipBox?.appendChild(popover);
  }

  /**
   * Add tool tips by create popovers.
   */
  async addToolTips() {
    this._createPopover('#summary-menu-icon', 'Summary');
    this._createPopover('#builder-menu-icon', 'Builder');
    this._createPopover('#tasks-menu-icon', 'Tasks');
    this._createPopover('#environments-menu-icon', 'Environments');
  }

  /**
   * Check Fullname exists, and if not then use user_id instead.
   *
   * @return {string} Name from full name or user ID
   */
  _getUsername() {
    let name = this.full_name ? this.full_name : this.user_id;
    // mask username only when the configuration is enabled
    if (this.isUserInfoMaskEnabled) {
      const maskStartIdx = 2;
      const emailPattern = /^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*$/;
      const isEmail: boolean = emailPattern.test(name);
      const maskLength = isEmail ? name.split('@')[0].length - maskStartIdx : name.length - maskStartIdx;
      name = ForkliftUtils._maskString(name, '*', maskStartIdx, maskLength);
    }
    return name;
  }

  /**
   *  Get user id according to configuration
   *
   *  @return {string} userId
   */
  _getUserId() {
    let userId = this.user_id;
    // mask user id(email) only when the configuration is enabled
    if (this.isUserInfoMaskEnabled) {
      const maskStartIdx = 2;
      const maskLength = userId.split('@')[0].length - maskStartIdx;
      userId = ForkliftUtils._maskString(userId, '*', maskStartIdx, maskLength);
    }
    return userId;
  }

  /**
   * Set the drawer width by browser size.
   */
  toggleSidePanelUI(): void {
    if (this.contentBody.open) {
      this.contentBody.open = false;
      if (this.mini_ui) {
        this.mainToolbar.style.setProperty('--mdc-drawer-width', this.minibarWidth + 'px');// 54
      } else {
        this.mainToolbar.style.setProperty('--mdc-drawer-width', this.sidebarWidth + 'px');// 250
      }
    } else {
      this.contentBody.open = true;
      this.contentBody.style.setProperty('--mdc-drawer-width', this.sidepanelWidth + 'px');
      if (this.mini_ui) {
        this.mainToolbar.style.setProperty('--mdc-drawer-width', this.minibarWidth + this.sidepanelWidth + 'px');// 54+250
      } else {
        this.mainToolbar.style.setProperty('--mdc-drawer-width', this.sidebarWidth + this.sidepanelWidth + 'px');// 250+250
      }
    }
  }

  /**
   * Change the drawer layout according to browser size.
   *
   * @param {number} width
   * @param {number} height
   * @param {boolean} applyMiniui
   */
  _changeDrawerLayout(width: number, height: number, applyMiniui = false): void {
    this.mainToolbar.style.setProperty('--mdc-drawer-width', '0px');
    if (width < 700 && !applyMiniui) { // Close drawer
      this.appBody.style.setProperty('--mdc-drawer-width', this.sidebarWidth + 'px');
      this.appBody.type = 'modal';
      this.appBody.open = false;
      // this.contentBody.style.width = 'calc('+width+'px - 190px)';
      this.mainToolbar.style.setProperty('--mdc-drawer-width', '0px');
      if (this.mini_ui) {
        this.mini_ui = false;
        globalThis.mini_ui = this.mini_ui;
      }
      /* close opened sidepanel immediately */
      if (this.contentBody.open) {
        this.contentBody.open = false;
      }
    } else { // Open drawer
      if (this.mini_ui) {
        this.appBody.style.setProperty('--mdc-drawer-width', this.minibarWidth + 'px');
        this.mainToolbar.style.setProperty('--mdc-drawer-width', this.minibarWidth + 'px');
        this.contentBody.style.width = (width - this.minibarWidth) + 'px';
        if (this.contentBody.open) {
          this.mainToolbar.style.setProperty('--mdc-drawer-width', this.minibarWidth + this.sidebarWidth + 'px');// 54+250
        }
      } else {
        this.appBody.style.setProperty('--mdc-drawer-width', this.sidebarWidth + 'px');
        this.mainToolbar.style.setProperty('--mdc-drawer-width', this.sidebarWidth + 'px');
        this.contentBody.style.width = (width - this.sidebarWidth) + 'px';
        if (this.contentBody.open) {
          this.mainToolbar.style.setProperty('--mdc-drawer-width', this.sidebarWidth + this.sidepanelWidth + 'px'); // 250+250
        }
      }
      this.appBody.type = 'dismissible';
      this.appBody.open = true;
    }
    if (this.contentBody.open) {
      this.contentBody.style.setProperty('--mdc-drawer-width', this.sidepanelWidth + 'px');
    }
  }

  /**
   * Display the toggle sidebar when this.mini_ui is true.
   */
  toggleSidebarUI(): void {
    if (this.contentBody.open === true) {
      this._sidepanel = '';
      this.toggleSidePanelUI();
    }
    if (!this.mini_ui) {
      this.mini_ui = true;
    } else {
      this.mini_ui = false;
    }
    globalThis.mini_ui = this.mini_ui;
    const event: CustomEvent = new CustomEvent('backend-ai-ui-changed', {'detail': {'mini-ui': this.mini_ui}});
    document.dispatchEvent(event);
    this._changeDrawerLayout(document.body.clientWidth, document.body.clientHeight);
  }

  /**
   * Control the mwc-drawer.
   */
  toggleDrawer() {
    const drawer = this.shadowRoot?.querySelector<Drawer>('mwc-drawer')!;
    if (drawer?.open === true) {
      drawer.open = false;
    } else {
      drawer.open = true;
    }
  }

  hideMenu() {
    (this.shadowRoot?.querySelector('.drawer-menu') as HTMLDivElement).style.visibility = 'hidden';
    (this.shadowRoot?.querySelector('#main-toolbar') as TopAppBarFixed).style.visibility = 'hidden';
  }

  showMenu() {
    (this.shadowRoot?.querySelector('.drawer-menu') as HTMLDivElement).style.visibility = 'visible';
    (this.shadowRoot?.querySelector('#main-toolbar') as TopAppBarFixed).style.visibility = 'visible';
  }

  /**
   * Control the side panel by panel's state.
   *
   * @param {string} panel
   */
  _openSidePanel(panel: string): void {
    if (document.body.clientWidth < 750) {
      this.mini_ui = true;
      this._changeDrawerLayout(document.body.clientWidth, document.body.clientHeight, true);
    }

    if (this.contentBody.open === true) {
      if (panel != this._sidepanel) { // change panel only.
        this._sidepanel = panel;
      } else { // just close panel. (disable icon amp.)
        this._sidepanel = '';
        this.toggleSidePanelUI();
      }
    } else { // open new panel.
      this._sidepanel = panel;
      this.toggleSidePanelUI();
    }
  }

  logout() {
    const keys = Object.keys(localStorage);
    for (let i = 0; i < keys.length; i++) {
      const key = keys[i];
      if (/^(BackendAIWebUI\.login\.)/.test(key)) {
        localStorage.removeItem(key);
      }
    }
    // remove data in sessionStorage
    sessionStorage.clear();
    globalThis.location.reload();
  }

  protected render() {
    return html`
      <link rel="stylesheet" href="resources/fonts/font-awesome-all.min.css">
      <link rel="stylesheet" href="resources/custom.css">
      <div id="loading-curtain" class="loading-background"></div>
      <mwc-drawer id="app-body" class="${this.mini_ui ? 'mini-ui' : ''}">
        <div class="drawer-menu" style="height:100vh;">
          <div id="portrait-bar" class="draggable">
            <button class="horizontal center layout flex bar draggable" style="cursor:pointer;border:0;outline:0;background:transparent;" @click="${() => ForkliftUtils._moveTo('/summary')}">
              <div class="portrait-canvas"></div>
              <div class="vertical start-justified layout full-menu" style="margin-left:10px;margin-right:10px;">
                <div class="site-name"><span class="bold">Forklift</span></div>
              </div>
              <span class="flex"></span>
            </button>
          </div>
          <div class="horizontal center-justified center layout flex" style="max-height:40px;">
            <mwc-icon-button id="mini-ui-toggle-button" style="color:#fff;" icon="menu" slot="navigationIcon" @click="${() => this.toggleSidebarUI()}"></mwc-icon-button>
            <mwc-icon-button class="full-menu side-menu fg ${this.contentBody && this.contentBody.open === true && this._sidepanel === 'notification' ? 'yellow' : 'white'}" id="notification-icon" icon="notification_important" @click="${() => this._openSidePanel('notification')}"></mwc-icon-button>
            <mwc-icon-button class="full-menu side-menu fg ${this.contentBody && this.contentBody.open === true && this._sidepanel === 'task' ? 'yellow' : 'white'}" id="task-icon" icon="ballot" @click="${() => this._openSidePanel('task')}"></mwc-icon-button>
          </div>
          <mwc-list id="sidebar-menu" class="sidebar list">
            <mwc-list-item graphic="icon" ?selected="${this._page === 'summary'}" @click="${() => ForkliftUtils._moveTo('/summary')}">
              <i class="fas fa-th-large" slot="graphic" id="summary-menu-icon"></i>
              <span class="full-menu">Summary</span>
            </mwc-list-item>
            <mwc-list-item graphic="icon" ?selected="${this._page === 'builder'}" @click="${() => ForkliftUtils._moveTo('/builder')}">
              <i class="fas fa-cogs" slot="graphic" id="builder-menu-icon"></i>
              <span class="full-menu">Builder</span>
            </mwc-list-item>
            <mwc-list-item graphic="icon" ?selected="${this._page === 'tasks'}" @click="${() => ForkliftUtils._moveTo('/tasks')}">
              <i class="fa fa-list" slot="graphic" id="tasks-menu-icon"></i>
              <span class="full-menu">Tasks</span>
            </mwc-list-item>
            <mwc-list-item graphic="icon" ?selected="${this._page === 'environments'}" @click="${() => ForkliftUtils._moveTo('/environments')}">
              <i class="fas fa-microchip" slot="graphic" id="environments-menu-icon"></i>
              <span class="full-menu">Environments</span>
            </mwc-list-item>
          </mwc-list>
        </div>
        <div id="app-content" slot="appContent">
          <mwc-drawer id="content-body">
            <div class="sidepanel-drawer">
              <backend-ai-sidepanel-notification class="sidepanel" ?active="${this._sidepanel === 'notification'}"></backend-ai-sidepanel-notification>
              <backend-ai-sidepanel-task class="sidepanel" ?active="${this._sidepanel === 'task'}"></backend-ai-sidepanel-task>
            </div>
            <div slot="appContent">
              <mwc-top-app-bar-fixed id="main-toolbar" class="draggable">
                <div class="horizontal layout" slot="title" id="welcome-message" style="font-size:12px;margin-left:10px;padding-top:10px;">
                  <p>Welcome,</p>
                  <p class="user-name">${this._getUsername()}</p>
                  <p>.</p>
                </div>
                <div slot="actionItems" style="margin:0;">
                  <div class="horizontal center layout">
                    <div class="vertical layout center" style="position:relative;padding-top:10px;">
                      <span class="email" style="color:#8c8484;font-size:12px;line-height:22px;text-align:left;-webkit-font-smoothing:antialiased;margin:auto 10px;">
                        User Name
                      </span>
                      <mwc-menu id="dropdown-menu" class="user-menu">
                        <mwc-list-item class="horizontal layout start center" style="border-bottom:1px solid #ccc;">
                          <mwc-icon class="dropdown-menu">perm_identity</mwc-icon>
                          <span class="dropdown-menu-name">${this._getUserId()}</span>
                        </mwc-list-item>
                      </mwc-menu>
                    </div>
                    <span class="full_name user-name" style="font-size:14px;text-align:right;-webkit-font-smoothing:antialiased;margin:auto 0px auto 10px; padding-top:10px;">
                      ${this._getUsername()}
                    </span>
                    <mwc-icon-button id="dropdown-button" style="font-size: 0.5rem;">
                      <i class="fas fa-user-alt fa-xs" style="color:#8c8484;"></i>
                    </mwc-icon-button>
                    <div class="vertical-line" style="height:35px;"></div>
                      <div class="horizontal layout center" style="margin:auto 10px;padding-top:10px;">
                        <span class="log_out" style="font-size:12px;margin:auto 0px;color:#8c8484;">
                          Logout
                        </span>
                        <mwc-icon-button @click="${() => this.logout()}" style="padding-bottom:5px;">
                          <i class="fas fa-sign-out-alt fa-xs" style="color:#8c8484;"></i>
                        </mwc-icon-button>
                      </div>
                    </div>
                </div>
              </mwc-top-app-bar-fixed>
              <div class="content" style="box-sizing:border-box; padding:14px;">
                <div id="navbar-top" class="navbar-top horizontal flex layout wrap">
                  <section role="main" id="content" class="container layout vertical center">
                    <div id="app-page">
                      <forklift-login-view class="page" name="login" ?active="${this._page === 'login'}"><mwc-circular-progress indeterminate></mwc-circular-progress></forklift-login-view>
                      <forklift-summary-view class="page" name="summary" ?active="${this._page === 'summary'}"><mwc-circular-progress indeterminate></mwc-circular-progress></forklift-summary-view>
                      <forklift-build-view class="page" name="builder" ?active="${this._page === 'builder'}"><mwc-circular-progress indeterminate></mwc-circular-progress></forklift-build-view>
                      <forklift-environment-view class="page" name="environments" ?active="${this._page === 'environments'}"><mwc-circular-progress indeterminate></mwc-circular-progress></forklift-environment-view>
                      <forklift-tasks-view class="page" name="tasks" ?active="${this._page === 'tasks'}"><mwc-circular-progress indeterminate></mwc-circular-progress></forklift-tasks-view>
                      <forklift-error-view class="page" name="error" ?active="${this._page === 'error'}"><mwc-circular-progress indeterminate></mwc-circular-progress></forklift-error-view>
                    </div>
                  </section>
                </div>
              </div>
            </div>
          </mwc-drawer>
        </div>
      </mwc-drawer>
      <div id="mini-tooltips" style="display:${this.mini_ui ? 'block' : 'none'};">
        <wl-popover anchor="#mini-ui-toggle-button" .anchorOpenEvents="${['mouseover']}" fixed disablefocustrap
           anchororiginx="right" anchororiginy="center" transformoriginx="left" transformOriginY="center">
          <wl-popover-card>
            <div style="padding:5px">
              <mwc-icon-button disabled class="temporarily-hide side-menu fg ${this.contentBody && this.contentBody.open === true && this._sidepanel === 'feedback' ? 'red' : 'black'}" id="feedback-icon-popover" icon="question_answer"></mwc-icon-button>
              <mwc-icon-button class="side-menu fg ${this.contentBody && this.contentBody.open === true && this._sidepanel === 'notification' ? 'red' : 'black'}" id="notification-icon-popover" icon="notification_important" @click="${() => this._openSidePanel('notification')}"></mwc-icon-button>
              <mwc-icon-button class="side-menu fg ${this.contentBody && this.contentBody.open === true && this._sidepanel === 'task' ? 'red' : 'black'}" id="task-icon-popover" icon="ballot" @click="${() => this._openSidePanel('task')}"></mwc-icon-button>
            </div>
          </wl-popover-card>
        </wl-popover>
      </div>
      <backend-ai-offline-indicator ?active="${this._offlineIndicatorOpened}">
        ${this._offline ? 'You are offline' : 'You are online'}.
      </backend-ai-offline-indicator>
      <forklift-notification id="notification"></forklift-notification>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'forklift-webui': ForkliftWebUI;
  }
}
