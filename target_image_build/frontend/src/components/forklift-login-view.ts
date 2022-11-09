import {css, html, CSSResultGroup} from 'lit';
import {customElement, property} from 'lit/decorators.js';

import {BackendAIPage} from './backend-ai-page';
import {BackendAiStyles} from './backend-ai-general-styles';
import {
  IronFlex,
  IronFlexAlignment,
  IronFlexFactors,
  IronPositioning
} from '../plastics/layout/iron-flex-layout-classes';
import './backend-ai-dialog';
import './forklift-signup';
import ForkliftSignup from './forklift-signup';

import '@material/mwc-icon/mwc-icon';
import '@material/mwc-button/mwc-button';
import '@material/mwc-textfield/mwc-textfield';
import {ForkliftUtils} from './forklift-utils';

@customElement('forklift-login-view')
export default class ForkliftLoginView extends BackendAIPage {
  @property({type: String}) email = '';
  @property({type: String}) user_id = '';
  @property({type: String}) password = '';
  @property({type: String}) blockType = '';
  @property({type: String}) blockMessage = '';
  @property({type: Number}) login_attempt_limit = 500;
  @property({type: Number}) login_block_time = 180;
  @property({type: Object}) client;
  @property({type: Object}) clientConfig;
  @property({type: Object}) loginPanel;
  @property({type: Object}) blockPanel;
  @property({type: Boolean}) is_connected = false;
  @property({type: Boolean}) signup_support = false;
  @property({type: Boolean}) allowAnonymousChangePassword = false;

  firstUpdated() {
    this.loginPanel = this.shadowRoot?.querySelector('#login-panel');
    this.blockPanel = this.shadowRoot?.querySelector('#block-panel');
    document.addEventListener('refresh-login-panel-with-config', (e: any) => this.refreshWithConfig(e.detail.config));
    ForkliftUtils._observeElementVisibility(this, (isVisible) => {
      if (isVisible) {
        this._enableUserInput();
        this.login();
      }
    });
  }

  _disableUserInput() {
    (this.shadowRoot?.querySelector('#id_user_email') as HTMLInputElement).disabled = true;
    (this.shadowRoot?.querySelector('#id_password') as HTMLInputElement).disabled = true;
    (this.shadowRoot?.querySelector('.waiting-animation') as HTMLDivElement).style.display = 'flex';
  }

  _enableUserInput() {
    (this.shadowRoot?.querySelector('#id_user_email') as HTMLInputElement).disabled = false;
    (this.shadowRoot?.querySelector('#id_password') as HTMLInputElement).disabled = false;
    (this.shadowRoot?.querySelector('.waiting-animation') as HTMLDivElement).style.display = 'none';
  }

  refreshWithConfig(config) {
    if (typeof config.plugin === 'undefined' || typeof config.plugin.login === 'undefined' || config.plugin.login === '') {
      this._enableUserInput();
    }
    if (typeof config.general === 'undefined' || typeof config.general.signupSupport === 'undefined' || config.general.signupSupport === '' || config.general.signupSupport == false) {
      this.signup_support = false;
    } else {
      this.signup_support = true;
      (this.shadowRoot?.querySelector('#signup-dialog') as ForkliftSignup).active = true;
    }
    if (typeof config.general === 'undefined' || typeof config.general.allowAnonymousChangePassword === 'undefined' || config.general.allowAnonymousChangePassword === '' || config.general.allowAnonymousChangePassword == false) {
      this.allowAnonymousChangePassword = false;
    } else {
      this.allowAnonymousChangePassword = true;
    }
    if (typeof config.general === 'undefined' || typeof config.general.loginAttemptLimit === 'undefined' || config.general.loginAttemptLimit === '') {
    } else {
      this.login_attempt_limit = parseInt(config.general.loginAttemptLimit);
    }
    if (typeof config.general === 'undefined' || typeof config.general.loginBlockTime === 'undefined' || config.general.loginBlockTime === '') {
    } else {
      this.login_block_time = parseInt(config.general.loginBlockTime);
    }
  }

  /**
   * Open loginPanel.
   * */
  open() {
    if (this.loginPanel.open !== true) {
      this.loginPanel.show();
    }
    if (this.blockPanel.open === true) {
      this.blockPanel.hide();
    }
  }

  /**
  * Close the loginPanel
  * */
  close() {
    if (this.loginPanel.open === true) {
      this.loginPanel.hide();
    }
    if (this.blockPanel.open === true) {
      this.blockPanel.hide();
    }
  }

  /**
   * Hide the blockPanel.
   * */
  free() {
    this.blockPanel.hide();
  }

  /**
   * Show the blockPanel.
   *
   * @param {string} message - block message
   * @param {string} type - block type
   * */
  block(message = '', type = '') {
    this.blockMessage = message;
    this.blockType = type;
    setTimeout(() => {
      if (this.blockPanel.open === false && this.is_connected === false && this.loginPanel.open === false) {
        this.blockPanel.show();
      }
    }, 2000);
  }

  _validate_login_data() {
    let msg = '';
    const emailEl = this.shadowRoot?.querySelector('#id_user_email') as HTMLInputElement;
    const passwordEl = this.shadowRoot?.querySelector('#id_password') as HTMLInputElement;
    // show error message when email or password input is empty
    if (this.email === '' || this.email === 'undefined' || this.password === '' || this.password === 'undefined') {
      msg = 'Please input login info';
    }
    if (!emailEl.validity.valid || !passwordEl.validity.valid) {
      msg = 'Please input valid data';
    }
    if (msg !== '') {
      globalThis.notification.show(msg);
      this._enableUserInput();
      return false;
    }
    return true;
  }

  /**
   * Login according to connection_mode and api_endpoint.
   *
   * NOTE: For now, it just supports e-mail based login.
   *
   * @param {boolean} showError
   * */
  login(showError = true) {
    this.open();
  }

  _login(e) {
    const loginAttempt = globalThis.backendaioptions.get('login_attempt', 0, 'general');
    const lastLogin = globalThis.backendaioptions.get('last_login', Math.floor(Date.now() / 1000), 'general');
    const currentTime = Math.floor(Date.now() / 1000);
    if (loginAttempt >= this.login_attempt_limit && currentTime - lastLogin > this.login_block_time) { // Reset login counter and last login after 180sec.
      globalThis.backendaioptions.set('last_login', currentTime, 'general');
      globalThis.backendaioptions.set('login_attempt', 0, 'general');
    } else if (loginAttempt >= this.login_attempt_limit) { // login count exceeds limit, block login and set the last login.
      globalThis.backendaioptions.set('last_login', currentTime, 'general');
      globalThis.backendaioptions.set('login_attempt', loginAttempt + 1, 'general');
      const msg = 'Too many attempt';
      globalThis.notification.show(msg);
      return;
    } else {
      globalThis.backendaioptions.set('login_attempt', loginAttempt + 1, 'general');
    }

    this._disableUserInput();
    this.email = (this.shadowRoot?.querySelector('#id_user_email') as HTMLInputElement).value;
    this.password = (this.shadowRoot?.querySelector('#id_password') as HTMLInputElement).value;

    if (!this._validate_login_data()) return;

    const body = {
      username: this.email,
      password: this.password,
    };
    ForkliftUtils.submitForm(e, body)
      .then((response) => {
        if (this.loginPanel.open !== true) {
          this.block();
        }
        // Not authenticated yet.
        if (!response) {
          this.open();
          this._enableUserInput();
          if (this.email != '' && this.password != '') {
            const msg = 'Login information mismatch. Please check your login information.';
            globalThis.notification.show(msg);
          }
        } else {
          if (this.loginPanel.open !== true) {
            this.block();
          }
          new Promise(() => {
            const currentTime = Math.floor(Date.now() / 1000);
            globalThis.backendaioptions.set('last_login', currentTime, 'general');
            globalThis.backendaioptions.set('login_attempt', 0, 'general');
            globalThis.backendaioptions.set('email', this.email);
            sessionStorage.setItem('token', 'Bearer ' + response.access_token);
            let event = new CustomEvent('backend-ai-connected');
            document.dispatchEvent(event);
            event = new CustomEvent('show-menu');
            document.dispatchEvent(event);
            this.is_connected = true;
            this.close();
            ForkliftUtils._moveTo('/');
          });
        }
      }).catch((err) => {
        globalThis.notification.show(err);
        this.open();
        this._enableUserInput();
      });
  }

  _cancelLogin(e) {
    this._hideDialog(e);
    this.open();
    this._enableUserInput();
  }

  _submitIfEnter(e) {
    if (e.keyCode == 13) this._login(e);
  }

  _showSignupDialog() {
    (this.shadowRoot?.querySelector('#signup-dialog') as ForkliftSignup).open();
  }

  _showChangePasswordEmailDialog() {
    // this.shadowRoot?.querySelector('#change-password-confirm-dialog')?.show();
  }

  static get styles(): CSSResultGroup {
    return [
      BackendAiStyles,
      IronFlex,
      IronFlexAlignment,
      IronFlexFactors,
      IronPositioning,
      // language=CSS
      css`
        .warning {
          color: red;
        }

        backend-ai-dialog {
          --component-width: 400px;
          --component-padding: 0;
          --component-background-color: rgba(247, 247, 246, 1);
        }

        fieldset input {
          width: 100%;
          border: 0;
          margin: 15px 0 0 0;
          font: inherit;
          font-size: 16px;
          outline: none;
        }

        mwc-textfield {
          font-family: var(--general-font-family);
          --mdc-theme-primary: black;
          --mdc-text-field-fill-color: rgb(250, 250, 250);
          width: 100%;
        }

        .endpoint-text {
          --mdc-text-field-disabled-line-color: rgba(0, 0, 0, 0.0);
        }

        mwc-icon-button {
          /*color: rgba(0, 0, 0, 0.54); Matched color with above icons*/
          color: var(--paper-blue-600);
          --mdc-icon-size: 24px;
        }

        mwc-icon-button.endpoint-control-button {
          --mdc-icon-size: 16px;
          --mdc-icon-button-size: 24px;
          color: red;
        }

        mwc-menu {
          font-family: var(--general-font-family);
          --mdc-menu-min-width: 400px;
          --mdc-menu-max-width: 400px;
        }

        mwc-list-item[disabled] {
          --mdc-menu-item-height: 30px;
          border-bottom: 1px solid #ccc;
        }

        mwc-button {
          background-image: none;
          --mdc-theme-primary: var(--general-button-background-color);
          --mdc-on-theme-primary: var(--general-button-background-color);
        }

        mwc-button[unelevated] {
          background-image: none;
          --mdc-theme-primary: var(--general-button-background-color);
        }

        mwc-button[outlined] {
          background-image: none;
          --mdc-button-outline-width: 2px;
          --mdc-button-disabled-outline-color: var(--general-button-background-color);
          --mdc-button-disabled-ink-color: var(--general-button-background-color);
          --mdc-theme-primary: var(--general-button-background-color);
          --mdc-on-theme-primary: var(--general-button-background-color);
        }

        h3 small {
          --button-font-size: 12px;
        }

        wl-icon {
          --icon-size: 16px;
          padding: 0;
        }

        .login-input {
          background-color: #FAFAFA;
          border-bottom: 1px solid #ccc;
          height: 50px;
        }

        .login-input mwc-icon {
          margin: 5px 15px 5px 15px;
          color: #737373;
        }

        .login-input input {
          width: 100%;
          background-color: #FAFAFA;
          margin-bottom: 5px;
          font-size: 18px;
          margin-top: 5px;
        }

        #login-title-area {
          height: var(--login-banner-height, 0);
          width: var(--login-banner-width, 0);
          background: var(--login-banner-background, none);
        }

        .login-form {
          position: relative;
        }

        .waiting-animation {
          top: 20%;
          left: 40%;
          position: absolute;
          z-index: 2;
        }

        .sk-folding-cube {
          margin: 20px auto;
          width: 15px;
          height: 15px;
          position: relative;
          margin: auto;
          -webkit-transform: rotateZ(45deg);
          transform: rotateZ(45deg);
        }

        .sk-folding-cube .sk-cube {
          float: left;
          width: 50%;
          height: 50%;
          position: relative;
          -webkit-transform: scale(1.1);
          -ms-transform: scale(1.1);
          transform: scale(1.1);
        }

        .sk-folding-cube .sk-cube:before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          background-color: #3e872d;
          -webkit-animation: sk-foldCubeAngle 2.4s infinite linear both;
          animation: sk-foldCubeAngle 2.4s infinite linear both;
          -webkit-transform-origin: 100% 100%;
          -ms-transform-origin: 100% 100%;
          transform-origin: 100% 100%;
        }

        .sk-folding-cube .sk-cube2 {
          -webkit-transform: scale(1.1) rotateZ(90deg);
          transform: scale(1.1) rotateZ(90deg);
        }

        .sk-folding-cube .sk-cube3 {
          -webkit-transform: scale(1.1) rotateZ(180deg);
          transform: scale(1.1) rotateZ(180deg);
        }

        .sk-folding-cube .sk-cube4 {
          -webkit-transform: scale(1.1) rotateZ(270deg);
          transform: scale(1.1) rotateZ(270deg);
        }

        .sk-folding-cube .sk-cube2:before {
          -webkit-animation-delay: 0.3s;
          animation-delay: 0.3s;
        }

        .sk-folding-cube .sk-cube3:before {
          -webkit-animation-delay: 0.6s;
          animation-delay: 0.6s;
        }

        .sk-folding-cube .sk-cube4:before {
          -webkit-animation-delay: 0.9s;
          animation-delay: 0.9s;
        }

        @-webkit-keyframes sk-foldCubeAngle {
          0%,
          10% {
            -webkit-transform: perspective(140px) rotateX(-180deg);
            transform: perspective(140px) rotateX(-180deg);
            opacity: 0;
          }
          25%,
          75% {
            -webkit-transform: perspective(140px) rotateX(0deg);
            transform: perspective(140px) rotateX(0deg);
            opacity: 1;
          }
          90%,
          100% {
            -webkit-transform: perspective(140px) rotateY(180deg);
            transform: perspective(140px) rotateY(180deg);
            opacity: 0;
          }
        }

        @keyframes sk-foldCubeAngle {
          0%,
          10% {
            -webkit-transform: perspective(140px) rotateX(-180deg);
            transform: perspective(140px) rotateX(-180deg);
            opacity: 0;
          }
          25%,
          75% {
            -webkit-transform: perspective(140px) rotateX(0deg);
            transform: perspective(140px) rotateX(0deg);
            opacity: 1;
          }
          90%,
          100% {
            -webkit-transform: perspective(140px) rotateY(180deg);
            transform: perspective(140px) rotateY(180deg);
            opacity: 0;
          }
        }

        #loading-message {
          margin-left: 10px;
        }

      `];
  }

  render() {
    // language=HTML
    return html`
      <link rel="stylesheet" href="resources/custom.css">
      <backend-ai-dialog id="login-panel" noclosebutton fixed blockscrolling persistent disablefocustrap escapeKeyAction="">
        <div slot="title">
          <div id="login-title-area"></div>
          <div class="horizontal center layout">
            <h4 style="padding:15px 0 0 5px;margin:0;margin-bottom:5px;">Forklift</h4>
            <div class="flex"></div>
          </div>
        </div>
        <div slot="content" class="login-panel intro centered" style="margin:0;">
          <h3 class="horizontal center layout" style="margin:0 25px;font-weight:700;min-height:40px;">
            <div>Login With E-mail</div>
            <div class="flex"></div>
          </h3>
          <div class="login-form">
            <div class="waiting-animation horizontal layout wrap">
              <div class="sk-folding-cube">
                <div class="sk-cube1 sk-cube"></div>
                <div class="sk-cube2 sk-cube"></div>
                <div class="sk-cube4 sk-cube"></div>
                <div class="sk-cube3 sk-cube"></div>
              </div>
              <div id="loading-message">Waiting...</div>
            </div>
            <form id="session-login-form" method="post" uri="/login/access-token/">
              <fieldset>
                <div class="horizontal layout start-justified center login-input">
                  <mwc-icon>email</mwc-icon>
                  <input type="email" id="id_user_email" maxlength="64" autocomplete="username"
                      label="E-mail" placeholder="E-mail" icon="email" value="${this.email}" @keyup="${this._submitIfEnter}">
                </div>
                <div class="horizontal layout start-justified center login-input">
                  <mwc-icon>vpn_key</mwc-icon>
                  <input type="password" id="id_password" autocomplete="current-password"
                      label="Password" placeholder="Password" icon="vpn_key" value="${this.password}" @keyup="${this._submitIfEnter}">
                </div>
              <!-- </fieldset>
            </form> -->
            <!-- <form>
              <fieldset> -->
                <mwc-button type="submit" unelevated id="login-button" icon="check" fullwidth label="Login" @click="${(e) => this._login(e)}"></mwc-button>
                <div class="layout horizontal" style="margin-top:2em;">
                  ${this.signup_support ? html`
                    <div class="vertical center-justified layout" style="width:100%;">
                      <div style="font-size:12px; margin:0 10px; text-align:center;">Not a user?</div>
                      <mwc-button
                          outlined
                          label="Sign Up"
                          @click="${() => this._showSignupDialog()}"></mwc-button>
                    </div>
                  `: html``}
                  <!-- Future work (Not implemented in server side) -->
                  <!-- ${this.signup_support && this.allowAnonymousChangePassword ? html`
                    <span class="flex" style="min-width:1em;"></span>
                  `: html``}
                  ${this.allowAnonymousChangePassword ? html`
                    <div class="vertical center-justified layout" style="width:100%;">
                      <div style="font-size:12px; margin:0 10px; text-align:center;">Forgot password?</div>
                      <mwc-button
                          outlined
                          label="Change Password"
                          @click="${() => this._showChangePasswordEmailDialog()}"></mwc-button>
                    </div>
                  ` : html``} -->
                </div>
              </fieldset>
            </form>
          </div>
        </div>
      </backend-ai-dialog>
      <backend-ai-dialog id="block-panel" fixed blockscrolling persistent>
        ${this.blockMessage != '' ? html`
          ${this.blockType !== '' ? html`
            <span slot="title" id="work-title">${this.blockType}</span>
          ` : html``}
          <div slot="content" style="text-align:center;padding-top:15px;">
          ${this.blockMessage}
          </div>
          <div slot="footer" class="horizontal center-justified flex layout">
          <mwc-button
              outlined
              fullwidth
              label="Cancel Login"
              @click="${(e) => this._cancelLogin(e)}"></mwc-button>
          </div>
        ` : html``}
      </backend-ai-dialog>
      <forklift-signup id="signup-dialog"></forklift-signup>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'forklift-login-view': ForkliftLoginView;
  }
}
