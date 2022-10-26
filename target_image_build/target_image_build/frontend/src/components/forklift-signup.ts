import {css, CSSResultGroup, html} from 'lit';
import {customElement, property} from 'lit/decorators.js';

import {BackendAiStyles} from './backend-ai-general-styles';
import {
  IronFlex,
  IronFlexAlignment,
  IronFlexFactors,
  IronPositioning
} from '../plastics/layout/iron-flex-layout-classes';


import '@material/mwc-checkbox/mwc-checkbox';
import '@material/mwc-icon-button-toggle/mwc-icon-button-toggle';
import {Button} from '@material/mwc-button';
import {TextField} from '@material/mwc-textfield';

import BackendAiDialog from './backend-ai-dialog';
import {BackendAIPage} from './backend-ai-page';
import {ForkliftUtils} from './forklift-utils';

@customElement('forklift-signup')
export default class ForkliftSignup extends BackendAIPage {
  @property({type: String}) user_email = '';
  @property({type: String}) errorMsg = '';
  @property({type: Object}) signupPanel = Object();
  @property({type: Object}) blockPanel = Object();

  firstUpdated() {
    this.signupPanel = this.shadowRoot?.querySelector<BackendAiDialog>('#signup-panel');
    this.blockPanel = this.shadowRoot?.querySelector<BackendAiDialog>('#block-panel');
    const textfields = this.shadowRoot?.querySelectorAll<TextField>('mwc-textfield');
    for (let i = 0; i < textfields!.length; i++) {
      this._addInputValidator(textfields![i]);
    }
  }

  /**
   * Change state to 'ALIVE' when backend.ai client connected.
   *
   * @param {Booelan} active - The component will work if active is true.
   */
  async _viewStateChanged(active: boolean) {
    await this.updateComplete;
    if (active === false) {
      return;
    }
  }

  open() {
    if (this.signupPanel.open !== true) {
      this._clearUserInput();
      this.signupPanel.show();
    }
  }

  close() {
    if (this.signupPanel.open === true) {
      this.signupPanel.hide();
    }
  }

  block(message = '') {
    this.errorMsg = message;
    this.blockPanel.show();
  }

  _validate_data(value) {
    if (value != undefined && value != null && value != '') {
      return true;
    }
    return false;
  }

  _clearUserInput() {
    this._toggleInputField(true);
    const inputFields: Array<string> = ['#id_user_email', '#id_password1', '#id_password2'];
    inputFields.forEach((el: string) => {
      (this.shadowRoot?.querySelector(el) as TextField).value = '';
    });
    (this.shadowRoot?.querySelector('#signup-button-message') as HTMLSpanElement).innerHTML = 'Signup';
  }

  _signup() {
    const inputFields: Array<string> = ['#id_user_email', '#id_password1', '#id_password2'];
    const inputFieldsValidity: Array<boolean> = inputFields.map((el: string) => {
      (this.shadowRoot?.querySelector(el) as TextField).reportValidity();
      return (this.shadowRoot?.querySelector(el) as TextField).checkValidity();
    });
    // if any input is invalid, signup fails.
    if (inputFieldsValidity.includes(false)) {
      return;
    }
    const user_email = (this.shadowRoot?.querySelector('#id_user_email') as HTMLInputElement).value;
    const password = (this.shadowRoot?.querySelector('#id_password1') as HTMLInputElement).value;
    let msg = 'Processing...';
    globalThis.notification.show(msg);
    const body = {
      'email': user_email,
      'password': password,
    };
    ForkliftUtils.fetch(`/register/`, {method: 'POST', body: JSON.stringify(body)})
      .then(() => {
        this._toggleInputField(false);
        (this.shadowRoot?.querySelector('#signup-button-message') as HTMLSpanElement).innerHTML = 'Signup succeeded.';
        msg = 'Signup succeeded.';
        globalThis.notification.show(msg);
        setTimeout(() => {
          this.signupPanel.hide();
          this._clearUserInput();
        }, 1000);
      }).catch((e) => {
        if (e.msg) {
          msg = e.msg;
          globalThis.notification.show(msg);
        }
        console.log(e);
      });
  }

  _toggleInputField(isActive: boolean) {
    const inputFields: Array<string> = ['#signup-button'];
    inputFields.forEach((el: string) => {
      if (isActive) {
        (this.shadowRoot?.querySelector(el) as Button).removeAttribute('disabled');
      } else {
        (this.shadowRoot?.querySelector(el) as Button).setAttribute('disabled', 'true');
      }
    });
  }

  _togglePasswordVisibility(element) {
    const isVisible = element.__on;
    const password = element.closest('div').querySelector('mwc-textfield');
    isVisible ? password.setAttribute('type', 'text') : password.setAttribute('type', 'password');
  }

  _validateEmail() {
    const emailInput = this.shadowRoot?.querySelector('#id_user_email') as any;
    emailInput.validityTransform = (newValue, nativeValidity) => {
      if (!nativeValidity.valid) {
        if (nativeValidity.valueMissing) {
          emailInput.validationMessage = 'Email address is required.';
          return {
            valid: nativeValidity.valid,
            customError: !nativeValidity.valid
          };
        } else {
          emailInput.validationMessage = 'Invalid emaill address.';
          return {
            valid: nativeValidity.valid,
            customError: !nativeValidity.valid
          };
        }
      } else {
        // custom validation for email address using regex
        const regex = /^(([^<>()[\]\\.,;:\s@"]+(\.[^<>()[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/;
        const isValid = regex.exec(emailInput.value);
        if (!isValid) {
          emailInput.validationMessage = 'Invalid emaill address.';
        }
        return {
          valid: isValid,
          customError: !isValid
        };
      }
    };
  }

  _validatePassword1() {
    const passwordInput = this.shadowRoot?.querySelector('#id_password1') as TextField;
    const password2Input = this.shadowRoot?.querySelector('#id_password2') as TextField;
    password2Input.reportValidity();
    passwordInput.validityTransform = (newValue, nativeValidity) => {
      if (!nativeValidity.valid) {
        if (nativeValidity.valueMissing) {
          passwordInput.validationMessage = 'Password is required.';
          return {
            valid: nativeValidity.valid,
            customError: !nativeValidity.valid
          };
        } else {
          passwordInput.validationMessage = 'Use 8 or more characters with a mix of letters, numbers & symbols.';
          return {
            valid: nativeValidity.valid,
            customError: !nativeValidity.valid
          };
        }
      } else {
        return {
          valid: nativeValidity.valid,
          customError: !nativeValidity.valid
        };
      }
    };
  }

  _validatePassword2() {
    const password2Input = this.shadowRoot?.querySelector('#id_password2') as TextField;
    password2Input.validityTransform = (newValue, nativeValidity) => {
      if (!nativeValidity.valid) {
        if (nativeValidity.valueMissing) {
          password2Input.validationMessage = 'Password is required.';
          return {
            valid: nativeValidity.valid,
            customError: !nativeValidity.valid
          };
        } else {
          password2Input.validationMessage = 'Use 8 or more characters with a mix of letters, numbers & symbols.';
          return {
            valid: nativeValidity.valid,
            customError: !nativeValidity.valid
          };
        }
      } else {
        // custom validation for password input match
        const passwordInput = this.shadowRoot?.querySelector('#id_password1') as TextField;
        const isMatched = (passwordInput.value === password2Input.value);
        if (!isMatched) {
          password2Input.validationMessage = 'Those passwords didn\'t match. Try it again.';
        }
        return {
          valid: isMatched,
          customError: !isMatched
        };
      }
    };
  }

  _validatePassword() {
    this._validatePassword1();
    this._validatePassword2();
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
        fieldset input {
          width: 100%;
          border: 0;
          border-bottom: 1px solid #aaa;
          margin: 15px 0;
          font: inherit;
          font-size: 16px;
          outline: none;
        }

        fieldset input:focus {
          border-bottom: 1.5px solid #0d47a1;
        }

        #signup-panel {
          --dialog-width: 400px;
          --component-min-width: 400px;
          --component-max-width: 400px;
          --dialog-elevation: 0px 0px 5px 5px rgba(0, 0, 0, 0.1);
        }

        mwc-textfield {
          width: 100%;
          --mdc-text-field-fill-color: transparent;
          --mdc-theme-primary: var(--general-textfield-selected-color);
          --mdc-typography-font-family: var(--general-font-family);
        }

        mwc-textfield#id_user_name {
          margin-bottom: 18px;
        }

        mwc-button.full {
          width: 70%;
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

        mwc-checkbox {
          --mdc-theme-secondary: var(--general-checkbox-color);
        }
      `];
  }

  render() {
    // language=HTML
    return html`
      <backend-ai-dialog id="signup-panel" fixed blockscrolling persistent disablefocustrap>
        <span slot="title">Signup</span>
        <div slot="content" class="vertical flex layout">
          <mwc-textfield type="email" name="user_email" id="id_user_email"
                       maxlength="64" label="E-mail"
                       validateOnInitialRender
                       @change="${this._validateEmail}"
                       validationMessage="Email address is required."
                       value="${this.user_email}" required></mwc-textfield>
          <div class="horizontal flex layout">
            <mwc-textfield type="password" name="password1" id="id_password1"
                        label="Password" maxLength="64"
                        pattern="^(?=.*[A-Za-z])(?=.*\\d)(?=.*[@$!%*#?&])[A-Za-z\\d@$!%*#?&]{8,}$"
                        validationMessage="Password is required."
                        @change="${this._validatePassword}"
                        value="" required></mwc-textfield>
            <mwc-icon-button-toggle off onIcon="visibility" offIcon="visibility_off"
                        @click="${(e) => this._togglePasswordVisibility(e.target)}"></mwc-icon-button-toggle>
          </div>
          <div class="horizontal flex layout">
            <mwc-textfield type="password" name="password2" id="id_password2"
                        label="Password (again)" maxLength="64"
                        pattern="^(?=.*[A-Za-z])(?=.*\\d)(?=.*[@$!%*#?&])[A-Za-z\\d@$!%*#?&]{8,}$"
                        validationMessage="Password is required."
                        @change="${this._validatePassword}"
                        value="" required></mwc-textfield>
            <mwc-icon-button-toggle off onIcon="visibility" offIcon="visibility_off"
                                    @click="${(e) => this._togglePasswordVisibility(e.target)}">
            </mwc-icon-button-toggle>
          </div>
        </div>
        <div slot="footer" class="horizontal center-justified flex layout">
          <mwc-button
              id="signup-button"
              raised
              class="full"
              icon="check"
              @click="${() => this._signup()}">
                <span id="signup-button-message">Signup</span>
          </mwc-button>
        </div>
      </backend-ai-dialog>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'forklift-signup': ForkliftSignup;
  }
}
