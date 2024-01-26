import {css, CSSResultGroup, html, LitElement} from 'lit';
import {customElement, property} from 'lit/decorators.js';

import '../plastics/mwc/mwc-snackbar';

import '@material/mwc-button/mwc-button';
import '@material/mwc-icon-button/mwc-icon-button';

@customElement('forklift-notification')
export default class ForkliftNotification extends LitElement {
  @property({type: String}) text = '';
  @property({type: String}) actionButtonText = '';
  @property({type: Boolean}) isActionButton = false;
  @property({type: Boolean}) isCloseButton = false;
  @property({type: Boolean}) open = false;
  @property({type: Object}) notification: any;
  @property({type: Array}) notifications = [];
  @property({type: Number}) timeoutMs = 5000;

  constructor() {
    super();
  }

  static get is() {
    return 'forklift-notification';
  }

  static get styles(): CSSResultGroup {
    return [
      // language=CSS
      css`
        mwc-snackbar {
          position: fixed;
          right: 20px;
          font-size: 16px;
          font-weight: 400;
          font-family: 'Ubuntu', Roboto, sans-serif;
          z-index: 12345678;
          --mdc-snackbar-action-color: #72EB51;
        }

        mwc-button {
          font-size: 11px;
        }

        mwc-icon-button {
          --mdc-icon-size: 10px;
        }
      `,
    ];
  }

  firstUpdated() {
    this.notification = this.shadowRoot?.querySelector('#notification');
  }

  show(text='') {
    this.text = text;
    this.notification.show();
  }

  close(reason = '') {
    this.notification.close(reason);
  }

  render() {
    return html`
      <mwc-snackbar id="notification" labelText="${this.text}" timeoutMs="${this.timeoutMs}" ?open="${this.open}">
        <mwc-button slot="action" style="display:${this.isActionButton ? 'block' : 'none'}">${this.actionButtonText}</mwc-button>
        <mwc-icon-button icon="close" slot="dismiss" style="display:${this.isCloseButton ? 'block' : 'none'}"></mwc-icon-button>
      </mwc-snackbar>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'forklift-notification': ForkliftNotification;
  }
}
