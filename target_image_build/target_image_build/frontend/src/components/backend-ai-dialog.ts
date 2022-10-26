/**
 @license
 Copyright (c) 2015-2022 Lablup Inc. All rights reserved.
 */
// import {get as _text, registerTranslateConfig, translate as _t, use as setLanguage} from "lit-translate";
import {css, CSSResultGroup, html, LitElement} from 'lit';
import {customElement, property, query} from 'lit/decorators.js';

import {BackendAiStyles} from './backend-ai-general-styles';
import 'weightless/button';
import 'weightless/card';
import 'weightless/icon';

import '@material/mwc-icon-button';

import '../plastics/mwc/mwc-dialog';
import {IronFlex, IronFlexAlignment} from '../plastics/layout/iron-flex-layout-classes';

/**
 Backend.AI Dialog

 `backend-ai-dialog` is a dialog with close button

 Example:

 <backend-ai-dialog>
 ...
 </backend-ai-dialog>

@group Backend.AI Web UI
 @element backend-ai-dialog
 */
@customElement('backend-ai-dialog')
export default class BackendAiDialog extends LitElement {
  @property({type: Boolean}) fixed = false;
  @property({type: Boolean}) narrowLayout = false;
  @property({type: Boolean}) scrollable = false;
  @property({type: Boolean}) backdrop = false;
  @property({type: Boolean}) noclosebutton = false;
  @property({type: Boolean}) persistent = false;
  @property({type: Boolean}) blockscrolling = false;
  @property({type: Boolean}) hideActions = true;
  @property({type: Boolean}) open = false;
  @property({type: String}) type = 'normal';
  @property({type: Boolean}) closeWithConfirmation = false;
  @property({type: String}) escapeKeyAction = 'close';

  @query('#dialog') protected dialog;

  constructor() {
    super();
  }

  static get styles(): CSSResultGroup {
    return [
      BackendAiStyles,
      IronFlex,
      IronFlexAlignment,
      // language=CSS
      css`
        mwc-dialog {
          --mdc-dialog-min-width: var(--component-min-width, auto);
          --mdc-dialog-max-width: var(--component-max-width, 100%);
          --mdc-dialog-min-height: var(--component-min-height, auto);
          --mdc-dialog-max-height: var(--component-max-height, calc(100vh - 45px));
          --mdc-dialog-width: var(--component-width, auto);
          --mdc-dialog-height: var(--component-height, auto);
          --mdc-dialog-background-color: var(--component-background-color, auto);
          --mdc-typography-body1-font-family: var(--general-font-family);
          --mdc-typography-body1-font-color: black;
          --mdc-typography-headline6-font-family: var(--general-font-family);
          --mdc-typography-headline6-font-color: black;
          --mdc-shape-medium: 10px;
        }

        mwc-dialog > div.card {
          padding:0;
          margin:0;
        }

        mwc-dialog > div.card > h3 {
          background-color: var(--general-dialog-background-color, #ffffff);
        }

        mwc-dialog.warning h3 {
          color: red;
        }

        mwc-dialog div.content {
          padding: var(--component-padding, 15px);
          font-size: var(--component-font-size, 14px);
          word-break: keep-all;
          overflow-x: hidden;
        }

        mwc-dialog div.footer {
          padding: 5px 15px 15px 15px;
        }

        mwc-dialog[narrow] div.content,
        mwc-dialog[narrow] div.footer {
          padding: 0;
          margin: 0;
        }

        mwc-dialog[scrollable]::slotted([slot="content"]),
        mwc-dialog[scrollable] div.content-area {
          overflow-y: scroll; /* Has to be scroll (not auto) to get smooth scrolling on iOS */
          -webkit-overflow-scrolling: touch;
          max-height: calc(80vh - 127px);
        }

        mwc-dialog div.content h4 {
          font-size: 14px;
          padding: 5px 15px 5px 12px;
          margin: 0 0 10px 0;
          display: block;
          height: 20px;
          border-bottom: 1px solid #DDD !important;
        }
      `];
  }

  firstUpdated() {
    this.open = this.dialog.open;
    if (this.persistent) {
      this.dialog.scrimClickAction = '';
    }
    this.dialog.addEventListener('opened', () => {
      this.open = this.dialog.open;
    });
    this.dialog.addEventListener('closed', (e) => {
      // execute action only if the event target is dialog
      if (e.target.id === 'dialog' && 'action' in e.detail && e.detail.action === 'persistent') {
        this.show();
      } else {
        this.open = this.dialog.open;
      }

      /**
       * custom event for bubbling event of closing dialog
       */
      if (e.target.id === 'dialog' && 'action' in e.detail && e.detail.action === 'close') {
        const closeEvent = new CustomEvent('dialog-closed', {detail: ''});
        this.dispatchEvent(closeEvent);
      }
    });
  }

  connectedCallback() {
    super.connectedCallback();
  }

  /**
   * Hide a dialog.
   */
  _hideDialog() {
    this.hide();
  }

  /**
   * Open a dialog.
   */
  show() {
    this.dialog.show();
  }

  /**
   * Hide a dialog.
   */
  hide() {
    if (this.closeWithConfirmation) {
      const closeEvent = new CustomEvent('dialog-closing-confirm', {detail: ''});
      this.dispatchEvent(closeEvent);
    } else {
      this.dialog.close();
      this._resetScroll();
    }
  }

  /**
   * Move to top of the dialog.
   */
  _resetScroll() {
    const content = this.shadowRoot?.querySelector('.content-area');
    content?.scrollTo(0, 0);
  }

  render() {
    // language=HTML
    return html`
      <link rel="stylesheet" href="resources/custom.css">
      <mwc-dialog id="dialog"
                    ?fixed="${(this.fixed)}"
                    ?narrow="${(this.narrowLayout)}"
                    ?backdrop="${this.backdrop}"
                    ?persistent="${this.persistent}"
                    ?scrollable="${this.scrollable}"
                    ?hideActions="${this.hideActions}"
                    escapeKeyAction="${this.escapeKeyAction}"
                    blockscrolling="${this.blockscrolling}"
                    hideActions="${this.hideActions}"
                    style="padding:0;" class="${this.type}">
        <div elevation="1" class="card" style="margin:0;padding:0;">
          <h3 class="horizontal justified layout" style="font-weight:bold">
            <span class="vertical center-justified layout">
              <slot name="title"></slot>
            </span>
            <div class="flex"></div>
            <slot name="action"></slot>
            ${this.noclosebutton ? html`` : html`
            <mwc-icon-button icon="close" fab flat inverted @click="${() => this._hideDialog()}">
            </mwc-icon-button>
            `}
          </h3>
          <div class="content content-area">
            <slot name="content"></slot>
          </div>
          <div class="footer horizontal flex layout">
            <slot name="footer"></slot>
          </div>
        </div>
      </mwc-dialog>
      `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'backend-ai-dialog': BackendAiDialog;
  }
}
