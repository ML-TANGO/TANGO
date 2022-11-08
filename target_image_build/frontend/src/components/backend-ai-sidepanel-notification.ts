/**
 @license
 Copyright (c) 2015-2022 Lablup Inc. All rights reserved.
 */
import {css, CSSResultGroup, html} from 'lit';
import {customElement, property} from 'lit/decorators.js';

// import '../plastics/mwc/mwc-drawer';
import '@material/mwc-drawer/mwc-drawer';
import '@material/mwc-icon/mwc-icon';
import '@material/mwc-list/mwc-list';
import '@material/mwc-list/mwc-list-item';

import {BackendAIPage} from './backend-ai-page';
import {
  IronFlex,
  IronFlexAlignment,
  IronFlexFactors,
  IronPositioning
} from '../plastics/layout/iron-flex-layout-classes';

/**
 Backend.AI Sidepanel notification viewer for Console

 `backend-ai-sidepanel-notification` is a sidepanel notification viewer for web UI.

 Example:
@group Backend.AI Web UI
 @element backend-ai-sidepanel-notification
 */
@customElement('backend-ai-sidepanel-notification')
export default class BackendAiSidepanelNotification extends BackendAIPage {
  @property({type: Boolean}) active = true;
  @property({type: Array}) notifications = [];

  /**
   *  Backend.AI Task manager for Console
   *
   */
  constructor() {
    super();
  }

  static get styles(): CSSResultGroup {
    return [
      IronFlex,
      IronFlexAlignment,
      IronFlexFactors,
      IronPositioning,
      // language=css
      css`
        h3 {
          font-size: 16px;
          color: #242424;
          display: block;
          width: 100%;
          height: 25px;
          padding: 5px 15px;
          border-bottom: 1px solid #cccccc;
        }

        mwc-list {
          padding: 0;
          margin: 0;
        }
      `
    ];
  }

  async _viewStateChanged(active: boolean) {
    await this.updateComplete;
    if (active === false) {
      return;
    }
  }

  render() {
    // language=HTML
    return html`
      <div id="container">
        <h3>Notification</h3>
        <mwc-list>
        ${this.notifications.map((item: any) =>
    html`
          <mwc-list-item twoline>
            <span>${item.outerText}</span>
            <span slot="secondary">${item.getAttribute('created')}</span>
          </mwc-list-item>
          <li divider role="separator"></li>`)}
          ${this.notifications.length === 0 ? html`
            <div style="padding:15px 0;width:100%;text-align:center;">
              No Notification.
            </div>
        ` : html``}
        </mwc-list>
      </div>
    `;
  }

  shouldUpdate() {
    return this.active;
  }

  firstUpdated() {
    this.notifications = globalThis.lablupNotification.notifications;
    document.addEventListener('backend-ai-notification-changed', () => this.refresh());
  }

  connectedCallback() {
    super.connectedCallback();
  }

  disconnectedCallback() {
    super.disconnectedCallback();
  }

  async refresh() {
    this.notifications = globalThis.lablupNotification.notifications;
    await this.requestUpdate();
  }
}
declare global {
  interface HTMLElementTagNameMap {
    'backend-ai-sidepanel-notification': BackendAiSidepanelNotification;
  }
}
