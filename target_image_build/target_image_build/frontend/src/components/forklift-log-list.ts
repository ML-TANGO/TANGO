import {css, CSSResultGroup, html, render} from 'lit';
import {customElement, property} from 'lit/decorators.js';

import {BackendAIPage} from './backend-ai-page';
import {BackendAiStyles} from './backend-ai-general-styles';
import {IronFlex, IronFlexAlignment} from '../plastics/layout/iron-flex-layout-classes';
import {ForkliftUtils} from './forklift-utils';

import '@vaadin/vaadin-grid/vaadin-grid';
import {Grid} from '@vaadin/vaadin-grid/vaadin-grid';

import LablupLoadingSpinner from './lablup-loading-spinner';
import BackendAiDialog from './backend-ai-dialog';

@customElement('forklift-log-list')
export default class ForkliftLogList extends BackendAIPage {
  @property({type: Array}) logs: any[] = [];
  @property({type: Object}) spinner = Object();
  @property({type: Object}) clearLogsDialog = Object();
  @property({type: Object}) _grid = Object();
  @property({type: Object}) _boundMessageRenderer = this._messageRenderer.bind(this);

  firstUpdated() {
    this._grid = this.shadowRoot?.querySelector('#logs-table') as Grid;
    this.spinner = this.shadowRoot?.querySelector<LablupLoadingSpinner>('#loading-spinner');
    this.clearLogsDialog = this.shadowRoot?.querySelector<BackendAiDialog>('#clearlogs-dialog');

    if (!ForkliftUtils._checkLogin()) {
      ForkliftUtils._moveTo('/login');
    }
    ForkliftUtils._observeElementVisibility(this, (isVisible) => {
      if (isVisible && ForkliftUtils._checkLogin()) {
        this._getStreamLogData();
      }
    });

    document.addEventListener('refresh-log-data', () => this._getStreamLogData());
    document.addEventListener('show-clear-logs-dialog', () => this._showClearLogsDialog());
  }

  _parseLogData(e, logs, grid) {
    let tempLogs;
    if (ForkliftUtils.isJsonString(e)) {
      tempLogs = JSON.parse(e)[0]?.payload;
    }
    logs.push(tempLogs);
    grid?.clearCache();
  }

  _getFirstRunningTaskId() {
    return ForkliftUtils.fetch('/user/my_task/running/', {method: 'GET'}, null, true)
      .then((resp) => {
        return resp[0]?.task_id;
      });
  }

  async _getStreamLogData() {
    this.logs = [];
    const taskId = await this._getFirstRunningTaskId();
    if (taskId) {
      globalThis.tasker.add(
        'StreamingLogs',
        ForkliftUtils.attachEventSource(`/build/stream_log/${taskId}/`, this._parseLogData, this.logs, this._grid)
          .catch((err) => {
            globalThis.lablupNotification.show(err);
          }), '', 'log');
    }
  }

  // TODO: remove the recently requested logs from DB.
  _clearLogData() {
    this.logs = [];
    ForkliftUtils.closeEventSource('/build/stream_log/');
    this._grid.clearCache();
  }

  /**
   * Show clearLogsDialog.
   * */
  _showClearLogsDialog() {
    this.clearLogsDialog.show();
  }


  /**
   * Hide clearLogsDialog.
   * */
  _hideClearLogsDialog() {
    this.clearLogsDialog.hide();
  }

  /**
   * Remove log message.
   * */
  _removeLogMessage() {
    this._clearLogData();
    this.clearLogsDialog.hide();
    const msg = 'Log Messages have been removed.';
    globalThis.notification.show(msg);
    this.spinner.hide();
  }

  /**
   * Render an index.
   *
   * @param {DOMelement} root
   * @param {object} column (<vaadin-grid-column> element)
   * @param {object} rowData
   */
  _indexRenderer(root, column, rowData) {
    const idx = rowData.index + 1;
    render(
      html`
        <div>${idx}</div>
      `,
      root
    );
  }

  _messageRenderer(root, column, rowData) {
    render(
      html`
        ${rowData.item.error ? html`
          <span style="color:red;">${rowData.item.error}</span>
        ` : rowData.item.stream ? html`
          <span>${rowData.item.stream}</span>
        ` : html`
          <span>${JSON.stringify(rowData.item)}</span>
        `}
      `, root
    );
  }

  static get styles(): CSSResultGroup {
    return [
      BackendAiStyles,
      IronFlex,
      IronFlexAlignment,
      // language=CSS
      css`
        vaadin-grid {
          width: 100%;
          border: 0;
          font-size: 12px;
          height: calc(100vh - 235px);
        }
      `
    ];
  }

  render() {
    // language=HTML
    return html`
      <lablup-loading-spinner id="loading-spinner"></lablup-loading-spinner>
      <vaadin-grid id="logs-table" theme="row-stripes column-borders compact wrap-cell-content"
        aria-label="Logs" .items="${this.logs}">
        <vaadin-grid-column width="50px" header="#" flex-grow="0" text-align="end" resizable frozen .renderer="${this._indexRenderer}"></vaadin-grid-column>
        <vaadin-grid-column header="Message" resizable .renderer="${this._boundMessageRenderer}"></vaadin-grid-column>
      </vaadin-grid>
      <backend-ai-dialog id="clearlogs-dialog" fixed backdrop scrollable blockScrolling>
        <span slot="title">Are you sure you want to delete all of the log messages?</span>
        <div slot="content">WARNING: this cannot be undone!</div>
        <div slot="footer" class="horizontal end-justified flex layout">
          <mwc-button
              class="operation"
              id="discard-removal"
              label="No"
              @click="${() => this._hideClearLogsDialog()}"></mwc-button>
          <mwc-button
              unelevated
              class="operation"
              id="apply-removal"
              label="Yes"
              @click="${() => this._removeLogMessage()}"></mwc-button>
        </div>
      </backend-ai-dialog>
    `;
  }
}
declare global {
  interface HTMLElementTagNameMap {
    'forklift-log-list': ForkliftLogList;
  }
}
