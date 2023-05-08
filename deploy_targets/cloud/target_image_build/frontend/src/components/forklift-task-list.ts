import {css, CSSResultGroup, html, render} from 'lit';
import {customElement, property} from 'lit/decorators.js';

import {BackendAIPage} from './backend-ai-page';
import {BackendAiStyles} from './backend-ai-general-styles';
import {IronFlex, IronFlexAlignment} from '../plastics/layout/iron-flex-layout-classes';
import {ForkliftUtils} from './forklift-utils';

import '@vaadin/vaadin-grid/vaadin-grid';
import '@vaadin/vaadin-grid/vaadin-grid-column';
import '@vaadin/vaadin-grid/vaadin-grid-sort-column';
import '@vaadin/vaadin-grid/vaadin-grid-filter-column';
import {Grid} from '@vaadin/vaadin-grid/vaadin-grid';

import '@material/mwc-button/mwc-button';
import '@material/mwc-icon-button/mwc-icon-button';

import BackendAiDialog from './backend-ai-dialog';
import './lablup-codemirror';

@customElement('forklift-task-list')
export default class ForkliftTaskList extends BackendAIPage {
  @property({type: String}) condition = 'running';
  @property({type: String}) taskId;
  @property({type: Object}) spinner;
  @property({type: Object}) tasks;
  @property({type: Object}) taskGrid;
  @property({type: Array}) logs = [];
  @property({type: Object}) _boundCreatedAtRenderer = this._createdAtRenderer.bind(this);
  @property({type: Object}) _boundControlRenderer = this.controlRenderer.bind(this);
  @property({type: Object}) _boundMessageRenderer = this._messageRenderer.bind(this);

  firstUpdated() {
    this.taskGrid = this.shadowRoot?.querySelector('#tasks-grid') as Grid;
    this.spinner = this.shadowRoot?.querySelector('#loading-spinner');

    if (!ForkliftUtils._checkLogin()) {
      ForkliftUtils._moveTo('/login');
    }
    // Already connected
    ForkliftUtils._observeElementVisibility(this, (isVisible) => {
      if (isVisible && ForkliftUtils._checkLogin()) {
        this._refreshTaskData();
      }
    });
  }

  refresh() {
    this._refreshTaskData();
    // update current grid to new data
    this.taskGrid.clearCache();
  }

  _refreshTaskData() {
    this.spinner.show();
    ForkliftUtils.fetch(`/user/my_task/${this.condition}/`, {method: 'GET'}, null, true)
      .then((resp) => {
        this.taskGrid.items = resp;
      });
    this.spinner.hide();
  }

  _fetchRequestedDockerfile(taskId: string) {
    return ForkliftUtils.fetch(`/user/my_task/requested_dockerfile/${taskId}/`, {method: 'GET'}, null, true)
      .then((resp) => {
        return resp.requested_dockerfile_contents || '';
      }).catch((e) => {
        const msg = 'Error on fetching a Dockerfile.';
        globalThis.notification.show(msg);
      });
  }

  async _showRequestedDockerfileDialog(taskId: string) {
    this.spinner.show();
    const contents = await this._fetchRequestedDockerfile(taskId);
    const dialogEl = this.shadowRoot?.querySelector<BackendAiDialog>('#task-status-dialog');
    const cmEl = this.shadowRoot?.querySelector('#dockerfile-codemirror') as any;
    if (cmEl) {
      cmEl.setValue(contents);
    }
    this.spinner.hide();
    dialogEl?.show();
  }

  _fetchTaskLogs(taskId: string) {
    return ForkliftUtils.fetch(`/user/my_task/log_result/${taskId}/`, {method: 'GET'}, null, true)
      .then((resp) => {
        return this.logs = typeof(resp.detail.logs) === 'string' ? [{'stream': resp.detail.logs}] : resp.detail.logs;
      });
  }

  _downloadLogFile() {
    let fileName = '';
    ForkliftUtils.fetch(`/download_log_file/${this.taskId}/`, {method: 'GET'}, null, true)
      .then((data) => {
        const disposition = data['disposition'];
        fileName = decodeURI(disposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/)[1].replace(/['"]/g, ''));
        return data['body'].blob();
      })
      .then((blob) => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = fileName;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(blob);
      });
  }

  _showLogDialog(taskId: string) {
    this.spinner.show();
    this.taskId = taskId;
    const logListDialog = this.shadowRoot?.querySelector<BackendAiDialog>('#log-list-dialog');
    this._fetchTaskLogs(taskId);
    this.spinner.hide();
    logListDialog?.show();
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


  /**
   * Render created_at as "YYYY-MM-DD, HH:MM:SS"
   *
   * @param {DOMelement} root
   * @param {object} column (<vaadin-grid-column> element)
   * @param {object} rowData
   */
  _createdAtRenderer(root, column?, rowData?) {
    render(
      html`
        <div>${rowData.item.created_at.replace(/-([^-]*)$/, ', ' + '$1')}</div>
      `, root
    );
  }

  /**
   * Render key control buttons.
   *
   * @param {DOMelement} root
   * @param {object} column (<vaadin-grid-column> element)
   * @param {object} rowData
   */
  controlRenderer(root, column?, rowData?) {
    render(
      html`
        <div id="controls" class="layout horizontal flex center">
          <mwc-icon-button class="fg teal" icon="content_paste_search" fab flat inverted
            @click="${() => this._showRequestedDockerfileDialog(rowData.item.task_id)}"></mwc-icon-button>
          <mwc-icon-button class="fg green" icon="assignment" fab flat inverted
            @click="${() => this._showLogDialog(rowData.item.task_id)}"></mwc-icon-button>
        </div>
      `, root
    );
  }

  _messageRenderer(root, column, rowData) {
    let log;
    let color;
    if (rowData.item.error) {
      log = rowData.item.error + ' ' + rowData.item.errorDetail;
      color = 'red';
    } else {
      log = rowData.item.stream;
      color = 'black';
    }
    render(
      html`
        <span style="color:${color}">${log}</span>
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
          font-size: 14px;
          height: calc(100vh - 235px);
        }

        backend-ai-dialog {
          --component-max-width: 70%;
        }

        #task-status-dialog, #log-list-dialog {
          --component-min-width: 60vw;
        }
      `
    ];
  }

  render() {
    // language=HTML
    return html`
      <lablup-loading-spinner id="loading-spinner"></lablup-loading-spinner>
      <vaadin-grid theme="row-stripes column-borders compact" aria-label="Task list" id="tasks-grid">
        <vaadin-grid-column width="40px" flex-grow="0" header="#" text-align="center" .renderer="${this._indexRenderer.bind(this)}"></vaadin-grid-column>
        <vaadin-grid-column path="task_id" header="ID" resizable></vaadin-grid-column>
        <vaadin-grid-filter-column path="requested_image" header="Source" resizable></vaadin-grid-filter-column>
        <vaadin-grid-filter-column path="requested_target_img" header="Target" resizable></vaadin-grid-filter-column>
        <vaadin-grid-sort-column path="created_at" header="Created at" resizable .renderer="${this._boundCreatedAtRenderer}"></vaadin-grid-sort-column>
        <vaadin-grid-column width="150px" resizable header="Control" .renderer="${this._boundControlRenderer}"></vaadin-grid-column>
      </vaadin-grid>
      <backend-ai-dialog id="task-status-dialog" fixed backdrop>
        <span slot="title">Requested Dockerfile Contents</span>
        <div slot="content" class="layout center">
          <lablup-codemirror id="dockerfile-codemirror" readonly></lablup-codemirror>
        </div>
      </backend-ai-dialog>
      <backend-ai-dialog id="log-list-dialog" fixed backdrop>
        <span slot="title">Logs</span>
        <div slot="content" class="layout center">
          <vaadin-grid id="logs-table" theme="row-stripes column-borders compact wrap-cell-content"
            aria-label="Logs" .items="${this.logs}">
            <vaadin-grid-column width="50px" header="#" flex-grow="0" text-align="end" resizable frozen .renderer="${this._indexRenderer}"></vaadin-grid-column>
            <vaadin-grid-column header="Message" resizable .renderer="${this._boundMessageRenderer}"></vaadin-grid-column>
          </vaadin-grid>
        </div>
        <div slot="footer" class="end-justified layout flex horizontal">
          <mwc-button icon="download" outlined @click="${() => this._downloadLogFile()}">Download</mwc-button>
        </div>
      </backend-ai-dialog>
    `;
  }
}
declare global {
  interface HTMLElementTagNameMap {
    'forklift-task-list': ForkliftTaskList;
  }
}
