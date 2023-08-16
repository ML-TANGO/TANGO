import {css, html, CSSResultGroup, render} from 'lit';
import {customElement, property} from 'lit/decorators.js';

import {BackendAIPage} from './backend-ai-page';
import {BackendAiStyles} from './backend-ai-general-styles';
import {
  IronFlex,
  IronFlexAlignment,
  IronFlexFactors,
  IronPositioning
} from '../plastics/layout/iron-flex-layout-classes';
import {ForkliftUtils} from './forklift-utils';

import '../plastics/chart-js';
import './lablup-activity-panel';
import './lablup-loading-spinner';
import './forklift-image-list';

import '@material/mwc-button/mwc-button';
import '@material/mwc-icon-button/mwc-icon-button';

import {Grid} from '@vaadin/vaadin-grid';

@customElement('forklift-summary-view')
export default class ForkliftSummaryView extends BackendAIPage {
  @property({type: Array}) images: any[] = [];
  @property({type: Array}) tasks;
  @property({type: Number}) runningTaskCount = 0;
  @property({type: Number}) pendingTaskCount = 0;
  @property({type: Number}) completedTaskCount = 0;
  @property({type: Number}) errorTaskCount = 0;
  @property({type: Object}) options;

  firstUpdated() {
    this.options = {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: {
          display: true,
          position: 'bottom',
          align: 'center',
          labels: {
            fontSize: 20,
            boxWidth: 30,
          }
        }
      }
    };
    if (!ForkliftUtils._checkLogin()) {
      ForkliftUtils._moveTo('/login');
    }
    ForkliftUtils._observeElementVisibility(this, (isVisible) => {
      if (isVisible && ForkliftUtils._checkLogin()) {
        this._fetchImages();
        this._createTaskChart();
      }
    });
  }

  _fetchImages() {
    ForkliftUtils.fetch('/build/image/list/', {method: 'GET'}, null, true)
      .then((data: any) => {
        this.images = [];
        if (data.length > 0) {
          data.map((image) => {
            const [name, tag] = image.split(':');
            this.images.push({name: name, tag: tag});
          });
        }
        (this.shadowRoot?.querySelector('vaadin-grid') as Grid).items = this.images;
      }).catch(() => {
        globalThis.notification.show('Error on retrieving images');
      });
  }

  async _fetchTasks() {
    return await ForkliftUtils.fetch('/user/my_task/all/tasks/', {method: 'GET'}, null, true);
  }

  /**
   * Create Task Doughnut Chart
   */
  async _createTaskChart() {
    const tasks = await this._fetchTasks();
    this.runningTaskCount = tasks.filter((item) => item.status === 'running').length;
    this.pendingTaskCount = tasks.filter((item) => item.status === 'pending').length;
    this.completedTaskCount = tasks.filter((item) => item.status === 'complete').length;
    this.errorTaskCount = tasks.filter((item) => item.status === 'error').length;
    this.tasks = {
      labels: [
        'Running',
        'Pending',
        'Completed',
        'Error',
      ],
      datasets: [{
        data: [
          this.runningTaskCount,
          this.pendingTaskCount,
          this.completedTaskCount,
          this.errorTaskCount,
        ],
        backgroundColor: [
          '#722cd7',
          '#efefef',
          '#60bb43',
          '#ff1744',
        ],
      }],
    };
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

  static get styles(): CSSResultGroup {
    return [
      BackendAiStyles,
      IronFlex,
      IronFlexAlignment,
      IronFlexFactors,
      IronPositioning,
      // language=CSS
      css`
        #go-to-build {
          width: 284px;
          height: var(--component-height, 36px);
        }

        #images-table {
          height: 350px;
        }
        
        .task-chart-wrapper {
          margin: 20px 50px 0px 50px;
        }

        .task-status-indicator {
          width: 90px;
          color: black;
        }

        div.big {
          font-size: 72px;
        }
      `
    ];
  }

  render() {
    return html`
      <link rel="stylesheet" href="resources/fonts/font-awesome-all.min.css">
      <lablup-loading-spinner id="loading-spinner"></lablup-loading-spinner>
      <div class="item" elevation="1">
        <div class="horizontal wrap layout">
          <lablup-activity-panel title="Start Menu" elevation="1" height="450">
            <div slot="message">
              <img src="/resources/images/launcher-background.png" style="width:300px;margin-bottom:30px;"/>
              <div class="horizontal center-justified layout wrap">
                <mwc-button raised class="primary-action" id="go-to-build"
                  icon="power_settings_new"
                  @click="${() => ForkliftUtils._moveTo('/builder')}">Build</mwc-button>
              </div>
            </div>
          </lablup-activity-panel>
          <lablup-activity-panel title="Tasks" elevation="1" horizontalsize="2x" height="450">
            <div slot="right-header">
              <mwc-button @click="${() => ForkliftUtils._moveTo('/tasks')}" icon="chevron_right">See Details</mwc-button>
            </div>
            <div slot="message">
              <div class="horizontal layout wrap flex center center-justified">
                <div class="task-chart-wrapper">
                  <chart-js id="task-status" type="doughnut" .data="${this.tasks}" .options="${this.options}" height="250" width="250"></chart-js>
                </div>
                <div class="vertical layout center">
                  <div class="horizontal layout justified">
                    <div class="vertical layout center task-status-indicator">
                      <div class="big">${this.runningTaskCount}</div>
                      <span>Running</span>
                    </div>
                    <div class="vertical layout center task-status-indicator">
                      <div class="big">${this.pendingTaskCount}</div>
                      <span>Pending</span>
                    </div>
                  </div>
                  <div class="horizontal layout justified">
                    <div class="vertical layout center task-status-indicator">
                      <div class="big">${this.completedTaskCount}</div>
                      <span>Completed</span>
                    </div>
                    <div class="vertical layout center task-status-indicator">
                      <div class="big">${this.errorTaskCount}</div>
                      <span>Error</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </lablup-activity-panel>
          <lablup-activity-panel title="Installed Images" elevation="1" horizontalsize="3x" height="450">
            <div slot="right-header">
              <mwc-button @click="${() => ForkliftUtils._moveTo('/environments')}" icon="chevron_right">See Details</mwc-button>
            </div>
            <div slot="message">
              <vaadin-grid theme="row-stripes column-borders" aria-label="images" id="images-table" class="tab-content" .items="${this.images}">
                <vaadin-grid-column width="45px" header="#" flex-grow="0" text-align="end" .renderer="${this._indexRenderer}"></vaadin-grid-column>
                <vaadin-grid-filter-column path="name" header="Name" resizable></vaadin-grid-filter-column>
                <vaadin-grid-filter-column path="tag" header="Tag" resizable></vaadin-grid-filter-column>
              </vaadin-grid>
            </div>
          </lablup-activity-panel>
        </div>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'forklift-summary-view': ForkliftSummaryView;
  }
}
