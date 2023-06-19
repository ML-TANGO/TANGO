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

import './lablup-activity-panel';
import './forklift-log-list';
import './forklift-task-list';
import ForkliftTaskList from './forklift-task-list';

import '@material/mwc-tab-bar/mwc-tab-bar';
import '@material/mwc-button/mwc-button';

import 'weightless/tab';
import 'weightless/tab-group';
import {Tab} from 'weightless/tab';

@customElement('forklift-tasks-view')
export default class ForkliftTasksView extends BackendAIPage {
  @property({type: String}) _activeTab = 'task-lists';
  @property({type: String}) _status = 'inactive';

  /**
   * Display the tab.
   *
   * @param {any} tab - Tab webcomponent
   */
  _showTab(tab) {
    const els = this.shadowRoot?.querySelectorAll<HTMLDivElement>('.tab-content');
    for (let x = 0; x < els!.length; x++) {
      els![x].style.display = 'none';
    }
    this._activeTab = tab.title;
    (this.shadowRoot?.querySelector('#' + tab.title) as HTMLDivElement).style.display = 'block';
    let tabKeyword;
    let innerTab;
    // show inner tab(active) after selecting outer tab
    switch (this._activeTab) {
    case 'task-lists':
      tabKeyword = this._activeTab.substring(0, this._activeTab.length - 1); // to remove '-s'.
      innerTab = this.shadowRoot?.querySelector('wl-tab[value=running-' + tabKeyword + ']');
      innerTab.checked = true;
      this._showList(innerTab);
      break;
    default:
      break;
    }
  }

  /**
   * Display the list.
   *
   * @param {any} list - List webcomponent
   */
  _showList(list) {
    const els = this.shadowRoot?.querySelectorAll<ForkliftTaskList>('.list-content');
    for (let x = 0; x < els!.length; x++) {
      els![x].style.display = 'none';
    }
    (this.shadowRoot?.querySelector('#' + list.value) as Tab).style.display = 'block';
  }

  refreshLogData() {
    const event = new CustomEvent('refresh-log-data');
    document.dispatchEvent(event);
  }

  showClearLogsDialog() {
    const event = new CustomEvent('show-clear-logs-dialog');
    document.dispatchEvent(event);
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
        h3.tab {
          background-color: var(--general-tabbar-background-color);
          border-radius: 5px 5px 0 0;
          margin: 0 auto;
        }

        div.card > h4 {
          margin-bottom: 0px;
        }

        div.card h3 {
          padding-top: 0;
          padding-right: 15px;
          padding-bottom: 0;
        }

        div.card div.card {
          margin: 0;
          padding: 0;
          --card-elevation: 0;
        }

        mwc-tab-bar {
          --mdc-theme-primary: var(--general-sidebar-selected-color);
          --mdc-text-transform: none;
          --mdc-tab-color-default: var(--general-tabbar-background-color);
          --mdc-tab-text-label-color-default: var(--general-tabbar-tab-disabled-color);
        }

        wl-tab-group {
          border-radius: 5px 5px 0 0;
          --tab-group-indicator-bg: var(--general-tabbar-button-color);
        }

        wl-tab {
          border-radius: 5px 5px 0 0;
          --tab-color: var(--general-sidepanel-color);
          --tab-color-hover: #26272a;
          --tab-color-hover-filled: var(--general-tabbar-button-color);
          --tab-color-active:var(--general-tabbar-button-color);
          --tab-color-active-hover: var(--general-tabbar-button-color);
          --tab-color-active-filled: var(--general-tabbar-button-color);
          --tab-bg-active: #535457;
          --tab-bg-filled: #26272a;
          --tab-bg-active-hover: #535457;
        }

        @media screen and (max-width: 805px) {
          mwc-tab, mwc-button {
            --mdc-typography-button-font-size: 10px;
          }

          wl-tab {
            width: 5px;
          }
        }
      `
    ];
  }

  render() {
    // language=HTML
    return html`
      <lablup-activity-panel noheader narrow autowidth>
        <div slot="message">
          <h3 class="tab horizontal wrap layout">
            <mwc-tab-bar>
              <mwc-tab title="task-lists" label="Task" @click="${(e) => this._showTab(e.target)}"></mwc-tab>
              <mwc-tab title="log-lists" label="Logs" @click="${(e) => this._showTab(e.target)}"></mwc-tab>
            </mwc-tab-bar>
          </h3>
          <div id="task-lists" class="item tab-content card">
            <h4 class="horizontal flex center center-justified layout">
              <wl-tab-group style="margin-bottom:-8px;">
                <wl-tab value="running-task-list" checked @click="${(e) => this._showList(e.target)}">Running</wl-tab>
                <wl-tab value="pending-task-list" @click="${(e) => this._showList(e.target)}">Pending</wl-tab>
                <wl-tab value="complete-task-list" @click="${(e) => this._showList(e.target)}">Completed</wl-tab>
                <wl-tab value="error-task-list" @click="${(e) => this._showList(e.target)}">Error</wl-tab>
              </wl-tab-group>
              <span class="flex"></span>
            </h4>
            <div>
              <forklift-task-list class="list-content" id="running-task-list" condition="running" ?active="${this._activeTab === 'task-lists'}"></forklift-task-list>
              <forklift-task-list class="list-content" id="pending-task-list" style="display:none;" condition="pending" ?active="${this._activeTab === 'task-lists'}"></forklift-task-list>
              <forklift-task-list class="list-content" id="complete-task-list" style="display:none;" condition="complete" ?active="${this._activeTab === 'task-lists'}"></forklift-task-list>
              <forklift-task-list class="list-content" id="error-task-list" style="display:none;" condition="error" ?active="${this._activeTab === 'task-lists'}"></forklift-task-list>
            </div>
          </div>
          <div id="log-lists" class="tab-content card" style="display:none;">
            <h4 class="horizontal flex center center-justified layout">
              <span>Logs of images being built</span>
              <span class="flex"></span>
              <mwc-button raised id="refresh" icon="refresh" style="margin-right:15px;"
                @click="${() => this.refreshLogData()}">
                <span>Refresh</span>
              </mwc-button>
              <mwc-button raised id="delete" icon="delete"
                @click="${() => this.showClearLogsDialog()}">
                <span>Clear logs</span>
              </mwc-button>
            </h4>
            <div>
              <forklift-log-list id="log-list" ?active="${this._activeTab === 'log-lists'}"></forklift-log-list>
            </div>
          </div>
        </div>
      </lablup-activity-panel>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'forklift-tasks-view': ForkliftTasksView;
  }
}
