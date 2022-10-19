import {css, CSSResultGroup, html} from 'lit';
import {customElement, property} from 'lit/decorators.js';

import {BackendAIPage} from './backend-ai-page';
import {BackendAiStyles} from './backend-ai-general-styles';
import {IronFlex, IronFlexAlignment} from '../plastics/layout/iron-flex-layout-classes';

import './lablup-activity-panel';
import './forklift-image-list';
import './forklift-container-list';

import {TabBar} from '@material/mwc-tab-bar/mwc-tab-bar';
import {Button} from '@material/mwc-button/mwc-button';

@customElement('forklift-environment-view')
export default class ForkliftEnvironmentView extends BackendAIPage {
  @property({type: String}) _activeTab = 'image-lists';

  firstUpdated() {
    let evt = new Event('refresh-image-list');
    (this.shadowRoot?.querySelector('#images-refresh') as Button).addEventListener('click', () => document.dispatchEvent(evt));
    evt = new Event('refresh-container-list');
    (this.shadowRoot?.querySelector('#containers-refresh') as Button).addEventListener('click', () => document.dispatchEvent(evt));
  }

  /**
   * Display the tab.
   *
   * @param {any} tab - tab webcomponent that has 'title' property
   */
  _showTab(tab) {
    const els = this.shadowRoot?.querySelectorAll<HTMLDivElement>('.tab-content');
    for (let x = 0; x < els!.length; x++) {
      els![x].style.display = 'none';
    }
    this._activeTab = tab.title;
    (this.shadowRoot?.querySelector('#' + tab.title) as TabBar).style.display = 'block';
  }

  static get styles(): CSSResultGroup {
    return [
      BackendAiStyles,
      IronFlex,
      IronFlexAlignment,
      // language=CSS
      css`
        h4 {
          font-weight: 200;
          font-size: 14px;
          margin: 0px;
          padding: 5px 15px 5px 20px;
        }

        h3.tab {
          background-color: var(--general-tabbar-background-color);
          border-radius: 5px 5px 0px 0px;
          margin: 0px auto;
        }

        mwc-tab-bar {
          --mdc-theme-primary: var(--general-sidebar-selected-color);
          --mdc-text-transform: none;
          --mdc-tab-color-default: var(--general-tabbar-background-color);
          --mdc-tab-text-label-color-default: var(--general-tabbar-tab-disabled-color);
        }
        
        @media screen and (max-width: 805px) {
          mwc-tab, mwc-button {
            --mdc-typography-button-font-size: 10px;
          }
        }
      `
    ];
  }

  // Render the UI as a function of component state
  render() {
    // language=HTML
    return html`
      <lablup-activity-panel elevation="1" noheader narrow autowidth>
        <div slot="message">
          <h3 class="tab horizontal center layout">
            <mwc-tab-bar>
              <mwc-tab title="image-lists" label="Images" @click="${(e) => this._showTab(e.target)}"></mwc-tab>
              <mwc-tab title="container-lists" label="Containers" @click="${(e) => this._showTab(e.target)}"></mwc-tab>
            </mwc-tab-bar>
            <div class="flex"></div>
          </h3>
          <div id="image-lists" class="tab-content">
            <h4 class="horizontal flex center center-justified layout">
              <span>Images</span>
              <span class="flex"></span>
              <mwc-button raised id="images-refresh" icon="refresh" style="margin-right:15px;">
                <span>Refresh</span>
              </mwc-button>
            </h4>
            <forklift-image-list ?active="${this._activeTab === 'image-lists'}"></forklift-image-list>
          </div>
          <div id="container-lists" class="tab-content" style="display:none;">
            <h4 class="horizontal flex center center-justified layout">
                <span>Running containers</span>
                <span class="flex"></span>
                <mwc-button raised id="containers-refresh" icon="refresh" style="margin-right:15px;">
                  <span>Refresh</span>
                </mwc-button>
            </h4>
            <forklift-container-list ?active="${this._activeTab === 'container-lists'}"></forklift-container-list>
          </div>
        </div>
      </lablup-activity-panel>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'forklift-environment-view': ForkliftEnvironmentView;
  }
}
