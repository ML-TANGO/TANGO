import {css, CSSResultGroup, html} from 'lit';
import {customElement, property} from 'lit/decorators.js';

import {BackendAIPage} from './backend-ai-page';
import {BackendAiStyles} from './backend-ai-general-styles';
import {IronFlex, IronFlexAlignment} from '../plastics/layout/iron-flex-layout-classes';

import '@material/mwc-tab/mwc-tab';
import '@material/mwc-tab-bar/mwc-tab-bar';
import '@material/mwc-button/mwc-button';

import './lablup-activity-panel';
import './forklift-preset-build-view';
import './forklift-custom-build-view';

import {TabBar} from '@material/mwc-tab-bar/mwc-tab-bar';

@customElement('forklift-build-view')
export default class ForkliftBuildView extends BackendAIPage {
  @property({type: String}) _activeTab = 'preset-build';

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

  static get styles(): CSSResultGroup | undefined {
    return [
      BackendAiStyles,
      IronFlex,
      IronFlexAlignment,
      // language=CSS
      css`
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

        .tab-content {
          height: calc(100vh - 189px);
        }
        
        @media screen and (max-width: 805px) {
          mwc-tab, mwc-button {
            --mdc-typography-button-font-size: 10px;
          }
        }
      `
    ];
  }

  render() {
    return html`
      <lablup-activity-panel elevation="1" noheader narrow autowidth>
        <div slot="message">
          <h3 class="tab horizontal center layout">
            <mwc-tab-bar>
              <mwc-tab title="preset-build" label="Preset" @click="${(e) => this._showTab(e.target)}"></mwc-tab>
              <mwc-tab title="custom-build" label="Custom" @click="${(e) => this._showTab(e.target)}"></mwc-tab>
            </mwc-tab-bar>
          </h3>
          <div id="preset-build" class="item tab-content">
            <forklift-preset-build-view ?active="${this._activeTab === 'preset-build'}"></forklift-preset-build-view>
          </div>
          <div id="custom-build" class="item tab-content">
            <forklift-custom-build-view ?active="${this._activeTab === 'custom-build'}"></forklift-custom-build-view>
          </div>
        </div>
      </lablup-activity-panel>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'forklift-build-view': ForkliftBuildView;
  }
}
