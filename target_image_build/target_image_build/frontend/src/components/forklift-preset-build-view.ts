import {css, CSSResultGroup, html} from 'lit';
import {customElement, property} from 'lit/decorators.js';

import {BackendAIPage} from './backend-ai-page';
import {IronFlex, IronFlexAlignment} from '../plastics/layout/iron-flex-layout-classes';
import {ForkliftStyles} from './forklift-styles';
import {ForkliftUtils} from './forklift-utils';

import '@material/mwc-button/mwc-button';
import '@material/mwc-icon-button/mwc-icon-button';
import '@material/mwc-textfield/mwc-textfield';

import './lablup-activity-panel';
import './lablup-loading-spinner';
import './forklift-preset-panel';

@customElement('forklift-preset-build-view')
export default class ForkliftPresetBuildView extends BackendAIPage {
  @property({type: Object}) model;

  firstUpdated() {
  }

  build(e, run_container = false) {
    const presetPanel = e.target.closest('forklift-preset-panel');
    const body = {
      target: presetPanel.querySelector('.target')?.value,
      runtime_type: 'preset',
      auto_push: false,
      preset: presetPanel.id,
      run_container: run_container,
    };
    if (body.target !== '') {
      ForkliftUtils.fetch('/build/submit/preset/', {method: 'POST', body: JSON.stringify(body)}, null, true)
        .then((resp) => {
          globalThis.notification.show(resp.detail);
        }).catch((err) => {
          console.log(e);
          const msg = 'Error on building an image.';
          globalThis.notification.show(msg);
        });
    } else {
      globalThis.notification.show('Please input target docker image field.');
    }
  }

  static get styles(): CSSResultGroup | undefined {
    return [
      ForkliftStyles,
      IronFlex,
      IronFlexAlignment,
      // language=CSS
      css`
        mwc-button {
          margin: 10px;
        }

        span {
          font-size: 13px;
        }

        #transformer {
          --line-color: linear-gradient(to left, red, orange);
        }

        .padding {
          padding: 20px;
          padding-top: 10px;
        }
      `
    ];
  }

  render() {
    // language=HTML
    return html`
      <link rel="stylesheet" href="/resources/fonts/font-awesome-all.min.css">
      <lablup-loading-spinner id="loading-spinner"></lablup-loading-spinner>
      <div class="horizontal layout">
        <forklift-preset-panel id="yolov5" shieldColor="green" shieldDescription="Object Detection"
          title="YOLOv5" subtitle="You Only Look Once">
          <div slot="message">
            <div class="horizontal center justified layout padding">
              <mwc-button raised icon="miscellaneous_services" @click="${(e) => this.build(e)}">Build</mwc-button>
              <mwc-button outlined icon="play_arrow" @click="${(e) => this.build(e, true)}">Build and Run Container</mwc-button>
            </div>
          </div>
        </forklift-preset-panel>
        <!-- <forklift-preset-panel id="transformer" shieldColor="orange" shieldDescription="Seq2Seq"
          title="Transformer" subtitle="#sequential-processing #self-attention">
          <div slot="message">
            <div class="horizontal center justified layout padding">
              <span>Target docker image</span>
              <mwc-textfield class="target"></mwc-textfield>
            </div>
            <div class="horizontal center justified layout padding">
              <mwc-button raised icon="miscellaneous_services" @click="${(e) => this.build(e)}">Build</mwc-button>
              <mwc-button outlined icon="play_arrow" @click="${(e) => this.build(e, true)}">Build and Run Container</mwc-button>
            </div>
          </div>
        </forklift-preset-panel> -->
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'forklift-preset-build-view': ForkliftPresetBuildView;
  }
}
