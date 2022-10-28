import {css, CSSResultGroup, html, LitElement} from 'lit';
import {customElement, property} from 'lit/decorators.js';

import {IronFlex, IronFlexAlignment} from '../plastics/layout/iron-flex-layout-classes';
import {ForkliftStyles} from './forklift-styles';

import '../plastics/lablup-shields/lablup-shields';

import '@material/mwc-icon/mwc-icon';

@customElement('forklift-preset-panel')
export default class ForkliftPresetPanel extends LitElement {
  public shadowRoot: any; // ShadowRoot
  @property({type: String}) icon;
  @property({type: String}) shieldColor;
  @property({type: String}) shieldDescription;
  @property({type: String}) title;
  @property({type: String}) subtitle;
  @property({type: String}) horizontalsize = '';
  @property({type: Boolean}) autowidth = false;
  @property({type: Number}) width = 380;
  @property({type: Number}) widthpct = 0;
  @property({type: Number}) height = 0;
  @property({type: Number}) marginWidth = 14;
  @property({type: Number}) minwidth = 0;
  @property({type: Number}) maxwidth = 0;
  @property({type: Boolean}) narrow = false;
  @property({type: Boolean}) scrollableY = false;

  firstUpdated() {
    if (this.autowidth) {
      (this.shadowRoot.querySelector('.card') as any).style.width = 'auto';
    } else {
      (this.shadowRoot.querySelector('.card') as any).style.width = this.widthpct !== 0 ? this.widthpct + '%' : this.width + 'px';
    }

    if (this.minwidth) {
      (this.shadowRoot.querySelector('.card') as any).style.minWidth = this.minwidth + 'px';
    }
    if (this.maxwidth) {
      (this.shadowRoot.querySelector('.card') as any).style.minWidth = this.maxwidth + 'px';
    }
    if (this.horizontalsize) {
      if (this.horizontalsize == '2x') {
        (this.shadowRoot.querySelector('.card') as any).style.width = (this.width * 2 + 28) + 'px';
      }
      if (this.horizontalsize == '3x') {
        (this.shadowRoot.querySelector('.card') as any).style.width = (this.width * 3 + 56) + 'px';
      }
      if (this.horizontalsize == '4x') {
        (this.shadowRoot.querySelector('.card') as any).style.width = (this.width * 4 + 84) + 'px';
      }
    }
    (this.shadowRoot.querySelector('.card') as any).style.margin = this.marginWidth + 'px';
    if (this.narrow === true) {
      this.shadowRoot.querySelector('div.card > div').style.margin = '0';
      this.shadowRoot.querySelector('div.card > h4').style.marginBottom = '0';
    }
    if (this.height > 0) {
      this.height == 130 ?
        this.shadowRoot.querySelector('div.card').style.height = 'fit-content' :
        this.shadowRoot.querySelector('div.card').style.height = this.height + 'px';
    }
    if (this.scrollableY) {
      this.shadowRoot.querySelector('.card').style.overflowY = 'auto';
    }
  }

  static get styles(): CSSResultGroup | undefined {
    return [
      ForkliftStyles,
      IronFlex,
      IronFlexAlignment,
      // language=CSS
      css`
        div.card {
          display: block;
          background: var(--card-background-color, #ffffff);
          box-sizing: border-box;
          margin: 14px;
          border-radius: 5px;
          box-shadow: rgba(4, 7, 22, 0.7) 0px 0px 4px -2px;
          width: 280px;
        }

        #title-container {
          padding: 20px;
          margin: 20px;
          margin-bottom: 0px;
        }

        #message-container {
          font-size: 12px;
          overflow-wrap: break-word;
        }

        #title {
          margin-bottom: 10px;
          font-size: var(--title-font-size, 20px);
          font-weight: var(--title-font-weight, 500);
          font-family: var(--general-font-family)
        }

        #subtitle {
          color: var(--subtitle-font-color, #C1C1D1);
        }

        .title-divider {
          height: 4px;
          border-radius: 5px;
          margin: 20px auto;
          background: var(--line-color, linear-gradient(to left, #18aa7c, #60bb43));
        }

        .divider {
          height: 1px;
          background: #dddddd;
        }
      `
    ];
  }

  render() {
    // language=HTML
    return html`
      <div class="card">
        <div id="title-container">
          <h4 id="title">${this.title}</h4>
          <span id="subtitle">${this.subtitle}</span>
          <div class="title-divider"></div>
          <lablup-shields color="${this.shieldColor}" description="${this.shieldDescription}" ui="round"></lablup-shields>
        </div>
        <div id="message-container">
          <slot name="message"></slot>
        </div>
      </div>
    `;
  }
}
declare global {
  interface HTMLElementTagNameMap {
    'forklift-preset-panel': ForkliftPresetPanel;
  }
}
