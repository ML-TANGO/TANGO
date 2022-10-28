import {css, CSSResultGroup, html, LitElement} from 'lit';
import {customElement, property} from 'lit/decorators.js';

import {BackendAiStyles} from './backend-ai-general-styles';
import {IronFlex, IronFlexAlignment} from '../plastics/layout/iron-flex-layout-classes';

import {IconButtonToggle} from '@material/mwc-icon-button-toggle/mwc-icon-button-toggle';

/**
 Backend.AI Forklift Collapse

 Example:

 <forklift-collapse>
 ...
 </forklift-collapse>

@group Backend.AI Forklift
@element forklift-collapse
 */
@customElement('forklift-collapse')
export default class ForkliftExpansion extends LitElement {
  @property({type: Boolean}) floating = false;
  @property({type: Boolean}) scrollInto = false;

  toggle() {
    const content = this.shadowRoot?.querySelector('#content-container') as HTMLDivElement;
    const toggleIcon = this.shadowRoot?.querySelector('#toggle-icon') as IconButtonToggle;
    let isOn;
    if (this.floating) {
      if (content.style.visibility === 'hidden') {
        isOn = true;
        content.setAttribute('style', 'visibility:visible;position:absolute;opacity:1;');
      } else {
        isOn = false;
        content.setAttribute('style', 'visibility:hidden;position:absolute;opacity:0;');
      }
    } else {
      if (content.style.display === 'none') {
        isOn = true;
        content.setAttribute('style', 'display:grid;opacity:1;');
      } else {
        isOn = false;
        content.setAttribute('style', 'display:none;opacity:0;');
      }
    }

    if (isOn) {
      toggleIcon.removeAttribute('off');
      toggleIcon.setAttribute('on', '');
      if (this.scrollInto) {
        (this.shadowRoot?.querySelector('#collapse') as HTMLDivElement).scrollIntoView({behavior: "smooth", inline: "nearest"});
      }
    } else {
      toggleIcon.removeAttribute('on');
      toggleIcon.setAttribute('off', '');
    }
  }

  static get styles(): CSSResultGroup {
    return [
      BackendAiStyles,
      IronFlex,
      IronFlexAlignment,
      // language=CSS
      css`
        #collapse {
          width: 100%;
        }

        #title-container {
          border: 0;
          outline: 0;
          background-color: var(--component-title-color, transparent);
          padding-left: 18px;
          padding-right: 0px;
          height: var(--component-height, 56px);
          width: var(--component-width, 100%);
          border-bottom: 1px solid var(--general-border-color);
        }

        #title-container:hover {
          background-color: whitesmoke;
        }

        #title-container:focus, #title-container:active {
          background-color: #ececec;
          border-bottom: 2px solid var(--general-border-active-color);
        }

        #toggle-icon {
          color: #747474;
          pointer-events: none;
        }

        #content-container {
          left: 0px;
          right: 0px;
          top: calc(var(--component-height, 56px) + 2px);
          padding: 15px;
          z-index: 1;
          color: var(--component-content-color, #000);
          background-color: var(--component-content-background-color, #fff);
          box-shadow: rgb(0 0 0 / 20%) 0px 5px 5px -3px, rgb(0 0 0 / 14%) 0px 8px 10px 1px, rgb(0 0 0 / 12%) 0px 3px 14px 2px;
          opacity: 0;
          transition: opacity ease-in 0.05s;
          transition-property: opacity;
        }
      `
    ];
  }

  render() {
    // language=HTML
    return html`
      <div id="collapse" style="${this.floating ? 'position:relative;display:inline-block;' : 'display:grid;'}">
        <button id="title-container" class="horizontal layout center" @click="${() => this.toggle()}">
          <slot name="title"></slot>
          <span class="flex"></span>
          <mwc-icon-button-toggle id="toggle-icon" onIcon="arrow_drop_up" offIcon="arrow_drop_down"></mwc-icon-button-toggle>
        </button>
        <div style="${this.floating ? 'visibility:hidden;position:absolute;' : 'display:none;grid-area:auto;'}" id="content-container" class="center layout">
          <slot name="content"></slot>
        </div>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'forklift-collapse': ForkliftExpansion;
  }
}
