/**
 @license
 Copyright (c) 2015-2022 Lablup Inc. All rights reserved.
 */
import {css, CSSResultGroup, html, LitElement} from 'lit';
import {customElement} from 'lit/decorators.js';

/**
 Backend.AI Offline Indicator

 Example:

 <backend-ai-offline-indicator ?active="${0}"></backend-ai-offline-indicator>

@group Backend.AI Web UI
 @element backend-ai-offline-indicator
 */

@customElement('backend-ai-offline-indicator')
export default class BackendAIOfflineIndicator extends LitElement {
  static get styles(): CSSResultGroup {
    return [
      css`
        :host {
          display: block;
          position: fixed;
          top: 100%;
          left: 0;
          right: 0;
          padding: 12px;
          background-color: #246;
          color: white;
          box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
          text-align: center;
          will-change: transform;
          transform: translate3d(0, 0, 0);
          transition-property: visibility, transform;
          transition-duration: 0.2s;
          visibility: hidden;
        }

        :host([active]) {
          visibility: visible;
          transform: translate3d(0, -100%, 0);
        }

        @media (min-width: 460px) {
          :host {
            width: 320px;
            margin: auto;
          }
        }
      `
    ];
  }

  render() {
    return html`
      <slot></slot>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'backend-ai-offline-indicator': BackendAIOfflineIndicator;
  }
}
