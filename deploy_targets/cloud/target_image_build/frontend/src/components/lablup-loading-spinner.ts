/**
 @license
 Copyright (c) 2015-2022 Lablup Inc. All rights reserved.
 */

import {css, CSSResultGroup, html, LitElement} from 'lit';
import {customElement, property} from 'lit/decorators.js';

import 'weightless/progress-spinner';

/**
 Lablup Loading Spinner

 `lablup-loading-spinner` is loading spinner.

 Example:

 <lablup-loading-spinner></lablup-loading-spinner>

@group Backend.AI Web UI
 @element lablup-loading-spinner
 */

@customElement('lablup-loading-spinner')
export default class LablupLoadingSpinner extends LitElement {
  @property({type: Object}) spinner;
  @property({type: Boolean}) active = false;

  constructor() {
    super();
  }

  static get styles(): CSSResultGroup {
    return [
      // language=CSS
      css`
        wl-progress-spinner {
          --progress-spinner-size: 48px;
          --progress-spinner-stroke-width: 12px;
          width: 48px;
          height: 48px;
          position: fixed;
          bottom: 60px;
          right: 60px;
        }
      `];
  }

  render() {
    // language=HTML
    return html`
      <wl-progress-spinner id="spinner"></wl-progress-spinner>
    `;
  }

  shouldUpdate() {
    return this.active;
  }

  firstUpdated() {
    this.spinner = this.shadowRoot?.querySelector('#spinner');
    this.active = true;
  }

  connectedCallback() {
    super.connectedCallback();
  }

  disconnectedCallback() {
    super.disconnectedCallback();
  }

  /**
   * Set up active state and spinner style to show the loading spinner.
   * */
  async show() {
    this.active = true;
    await this.updateComplete;
    this.spinner.style.display = 'block';
  }

  /**
   * Set up active state and spinner style to hide the loading spinner.
   * */
  async hide() {
    this.active = true;
    await this.updateComplete;
    this.spinner.style.display = 'none';
    this.active = false;
  }

  /**
   * Change whether spinner is visible or not.
   * */
  async toggle() {
    await this.updateComplete;
    if (this.spinner.active === true) {
      this.active = true;
      this.spinner.style.display = 'none';
      this.active = false;
    } else {
      this.active = true;
      this.spinner.style.display = 'block';
    }
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lablup-loading-spinner': LablupLoadingSpinner;
  }
}
