/*
 @license
 Copyright (c) 2015-2022 Lablup Inc. All rights reserved.
 */
//import Chart from '../lib/Chart.min';
import Chart from 'chart.js/auto';
import {css, html, LitElement, property, TemplateResult} from 'lit-element';

export default class ChartJs extends LitElement {
  public shadowRoot: any; // ShadowRoot
  @property({type: Object}) data = {};
  @property({type: Object}) options = {};
  @property({type: Object}) chart;
  @property({type: String}) type = 'line';
  @property({type: Number}) height = 0;
  @property({type: Number}) width = 0;

  updated(prop) {
    super.update(prop);
    if (this.chart) {
      this.chart.data = this.data;
      if(!this.options) {
        this.options = {
          responsive: true,
          maintainAspectRatio: true
        }
      }
      this.chart.options = this.options;
      if (typeof this.data !== 'undefined' && typeof this.options !== 'undefined' && this.type !== '' && this.data != {} && this.options != {}) {
        this.updateChart();
      }

    } else {
      if (typeof this.data !== 'undefined' && typeof this.options !== 'undefined' && this.type !== '' && this.data != {} && this.options != {}) {
        this._initializeChart();
      }
    }
  }

  _initializeChart() {
    const ctx: CanvasRenderingContext2D = (this.shadowRoot as any)
      .querySelector('canvas')
      .getContext('2d');
    this.chart = new Chart(ctx, {
      type: this.type as any,
      data: this.data as any,
      options: this.options
    });
  }

  static get styles() {
    return [
      // language=CSS
      css`
        .chart-top-container {
          margin: 0 auto;
          padding: 5px;
          overflow: hidden;
          width: auto;
        }

        .chart-top-container > .chart-sub-container {
          position: relative;
          width: 65vw;
          max-width: 70vw;
          height: 200px;
        }

        @media only screen and (max-width: 1015px) {
          .chart-top-container > .chart-sub-container {
            max-width: 55vw;
          }
        }
      `
    ];
  }

  public firstUpdated(): void {
    if (this.type != '' && this.data != {} && this.options != {}) {
      this._initializeChart();
    }
    if (this.height && this.width) {
      this.shadowRoot.querySelector('.chart-top-container .chart-sub-container').style.height = this.height + 'px';
      this.shadowRoot.querySelector('.chart-top-container .chart-sub-container').style.width = this.width + 'px';
    }
  }

  public render(): void | TemplateResult {
    return html`
      <div class="chart-top-container" style="display:grid;place-items:center;">
        <div class="chart-sub-container">
          <canvas></canvas>
        </div>
      </div>
  `;
  }

  public updateChart = (): void => {
    if (this.chart) {
      this.chart.update();
    }
  }
}
if (!customElements.get('chart-js')) {
  customElements.define('chart-js', ChartJs);
}
