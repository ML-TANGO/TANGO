import {css, CSSResultGroup, html, render} from 'lit';
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

import BackendAiDialog from './backend-ai-dialog';
import './lablup-loading-spinner';
import './lablup-codemirror';

import '@material/mwc-list/mwc-list';
import '@material/mwc-list/mwc-list-item';
import '@material/mwc-icon-button/mwc-icon-button';
import {Checkbox} from '@material/mwc-checkbox/mwc-checkbox';

import '@vaadin/vaadin-grid/vaadin-grid-column';
import '@vaadin/vaadin-grid/vaadin-grid-filter-column';
import '@vaadin/vaadin-grid/vaadin-grid';
import {Grid} from '@vaadin/vaadin-grid/vaadin-grid';

@customElement('forklift-image-list')
export default class ForkliftEnvironmentList extends BackendAIPage {
  @property({type: Array}) images: any[] = [];
  @property({type: Object}) imageInfo = Object();
  @property({type: Object}) _boundControlsRenderer = this._controlsRenderer.bind(this);

  firstUpdated() {
    if (!ForkliftUtils._checkLogin()) {
      ForkliftUtils._moveTo('/login');
    }
    ForkliftUtils._observeElementVisibility(this, (isVisible) => {
      if (isVisible && ForkliftUtils._checkLogin()) {
        this._fetchImages();
      }
    });
    document.addEventListener('refresh-image-list', () => this._fetchImages());
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
        const grid = this.shadowRoot?.querySelector('#images-table') as Grid;
        grid.items = this.images;
        grid.clearCache();
      });
  }

  _clearImageWithCheck(e) {
    const force = this.shadowRoot?.querySelector<Checkbox>('#force-button')?.checked;
    const url = `/image/?target_image_name=${this.imageInfo.name}:${this.imageInfo.tag}&force=${force}`;
    ForkliftUtils.fetch(url, {method: 'DELETE'}, null, true)
      .then((resp) => {
        globalThis.notification.show('Successfully delete an image.');
        this._fetchImages();
      })
      .catch(() => {
        globalThis.notification.show('Error on deleting an image.');
      });
    this._hideDialog(e);
  }

  _fetchImageInfo() {
    const url = `/build/image/info/?target_image_name=${this.imageInfo.name}:${this.imageInfo.tag}`;
    return ForkliftUtils.fetch(url, {method: 'GET'}, null, true)
      .then((resp) => {
        return JSON.stringify(resp, null, 2);
      });
  }

  /**
   * Inform about image using dialog.
   *
   * @param {Event} e - Dispatches from the native input event each time the input changes.
   * */
  async _showImageInfoDialog(e: any) {
    const controls = e.target.closest('#controls');
    this.imageInfo.name = controls['name'];
    this.imageInfo.tag = controls['tag'];
    (this.shadowRoot?.querySelector('#image-name') as HTMLSpanElement).innerHTML = this.imageInfo.name;
    (this.shadowRoot?.querySelector('#image-tag') as HTMLSpanElement).innerHTML = this.imageInfo.tag;
    try {
      const metadata = await this._fetchImageInfo();
      const cmEl = this.shadowRoot?.querySelector('#image-metadata-codemirror') as any;
      this.imageInfo.metadata = metadata;
      if (cmEl) {
        cmEl.setValue(metadata);
      }
      this.shadowRoot?.querySelector<BackendAiDialog>('#image-info-dialog')?.show();
    } catch (err) {
      globalThis.notification.show(err);
    }
  }

  _showClearImageDialog(e: any) {
    const controls = e.target.closest('#controls');
    this.imageInfo.name = controls['name'];
    this.imageInfo.tag = controls['tag'];
    (this.shadowRoot?.querySelector('#delete-image-name') as HTMLElement).innerText = this.imageInfo.name;
    this.shadowRoot?.querySelector<BackendAiDialog>('#delete-image-dialog')?.show();
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

  /**
   * Render controllers.
   *
   * @param {DOMelement} root
   * @param {object} column (<vaadin-grid-column> element)
   * @param {object} rowData
   */
  _controlsRenderer(root, column?, rowData?) {
    render(
      html`
        <div id="controls" class="layout horizontal flex center"
          .name="${rowData.item.name}" .tag="${rowData.item.tag}">
          <mwc-icon-button icon="info" class="fg green controls-running"
            @click="${(e) => this._showImageInfoDialog(e)}"></mwc-icon-button>
          <mwc-icon-button icon="delete_forever" class="fg red controls-running"
            @click="${(e) => this._showClearImageDialog(e)}"></mwc-icon-button>
        </div>
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
        vaadin-grid {
          font-size: 14px;
          height: calc(100vh - 235px);
        }

        #image-info-dialog {
          --component-max-width: 60vw;
        }
      `
    ];
  }

  // Render the UI as a function of component state
  render() {
    return html`
      <lablup-loading-spinner id="loading-spinner"></lablup-loading-spinner>
      <vaadin-grid theme="row-stripes column-borders compact" aria-label="images" id="images-table" .items="${this.images}">
        <vaadin-grid-column width="40px" header="#" flex-grow="0" text-align="end" .renderer="${this._indexRenderer}"></vaadin-grid-column>
        <vaadin-grid-filter-column path="name" header="Name" resizable></vaadin-grid-filter-column>
        <vaadin-grid-filter-column path="tag" header="Tag" resizable></vaadin-grid-filter-column>
        <vaadin-grid-column width="120px" flex-grow="0" resizable header="Controls" .renderer="${this._boundControlsRenderer}"></vaadin-grid-column>
      </vaadin-grid>
      <backend-ai-dialog id="image-info-dialog" fixed backdrop>
        <span slot="title">Image Inspect</span>
        <div slot="content" role="listbox" style="margin:0;">
          <mwc-list>
            <mwc-list-item twoline>
              <span><strong>Image name</strong></span>
              <span id="image-name" class="monospace" slot="secondary"></span>
            </mwc-list-item>
            <mwc-list-item twoline>
              <span><strong>Tag</strong></span>
              <span id="image-tag" class="monospace" slot="secondary"></span>
            </mwc-list-item>
            <mwc-list-item>
              <span><strong>Metadata</strong></span>
            </mwc-list-item>
          </mwc-list>
          <lablup-codemirror id="image-metadata-codemirror" readonly></lablup-codemirror>
        </div>
      </backend-ai-dialog>
      <backend-ai-dialog id="delete-image-dialog" fixed backdrop blockscrolling>
        <span slot="title">Let's Double-check</span>
        <div slot="content">
          <p>You are about to delete this image:</p>
          <p id="delete-image-name" style="text-align:center;"></p>
          <div class="horizontal layout">
            <mwc-checkbox id="force-button"></mwc-checkbox>
            <p>Remove the image even if it is being used by stopped containers or has other tags.</p>
          </div>
          <p>WARNING: this cannot be undone! Do you want to proceed?</p>
        </div>
        <div slot="footer" class="horizontal end-justified flex layout">
          <mwc-button
              class="operation"
              label="Cancel"
              @click="${(e) => this._hideDialog(e)}"></mwc-button>
          <mwc-button
              unelevated
              class="operation"
              label="Okay"
              @click="${(e) => this._clearImageWithCheck(e)}"></mwc-button>
        </div>
      </backend-ai-dialog>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'forklift-image-list': ForkliftEnvironmentList;
  }
}
