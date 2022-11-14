import {css, CSSResultGroup, html, render} from 'lit';
import {customElement, property} from 'lit/decorators.js';

import {BackendAIPage} from './backend-ai-page';
import {ForkliftStyles} from './forklift-styles';
import {IronFlex, IronFlexAlignment} from '../plastics/layout/iron-flex-layout-classes';
import {ForkliftUtils} from './forklift-utils';

import '@material/mwc-icon-button/mwc-icon-button';
import {TextField} from '@material/mwc-textfield/mwc-textfield';
import {Switch} from '@material/mwc-switch/mwc-switch';

import '@vaadin/vaadin-grid/vaadin-grid';
import '@vaadin/vaadin-grid/vaadin-grid-column';
import '@vaadin/vaadin-grid/vaadin-grid-sort-column';
import '@vaadin/vaadin-grid/vaadin-grid-filter-column';
import {Grid} from '@vaadin/vaadin-grid/vaadin-grid';

import BackendAiDialog from './backend-ai-dialog';
import './lablup-codemirror';

@customElement('forklift-container-list')
export default class ForkliftContainerList extends BackendAIPage {
  @property({type: Array}) containers: any[] = [];
  @property({type: String}) containerId;
  @property({type: Object}) container;
  @property({type: Object}) _boundCreatedAtRenderer = this._createdAtRenderer.bind(this);
  @property({type: Object}) _boundControlsRenderer = this._controlsRenderer.bind(this);

  firstUpdated() {
    if (!ForkliftUtils._checkLogin()) {
      ForkliftUtils._moveTo('/login');
    }
    ForkliftUtils._observeElementVisibility(this, (isVisible) => {
      if (isVisible && ForkliftUtils._checkLogin()) {
        this._fetchContainers();
      }
    });
    document.addEventListener('refresh-container-list', () => this._fetchContainers());
  }

  _fetchContainers() {
    ForkliftUtils.fetch('/container/list/', {method: 'GET'}, null, true)
      .then((data: any) => {
        this.containers = data.detail;
        (this.shadowRoot?.querySelector('#containers-table') as Grid).clearCache();
      }).catch(() => {
        globalThis.notification.show('Error on retrieving running containers');
      });
  }

  _commit() {
    const repo = this.shadowRoot?.querySelector('#repo') as TextField;
    const tag = this.shadowRoot?.querySelector('#tag') as TextField;
    const pause = this.shadowRoot?.querySelector('#pause') as Switch;
    repo.reportValidity();
    tag.reportValidity();
    if (!repo.checkValidity() || !tag.checkValidity()) return;
    const body = {
      'param': {
        'container_id': this.containerId,
        'repo': repo.value,
        'tag': tag.value,
      },
      'pause': pause.selected,
    };
    ForkliftUtils.fetch(`/container/commit/`, {method: 'POST', body: JSON.stringify(body)}, null, true)
      .then((resp) => {
        globalThis.notification.show(resp.detail);
        this.shadowRoot?.querySelector<BackendAiDialog>('#commit-dialog')?.hide();
      });
  }

  _showContainerDetailDialog(id) {
    this.containerId = id;
    try {
      const metadata = this.containers.filter((item) => item.Id === this.containerId)[0];
      const cmEl = this.shadowRoot?.querySelector('#container-detail-codemirror') as any;
      if (cmEl) {
        cmEl.setValue(JSON.stringify(metadata, null, 2));
      }
    } catch (err) {
      globalThis.notification.show(err);
    }
    this.shadowRoot?.querySelector<BackendAiDialog>('#container-detail-dialog')?.show();
  }

  _showCommitDialog(id) {
    this.containerId = id;
    this.shadowRoot?.querySelector<BackendAiDialog>('#commit-dialog')?.show();
  }

  _showHelpDescriptionDialog() {
    this.shadowRoot?.querySelector<BackendAiDialog>('#commit-help-description')?.show();
  }

  _createdAtRenderer(root, column, rowData) {
    render(
      html`
        <div>${ForkliftUtils._humanReadableElapsedSeconds(rowData.item.Created)}</div>
      `,
      root
    );
  }

  _controlsRenderer(root, column?, rowData?) {
    render(
      html`
        <div class="layout horizontal flex center"
          .id="${rowData.item.Id}" .imageId="${rowData.item.ImageId}">
          <mwc-icon-button icon="info" class="fg green controls-running"
            @click="${() => this._showContainerDetailDialog(rowData.item.Id)}"></mwc-icon-button>
          <mwc-icon-button icon="commit" class="fg blue controls-running"
            @click="${() => this._showCommitDialog(rowData.item.Id)}"></mwc-icon-button>
        </div>
      `,
      root
    );
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

  static get styles(): CSSResultGroup {
    return [
      ForkliftStyles,
      IronFlex,
      IronFlexAlignment,
      // language=CSS
      css`
        #commit-help-description {
          --component-max-width: 660px;
        }

        #container-detail-dialog {
          --component-max-width: 60vw;
        }

        .commit-option {
          margin: 12px;
        }

        .commit-option span {
          min-width: 30%;
        }

        code {
          line-height: 1.5;
          font-size: 14px;
        }

        vaadin-grid {
          font-size: 14px;
          height: calc(100vh - 235px);
        }

        mwc-textfield {
          width: 100%;
        }
      `
    ];
  }

  render() {
    // language=HTML
    return html`
      <vaadin-grid theme="row-stripes column-borders compact" aria-label="running-containers" id="containers-table" .items="${this.containers}">
        <vaadin-grid-column width="40px" header="#" flex-grow="0" text-align="end" .renderer="${this._indexRenderer}"></vaadin-grid-column>
        <vaadin-grid-column width="100px" path="Id" header="Container ID" flex-grow="0" resizable></vaadin-grid-column>
        <vaadin-grid-filter-column path="Image" header="Image" resizable></vaadin-grid-filter-column>
        <vaadin-grid-column path="Command" header="Command" resizable></vaadin-grid-column>
        <vaadin-grid-sort-column path="Created" header="Created" resizable .renderer="${this._boundCreatedAtRenderer}"></vaadin-grid-sort-column>
        <vaadin-grid-column path="Status" header="Status" resizable></vaadin-grid-column>
        <vaadin-grid-filter-column path="Names" header="Names" resizable></vaadin-grid-filter-column>
        <vaadin-grid-column width="120px" header="Controls" flex-grow="0" resizable .renderer="${this._boundControlsRenderer}"></vaadin-grid-column>
      </vaadin-grid>
      <backend-ai-dialog id="container-detail-dialog" fixed backdrop>
        <span slot="title">${this.containers.filter((item) => item.Id === this.containerId).map((item) => item.Image)}</span>
        <div slot="content">
          <lablup-codemirror id="container-detail-codemirror" readonly></lablup-codemirror>
        </div>
      </backend-ai-dialog>
      <backend-ai-dialog id="commit-dialog">
        <span slot="title">Create a new image from a container’s changes (Commit)</span>
        <mwc-icon-button slot="action" icon="info" fg green controls-running @click="${() => this._showHelpDescriptionDialog()}"></mwc-icon-button>
        <div slot="content">
          <div class="horizontal layout justified center commit-option">
            <span>Repository</span>
            <mwc-textfield id="repo" auto-validate required></mwc-textfield>
          </div>
          <div class="horizontal layout justified center commit-option">
            <span>Tag</span>
            <mwc-textfield id="tag" auto-validate required></mwc-textfield>
          </div>
          <div class="horizontal layout center commit-option">
            <span>Pause</span>
            <mwc-switch id="pause" selected></mwc-switch>
          </div>
        </div>
        <div slot="footer" class="horizontal flex layout">
          <mwc-button unelevated fullwidth icon="check" class="operation" label="Create"
            @click="${() => this._commit()}"></mwc-button>
        </div>
      </backend-ai-dialog>
      <backend-ai-dialog id="commit-help-description">
        <span slot="title">Create a new image from a container’s changes (Commit)</span>
        <div slot="content">
          This has the same operation as the following command.
          <pre><code>docker commit [OPTIONS] CONTAINER [REPOSITORY[:TAG]]</code></pre>
          It can be useful to commit a container’s file changes or settings into a new image. This allows you to debug a container by running an interactive shell, or to export a working dataset to another server. Generally, it is better to use Dockerfiles to manage your images in a documented and maintainable way.
          <a href="https://docs.docker.com/engine/reference/commandline/tag/" target="_blank">Read more about valid image names and tags.</a>
          <br /><br />
          The commit operation will not include any data contained in volumes mounted inside the container.
          <br /><br />
          By default, the container being committed and its processes will be paused while the image is committed. This reduces the likelihood of encountering data corruption during the process of creating the commit. If this behavior is undesired, set the <code>pause</code> option to false.
        </div>
      </backend-ai-dialog>
    `;
  }
}
declare global {
  interface HTMLElementTagNameMap {
    'forklift-container-list': ForkliftContainerList;
  }
}
