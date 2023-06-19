import {css, CSSResultGroup, html} from 'lit';
import {customElement, property} from 'lit/decorators.js';

import {BackendAIPage} from './backend-ai-page';
import {ForkliftUtils} from './forklift-utils';
import {BackendAiStyles} from './backend-ai-general-styles';
import {IronFlex, IronFlexAlignment, IronPositioning} from '../plastics/layout/iron-flex-layout-classes';

import '@material/mwc-icon/mwc-icon';
import '@material/mwc-button/mwc-button';
import '@material/mwc-select/mwc-select';
import '@material/mwc-switch/mwc-switch';
import '@material/mwc-textarea/mwc-textarea';
import '@material/mwc-textfield/mwc-textfield';
import '@material/mwc-formfield/mwc-formfield';
import '@material/mwc-icon-button/mwc-icon-button';
import '@material/mwc-list/mwc-check-list-item';
import {Switch} from '@material/mwc-switch/mwc-switch';
import {List} from '@material/mwc-list/mwc-list';
import {Select} from '@material/mwc-select/mwc-select';
import {TextArea} from '@material/mwc-textarea/mwc-textarea';
import {TextField} from '@material/mwc-textfield/mwc-textfield';

import 'weightless/expansion';

import BackendAiDialog from './backend-ai-dialog';
import './lablup-codemirror';
import './lablup-activity-panel';
import './forklift-collapse';

@customElement('forklift-custom-build-view')
export default class ForkliftCustomBuildView extends BackendAIPage {
  @property({type: String}) mode;
  @property({type: String}) iconsPath;
  @property({type: String}) resetItem;
  @property({type: Array}) importOpts;
  @property({type: Array}) supportedServiceTemplates;
  @property({type: Array}) supportedServiceProtocols;
  @property({type: Array}) selectedServiceApps;
  @property({type: Array}) baseDistro;
  @property({type: Array}) accelerators;
  @property({type: Array}) runtimeTypes;
  @property({type: Array}) servicePortsTemplates;
  @property({type: Array}) environ;
  @property({type: Array}) servicePorts;
  @property({type: Array}) customPackageTypes;
  @property({type: Object}) environValues;
  @property({type: Object}) customPackages;
  @property({type: Object}) deleteEnvRow;
  @property({type: Object}) dockerfileContents;
  @property({type: Object}) buildOptionInfo;

  constructor() {
    super();
    this.mode = 'build';
    this.resetItem = '';
    this.importOpts = [];
    this.supportedServiceTemplates = [];
    this.supportedServiceProtocols = ['http', 'tcp', 'pty', 'preopen', 'ssh', 'https'];
    this.selectedServiceApps = [];
    this.baseDistro = ['ubuntu'];
    this.accelerators = ['cuda'];
    this.runtimeTypes = ['python', 'app'];
    this.iconsPath = '../../resources/icons/';
    this.servicePortsTemplates = [
      {
        'name': 'jupyter',
        'protocol': 'http',
        'ports': [8081]
      },
      {
        'name': 'jupyterlab',
        'protocol': 'http',
        'ports': [8090]
      },
      {
        'name': 'tensorboard',
        'protocol': 'http',
        'ports': [6006]
      },
      {
        'name': 'digits',
        'protocol': 'http',
        'ports': [5000]
      },
      {
        'name': 'vscode',
        'protocol': 'http',
        'ports': [8180]
      },
      {
        'name': 'ipython',
        'protocol': 'pty',
        'ports': [3000]
      },
      {
        'name': 'mlflow-ui',
        'protocol': 'preopen',
        'ports': [5000]
      },
      {
        'name': 'sftp',
        'protocol': 'ssh',
        'ports': [22]
      },
      {
        'name': 'nni',
        'protocol': 'preopen',
        'ports': [8080]
      },
    ];
    this.environ = [];
    this.customPackages = {};
    this.customPackageTypes = ['apt', 'pip', 'conda'];
    this.dockerfileContents = {
      contents: '',
    };
  }

  firstUpdated() {
    this._fetchBuildOptionInfo();
  }

  /**
   * Check that required inputs are valid.
   *
   * @return {boolean} whether all required input are valid.
   */
  _checkValidity() {
    let isValid = true;
    const src = this.shadowRoot?.querySelector('#src');
    const target = this.shadowRoot?.querySelector('#target');
    const minCpu = this.shadowRoot?.querySelector('#min-cpu');
    const minMem = this.shadowRoot?.querySelector('#min-mem');
    const runtimePath = this.shadowRoot?.querySelector('#runtime-path');
    const runtimeType = this.shadowRoot?.querySelector('#runtime-type');
    const baseDistro = this.shadowRoot?.querySelector('#base-distro');
    const requiredInput = [src, target, minCpu, minMem, runtimePath, runtimeType, baseDistro];
    requiredInput.forEach((input) => {
      this._addInputValidator(input);
    });
    for (let i = 0; i < requiredInput.length; i++) {
      if (!(requiredInput[i] as any).checkValidity()) {
        isValid = false;
        break;
      }
    }
    return isValid;
  }

  _setInput() {
    const accelerators = this.shadowRoot?.querySelector('#accelerators') as List;
    const selectedAccelerators = Array.prototype.filter.call(
      accelerators.querySelectorAll('mwc-check-list-item'), (item) => item.selected
    ).map((item) => item.value);

    if (!this._checkValidity()) return;

    this._saveServiceList();
    this._parseEnvVariableList();
    this._saveEnvVariableList();
    this._saveCustomPackages();

    const src = this.shadowRoot?.querySelector('#src') as TextField;
    const target = this.shadowRoot?.querySelector('#target') as TextField;
    const input = {
      src: src.value,
      target: target.value,
      labels: {
        base_distro: (this.shadowRoot?.querySelector('#base-distro') as Select).value,
        service_ports: this.servicePorts,
        accelerators: selectedAccelerators,
        min_cpu: (this.shadowRoot?.querySelector('#min-cpu') as TextField).value,
        min_mem: (this.shadowRoot?.querySelector('#min-mem') as TextField).value,
      },
      runtime_type: (this.shadowRoot?.querySelector('#runtime-type') as Select).value,
      runtime_path: (this.shadowRoot?.querySelector('#runtime-path') as TextField).value,
      docker_commands: {
        environments: this.environ,
        custom_pkg: this.customPackages,
      },
      auto_push: (this.shadowRoot?.querySelector('#auto-push') as Switch).selected ?? false,
      allow_root: (this.shadowRoot?.querySelector('#allow-root') as Switch).selected ?? false,
    };
    return input;
  }

  _fetchDockerFilePreview(input) {
    return ForkliftUtils.fetch(`/build/dockerfile/preview/`, {method: 'POST', body: JSON.stringify(input)}, null, true)
      .then((resp) => {
        return resp;
      }).catch((e) => {
        const msg = 'Error on previewing a Dockerfile.';
        globalThis.notification.show(msg);
      });
  }

  async _showDockerfilePreview() {
    const msg = 'Loading Dockerfile Preview...';
    globalThis.notification.show(msg);

    const input = this._setInput();
    if (input) {
      const contents = await this._fetchDockerFilePreview(input);
      const cmEl = this.shadowRoot?.querySelector('#show-dockerfile-codemirror') as any;
      this.dockerfileContents.contents = contents;
      if (cmEl) {
        cmEl.setValue(contents);
        cmEl.focus();
      }
      this._showDialog('show-dockerfile-dialog');
    } else {
      const msg = 'Enter valid input.';
      globalThis.notification.show(msg);
    }
  }

  async _fetchBuildOptionInfo() {
    await fetch('resources/build_options_metadata.json').then(
      (response) => response.json()
    ).then(
      (json) => {
        this.buildOptionInfo = json.buildOptionInfo;
        const optionInfo = Object();
        for (const key in json.buildOptionInfo) {
          if ({}.hasOwnProperty.call(json.buildOptionInfo, key)) {
            optionInfo[key] = {};
            if ('name' in json.buildOptionInfo[key]) {
              optionInfo[key].name = json.buildOptionInfo[key].name;
            }
            if ('description' in json.buildOptionInfo[key]) {
              optionInfo[key].description = json.buildOptionInfo[key].description;
            } else {
              optionInfo[key].description = 'No Description';
            }
            if ('icon' in json.buildOptionInfo[key]) {
              optionInfo[key].icon = json.buildOptionInfo[key].icon;
            }
          }
        }
        this.buildOptionInfo = optionInfo;
      }
    );
  }

  _showDialog(id) {
    this.shadowRoot?.querySelector<BackendAiDialog>('#' + id)?.show();
  }

  _hideDialog(id) {
    this.shadowRoot?.querySelector<BackendAiDialog>('#' + id)?.hide();
  }

  _downloadDockerfile() {
    const msg = 'Downloading Dockerfile...';
    globalThis.notification.show(msg);

    let fileName = '';
    ForkliftUtils.fetch(`/build/dockerfile/download/`, {method: 'GET'}, null, true)
      .then((data) => {
        const disposition = data['disposition'];
        fileName = decodeURI(disposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/)[1].replace(/['"]/g, ''));
        return data['body'].blob();
      })
      .then((blob) => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = fileName;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(blob);
      });
  }

  _createEnvironmentRow() {
    const div = document.createElement('div');
    div.setAttribute('class', 'horizontal layout center row');

    const variable = document.createElement('mwc-textfield');

    const value = document.createElement('mwc-textfield');

    const removeButton = document.createElement('mwc-icon-button');
    removeButton.setAttribute('icon', 'remove');
    removeButton.setAttribute('class', 'minus-btn');
    removeButton.addEventListener('click', (e) => this._removeRow(e));

    div.append(variable);
    div.append(value);
    div.append(removeButton);
    return div;
  }

  _addEnvironmentItems() {
    const container = this.shadowRoot?.querySelector('#env-container') as HTMLDivElement;
    const lastChild = container?.children[container.children.length - 1];
    const div = this._createEnvironmentRow();
    container.insertBefore(div, lastChild);
  }

  /**
   * Check whether delete operation will proceed or not.
   *
   * @param {Event} e - Dispatches from the native input event each time the input changes.
   */
  _removeRow(e) {
    // htmlCollection should be converted to Array.
    this.deleteEnvRow = e.target.closest('.row');
    this.deleteEnvRow.remove();
  }

  /**
   * Remove rows from the environment variable.
   */
  _removeAllRows(id, force = false) {
    const container = this.shadowRoot?.querySelector('#' + id) as HTMLDivElement;
    const rows = container.querySelectorAll('.row:not(.header)');
    const firstRow = rows[0];
    // show confirm dialog if not empty.
    if (!force) {
      const nonempty = (row) => Array.prototype.filter.call(
        row.querySelectorAll('mwc-textfield, mwc-select'), (item) => item.value.length > 0
      ).length > 0;
      if (Array.prototype.filter.call(rows, (row) => nonempty(row)).length > 0) {
        this.resetItem = id;
        this._showDialog('delete-confirm-dialog');
        return;
      }
    }
    // remove values
    firstRow.querySelectorAll('mwc-textfield, mwc-select').forEach((item: any) => {
      item.value = '';
    });
    // delete rows except for the first row
    container.querySelectorAll('div.row').forEach((e, idx) => {
      if (idx !== 0) {
        e.remove();
      }
    });
    this._hideDialog('delete-confirm-dialog');
  }

  /**
   * Remove empty env input fields
   */
  _removeEmptyEnv() {
    const container = this.shadowRoot?.querySelector('#env-container') as HTMLDivElement;
    const rows = container.querySelectorAll('.row');
    const empty = (row) => Array.prototype.filter.call(
      row.querySelectorAll('mwc-textfield'), (tf, idx) => tf.value === ''
    ).length === 2;
    Array.prototype.filter.call(rows, (row) => empty(row)).map((row) => row.parentNode.removeChild(row));
  }

  /**
   * Parse environment variables on UI.
   */
  _parseEnvVariableList() {
    this.environValues = {};
    const container = this.shadowRoot?.querySelector('#env-container') as HTMLDivElement;
    const rows = container.querySelectorAll('.row:not(.header)');
    const nonempty = (row) => Array.prototype.filter.call(
      row.querySelectorAll('mwc-textfield'), (item) => item.value.length === 0
    ).length === 0;
    const encodeRow = (row) => {
      const items: Array<any> = Array.prototype.map.call(row.querySelectorAll('mwc-textfield'), (tf) => tf.value);
      this.environValues[items[0]] = items[1];
      return items;
    };
    Array.prototype.filter.call(rows, (row) => nonempty(row)).map((row) => encodeRow(row));
  }

  /**
   * Save Environment variables.
   */
  _saveEnvVariableList() {
    this.environ = [];
    Object.entries(this.environValues).map(([key, value]) => this.environ.push(`${key}=${value}`));
  }

  _saveCustomPackages() {
    this.customPackageTypes.map((item) => {
      this.customPackages[item] = (this.shadowRoot?.querySelector('#' + item) as TextArea).value?.replace(/^,|,$/g, '').split(',');
      if (this.customPackages[item][0] === '') {
        this.customPackages[item].pop();
      }
    });
  }

  _addServiceTemplate(e) {
    const selected = e.target.closest('mwc-list-item')?.value;
    if (!selected) return;
    const [name, protocol, ports] = selected.split(':');
    const row = e.target.closest('.row');
    (row?.querySelector('.name') as TextField).value = name;
    (row?.querySelector('.protocol') as Select).value = protocol;
    (row?.querySelector('.ports') as TextField).value = ports;
  }

  _createServiceRow() {
    const div = document.createElement('div');
    div.setAttribute('class', 'horizontal layout center row');

    const templates = document.createElement('mwc-select');
    templates.setAttribute('label', 'Templates');
    templates.addEventListener('click', (e) => this._addServiceTemplate(e));
    const templateItems = this.servicePortsTemplates.map((tmpl) => {
      const stringify = `${tmpl.name}:${tmpl.protocol}:${tmpl.ports}`;
      const item = document.createElement('mwc-list-item');
      item.setAttribute('value', stringify);
      item.innerHTML = stringify;
      return item;
    });
    templateItems.forEach((item) => templates.append(item));

    const name = document.createElement('mwc-textfield');
    name.setAttribute('label', 'Name');
    name.setAttribute('class', 'name');

    const protocol = document.createElement('mwc-select');
    protocol.setAttribute('label', 'Protocol');
    protocol.setAttribute('class', 'protocol');
    const protocolItems = this.supportedServiceProtocols.map((protocol) => {
      const item = document.createElement('mwc-list-item');
      item.setAttribute('value', protocol);
      item.innerHTML = protocol;
      return item;
    });
    protocolItems.forEach((item) => protocol.append(item));

    const ports = document.createElement('mwc-textfield');
    ports.setAttribute('label', 'Ports (comma-separated)');
    ports.setAttribute('class', 'ports');

    const removeButton = document.createElement('mwc-icon-button');
    removeButton.setAttribute('icon', 'remove');
    removeButton.setAttribute('class', 'minus-btn');
    removeButton.addEventListener('click', (e) => this._removeRow(e));

    div.append(templates);
    div.append(name);
    div.append(protocol);
    div.append(ports);
    div.append(removeButton);
    return div;
  }

  _addServicePortsItems() {
    const container = this.shadowRoot?.querySelector('#service-ports-container') as HTMLDivElement;
    const lastChild = container?.children[container.children.length - 1];
    const div = this._createServiceRow();
    container.insertBefore(div, lastChild);
  }

  /**
   * Parse and save supported service ports on UI.
   */
  _saveServiceList() {
    this.servicePorts = [];
    const container = this.shadowRoot?.querySelector('#service-ports-container') as HTMLDivElement;
    const rows = container.querySelectorAll('.row');
    const nonempty = (row) => Array.prototype.filter.call(
      row.querySelectorAll('mwc-textfield, mwc-select'), (item, idx) => {
        if (idx === 0) {
          return;
        }
        return item.value.length === 0;
      }).length === 0;
    const encodeRow = (row) => {
      const items: Array<any> = Array.prototype.map.call(row.querySelectorAll('mwc-textfield, mwc-select'), (item) => item.value);
      const ports = items[3].split(',').map((p) => parseInt(p)).filter((p) => Number.isInteger(p));
      if (ports.length > 0) {
        this.servicePorts.push({'name': items[1], 'protocol': items[2], 'ports': ports});
      }
    };
    Array.prototype.filter.call(rows, (row) => nonempty(row)).map((row) => encodeRow(row));
  }

  async submit() {
    const msg = 'Building an image...\nI\'ll let you know when the build is completed.';
    globalThis.notification.show(msg);
    const input = this._setInput();
    if (input) {
      await ForkliftUtils.fetch(`/build/submit/custom/`, {method: 'POST', body: JSON.stringify(input)}, null, true)
        .then((resp) => {
          globalThis.notification.show(resp.detail);
        })
        .catch((e) => {
          console.log(e);
          const msg = 'Error on building an image.';
          globalThis.notification.show(msg);
        });
    } else {
      const msg = 'Enter valid input.';
      globalThis.notification.show(msg);
    }
  }

  static get styles(): CSSResultGroup {
    return [
      BackendAiStyles,
      IronFlex,
      IronFlexAlignment,
      IronPositioning,
      // language=CSS
      css`
        mwc-textfield, mwc-select {
          width: 50%;
          margin: 20px;
          --mdc-text-field-fill-color: transparent;
          --mdc-theme-primary: var(--paper-green-600);
        }

        mwc-formfield {
          margin: 20px;
          padding-left: 18px;
          width: 50%;
          height: 56px;
          font-family: var(--general-font-family);
          border-bottom: 1px solid var(--general-border-color);
        }

        forklift-collapse {
          margin: 20px;
        }

        mwc-button.red {
          --mdc-theme-primary: var(--paper-red-600);
        }
        
        div[role=separator] {
            height: 20px;
            margin: 0px;
            border-bottom: 1px dashed rgba(0, 0, 0, 0.12);
        }

        .minus-btn {
          color: var(--general-button-background-color);
          --mdc-icon-size: 20px;
        }

        .content {
          padding: 10px 40px;
        }

        .subtitle {
          font-family: var(--general-font-family);
          font-size: 14px;
          margin: 40px auto 20px 25px;
        }

        .flex-item {
          flex: 1;
        }

        .content div[slot=footer] mwc-button {
          margin: 30px 7px;
        }

        #title {
          font-weight: 200;
          font-size: 14px;
          margin: 0;
          padding-left: 20px;
        }

        #title span {
          color: #DDDDDD;
        }

        #env-container mwc-textfield {
          height: 50px;
          width: 90%;
          margin: 10px;
        }

        #env-container div.header {
          margin: 15px auto auto 30px;
        }

        #service-add-btn, #env-add-btn {
          margin: 5px 10px;
        }

        #service-minus-btn {
          margin-top: 20px;
        }

        #custom-packages-container mwc-select {
          --mdc-menu-max-width: 200px;
          --mdc-menu-min-width: 200px;
          --mdc-select-min-width: 190px;
        }

        #help-description {
          --component-max-width: 90vw;
        }

        #help-description wl-expansion {
          width: 98%;
          border-bottom: none;
        }

        #show-dockerfile-dialog {
          --component-min-width: 60vw;
        }

        #accelerators {
          width: 50%;
          margin: auto 10px;
        }
      `
    ];
  }

  render() {
    return html`
      <div class="card" elevation="0">
        <h4 id="title" class="horizontal center layout">
          <span>Custom Image Build</span>
          <span class="flex"></span>
          <mwc-icon-button icon="info" class="fg green" @click="${() => this._showDialog('help-description')}"></mwc-icon-button>
        </h4>
        <div class="content">
          <div class="horizontal layout">
            <mwc-textfield required id="src" name="src"
              auto-validate label="Source docker image"
              placeholder="index.docker.io/lablup/tensorflow:2.0-source"
              maxLength="128"
              pattern="[a-zA-Z0-9-_:/. ]*"
              validationMessage="Allow characters, numbers, -_. and colon"></mwc-textfield>
            <mwc-textfield required id="target" name="target"
              auto-validate label="Target docker image"
              placeholder="index.docker.io/lablup/tensorflow:2.0-target"
              maxLength="128"
              pattern="[a-zA-Z0-9-_:/. ]*"
              validationMessage="Allow characters, numbers, -_. and colon"></mwc-textfield>
          </div>
          <div class="horizontal layout">
            <mwc-textfield required id="min-cpu" name="min-cpu" type="number"
              auto-validate label="Minimum required CPU core(s)"
              min="1"
              placeholder="1"
              value="1"></mwc-textfield>
            <mwc-textfield required id="min-mem" name="min-mem"
              auto-validate label="Minimum required memory size (minimum 64m)"
              placeholder="64m"
              value="64m"></mwc-textfield>
          </div>
          <div class="horizontal layout">
            <mwc-select required id="runtime-type" name="runtime-type"
              auto-validate label="Runtime type of the image">
              ${this.runtimeTypes.map((item, idx) => html`
                <mwc-list-item id="${item}" value="${item}" ?selected="${idx === 0}">${item}</mwc-list-item>
              `)}
            </mwc-select>
            <mwc-textfield required id="runtime-path" name="runtime-path"
              auto-validate label="Path of the runtime"
              placeholder="python3"
              value="python3"></mwc-textfield>
          </div>
          <div>
            <mwc-select required id="base-distro" name="base-distro"
              auto-validate label="Base LINUX distribution"
              help="The base Linux distribution used by the source image"
              style="width:calc(50% - 40px);">
              ${this.baseDistro.map((item, idx) => html`
                <mwc-list-item id="${item}" value="${item}" ?selected="${idx === 0}">${item}</mwc-list-item>
              `)}
            </mwc-select>
          </div>
          <forklift-collapse id="advanced-options" class="horizontal center layout" scrollInto>
            <span slot="title">Advanced Options</span>
            <div slot="content">
              <div>
                <h4 class="subtitle black">Supported accelerators</h4>
                <mwc-list slot="content" multi id="accelerators">
                  ${this.accelerators.map((item) => html`
                    <mwc-check-list-item id="${item}" value="${item}">${item}</mwc-check-list-item>
                  `)}
                </mwc-list>

              </div>
              <div role="separator"></div>
              <div>
                <div class="horizontal layout center flex justified">
                  <h4 class="subtitle black">Supported service ports</h4>
                  <mwc-button id="service-minus-btn" label="Reset"
                    @click="${() => this._removeAllRows('service-ports-container')}"></mwc-button>
                </div>
                <div id="service-ports-container">
                  <div class="horizontal layout center row">
                    <mwc-select id="service-templates" name="templates"
                      label="Templates"
                      @click="${(e) => this._addServiceTemplate(e)}">
                      ${this.servicePortsTemplates.map((tmpl) => {
                          const stringify = `${tmpl.name}:${tmpl.protocol}:${tmpl.ports}`;
                          return html`<mwc-list-item value="${stringify}">${stringify}</mwc-list-item>`;
                        })}
                    </mwc-select>
                    <mwc-textfield label="Name" class="name"></mwc-textfield>
                    <mwc-select id="service-protocol" label="Protocol" class="protocol">
                      ${this.supportedServiceProtocols.map((item) => html`
                        <mwc-list-item id="${item}" value="${item}">${item}</mwc-list-item>
                      `)}
                    </mwc-select>
                    <mwc-textfield label="Ports (comma-separated)" class="ports"></mwc-textfield>
                    <mwc-icon-button class="minus-btn" icon="remove"
                      @click="${(e) => this._removeRow(e)}"></mwc-icon-button>
                  </div>
                  <mwc-button id="service-add-btn" outlined icon="add" class="horizontal flex layout center"
                    @click="${() => this._addServicePortsItems()}">Add</mwc-button>
                </div>
              </div>
              <div role="separator"></div>
              <div>
                <h4 id="docker-commands-title" class="subtitle black">Docker commands</h4>
                <div class="horizontal layout">
                  <forklift-collapse id="environ-variables" class="flex-item" scrollInto>
                    <span slot="title">Set environment variable</span>
                    <div slot="content" id="env-container">
                      <div class="horizontal layout center flex justified header">
                        <div>Variable</div>
                        <div>Value</div>
                        <mwc-button id="env-minus-btn" label="Reset"
                          @click="${() => this._removeAllRows('env-container')}"></mwc-button>
                      </div>
                      <div id="env-fields-container" class="layout center">
                        <div class="horizontal layout center row">
                          <mwc-textfield></mwc-textfield>
                          <mwc-textfield></mwc-textfield>
                          <mwc-icon-button class="minus-btn" icon="remove"
                            @click="${(e) => this._removeRow(e)}"></mwc-icon-button>
                        </div>
                      </div>
                      <mwc-button id="env-add-btn" outlined icon="add" class="horizontal flex layout center"
                        @click="${() => this._addEnvironmentItems()}">Add</mwc-button>
                    </div>
                  </forklift-collapse>
                  <forklift-collapse id="custom-packages" class="flex-item" scrollInto>
                    <span slot="title">Custom packages (comma-separated)</span>
                    <div slot="content" id="custom-packages-container" class="vertical layout">
                      <div class="row vertical layout center">
                        ${this.customPackageTypes.map((item) => html`
                          <mwc-textarea id="${item}" label="${item}" cols="1000"></mwc-textarea>
                        `)}
                      </div>
                    </div>
                  </forklift-collapse>
                </div>
                <div class="horizontal layout">
                  <mwc-formfield alignEnd spaceBetween label="Auto push docker image">
                    <mwc-switch id="auto-push"></mwc-switch>
                  </mwc-formfield>
                  <mwc-formfield alignEnd spaceBetween label="Allow root">
                    <mwc-switch id="allow-root"></mwc-switch>
                  </mwc-formfield>
                </div>
              </div>
            </div>
          </forklift-collapse>
          <div slot="footer" class="end-justified layout flex horizontal">
            <mwc-button id="download-btn" outlined icon="download" @click="${() => this._downloadDockerfile()}">Download Dockerfile</mwc-button>
            <mwc-button id="preview-btn" outlined icon="content_paste_search" @click="${() => this._showDockerfilePreview()}">Preview Dockerfile</mwc-button>
            <mwc-button id="build-btn" unelevated icon="construction" @click="${() => this.submit()}">Build</mwc-button>
          </div>
        </div>
      </div>
      <backend-ai-dialog id="help-description">
        <span slot="title">Build Image Options Detail</span>
        <div slot="content">
          ${this.buildOptionInfo && Object.entries(this.buildOptionInfo).map(([key]) => html`
            <wl-expansion class="center layout">
              <span slot="title" class="horizontal center layout">
                <mwc-icon style="margin-right:10px;">${this.buildOptionInfo[key].icon}</mwc-icon>
                ${this.buildOptionInfo[key].name}
              </span>
              <div style="margin-left:40px;">${this.buildOptionInfo[key].description}</div>
            </wl-expansion>
          `)}
        </div>
      </backend-ai-dialog>
      <backend-ai-dialog id="show-dockerfile-dialog" fixed backdrop>
        <span slot="title">Dockerfile Contents</span>
        <div slot="content" class="layout center">
          <lablup-codemirror id="show-dockerfile-codemirror" readonly></lablup-codemirror>
        </div>
      </backend-ai-dialog>
      <backend-ai-dialog id="delete-confirm-dialog" fixed backdrop>
        <span slot="title">Let's Double-check</span>
        <div slot="content">
          <p>Any unsaved values will be disappeared.<br/><br/>you want to proceed?</p>
        </div>
        <div slot="footer" class="horizontal end-justified flex layout">
          <mwc-button
            outlined
            class="operation"
            label="Cancel"
            @click="${() => this._hideDialog('delete-confirm-dialog')}"></mwc-button>
          <mwc-button
            unelevated
            class="operation red"
            label="Dismiss And Proceed"
            @click="${() => this._removeAllRows(this.resetItem, true)}"></mwc-button>
        </div>
      </backend-ai-dialog>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'forklift-custom-build-view': ForkliftCustomBuildView;
  }
}
