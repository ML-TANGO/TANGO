/**
 @license
 Copyright (c) 2015-2019 Lablup Inc. All rights reserved.
 */
import {css, CSSResultGroup, html, LitElement} from 'lit';
import {customElement, property} from 'lit/decorators.js';

import {IronFlex, IronFlexAlignment} from '../plastics/layout/iron-flex-layout-classes';

import '@vanillawc/wc-codemirror/index';
import '@vanillawc/wc-codemirror/mode/python/python';
import '@vanillawc/wc-codemirror/mode/shell/shell';
import {CodemirrorThemeMonokai} from '../lib/codemirror/theme/monokai.css';
import {CodemirrorBaseStyle} from '../lib/codemirror/base-style.css';

/**
 Lablup Codemirror

 `lablup-codemirror` is wc-codemirror editor.

 Example:

 <lablup-codemirror></lablup-codemirror>

@group Backend.AI Web UI
 @element lablup-codemirror
 */

declare const window: any;


@customElement('lablup-codemirror')
export default class LablupCodemirror extends LitElement {
  public editor: any;

  @property({type: Object}) config = Object();
  @property({type: String}) mode = 'shell';
  @property({type: String}) theme = 'monokai';
  @property({type: String}) src = '';
  @property({type: Boolean}) readonly = false;

  constructor() {
    super();
    this.config = {
      // mode: this.mode,
      // theme: 'monokai',
      tabSize: 2,
      indentUnit: 2,
      cursorScrollMargin: 50,
      lineNumbers: true,
      // lineNumberFormatter: (line) => (line % 5 === 0 || line === 1) ? line : '',
      matchBrackets: true,
      styleActiveLine: true,
      viewportMargin: Infinity,
      extraKeys: {},
    };
  }

  firstUpdated() {
    this._initEditor();
  }

  /**
   * Initialize codemirror editor.
   * */
  _initEditor() {
    const cm = this.shadowRoot?.querySelector('#codemirror-editor') as any;
    if (!cm.__initialized) {
      setTimeout(this._initEditor.bind(this), 100);
      return;
    }
    this.editor = cm.editor;
    Object.assign(this.editor.options, this.config);
    this.refresh();
  }

  /**
   * Refresh the editor.
   * */
  refresh() {
    globalThis.setTimeout(() => this.editor.refresh(), 100);
  }

  /**
   * Give the editor focus
   * */
  focus() {
    globalThis.setTimeout(() => {
      // Set the cursor at the end of existing content
      this.editor.execCommand('goDocEnd');
      this.editor.focus();
      this.refresh();
    }, 100);
  }

  /**
   * Get the editor's contents.
   *
   * @return {string} Editor's contents
   * */
  getValue() {
    return this.editor.getValue();
  }

  /**
   * Set the editor's contents and then refresh.
   *
   * @param {string} val - content to update editor
   * */
  setValue(val) {
    this.editor.setValue(val);
    this.refresh();
  }

  static get styles(): CSSResultGroup {
    return [
      IronFlex,
      IronFlexAlignment,
      CodemirrorThemeMonokai,
      CodemirrorBaseStyle,
      // language=CSS
      css`
        .CodeMirror {
          height: auto !important;
          font-size: 15px;
        }
      `,
    ];
  }

  render() {
    // language=HTML
    return html`
      <wc-codemirror id="codemirror-editor" mode="${this.mode}" theme="monokai" ?readonly="${this.readonly}">
        <link rel="stylesheet" href="node_modules/@vanillawc/wc-codemirror/theme/monokai.css">
      </wc-codemirror>
    `;
  }
}
