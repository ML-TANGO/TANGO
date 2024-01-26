/**
@license
Copyright 2019 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
import {css, customElement} from 'lit-element';

import {SnackbarBase} from '@material/mwc-snackbar/mwc-snackbar-base';
import {styles} from '@material/mwc-snackbar/mwc-snackbar.css';

declare global {
    interface HTMLElementTagNameMap {
        'mwc-snackbar': Snackbar;
    }
}

@customElement('mwc-snackbar')
export class Snackbar extends SnackbarBase {
  static get styles() {
    return [styles,
      css`
        .mdc-snackbar .mdc-snackbar__surface {
          position: fixed;
          right: 20px;
          bottom: 20px;
          font-size: 16px;
          font-weight: 400;
          font-family: 'Ubuntu', Roboto, sans-serif;
        }
      `];
  }
}
