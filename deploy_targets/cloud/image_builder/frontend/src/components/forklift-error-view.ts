import {css, html, CSSResultGroup} from 'lit';
import {customElement, property} from 'lit/decorators.js';

import {BackendAiStyles} from './backend-ai-general-styles';
import {BackendAIPage} from './backend-ai-page';
import {IronFlex, IronFlexAlignment, IronPositioning} from '../plastics/layout/iron-flex-layout-classes';
import {store} from '../store';
import {navigate} from '../backend-ai-app';

@customElement('forklift-error-view')
export default class ForkliftErrorView extends BackendAIPage {
  @property({type: Number}) error_code = 404;

  static get styles(): CSSResultGroup {
    return [
      BackendAiStyles,
      IronFlex,
      IronFlexAlignment,
      IronPositioning,
      // language=CSS
      css`
      .title {
        font-size: 2em;
        font-weight: bolder;
        color: var(--general-navbar-footer-color, #424242);
        line-height: 1em;
      }

      .description {
        font-size: 1em;
        font-weight: normal;
        color: var(--general-sidebar-color, #949494);
      }

      mwc-button {
        width: auto;
      }
      `
    ];
  }

  /**
   *
   * @param {string} url - page to redirect from the current page.
   */
  _moveTo(url = '') {
    const page = url !== '' ? url : 'summary';
    globalThis.history.pushState({}, '', '/summary');
    store.dispatch(navigate(decodeURIComponent('/' + page), {}));
  }

  render() {
    // language=HTML
    return html`
    <div class="horizontal center flex layout" style="margin:20px;">
      <img src="/resources/images/404_not_found.svg" alt="404 not found" style="width:500px;margin:20px;"/>
      <div class="vertical layout" style="width:100%;">
        <div class="title">NOT FOUND</div>
        <p class="description">Sorry, the page you are looking for could not be found.</p>
        <div>
          <mwc-button
              unelevated
              fullwidth
              id="go-to-summary"
              label="Go back to Summary page"
              @click="${() => this._moveTo('summary')}"></mwc-button>
        </div>
      </div>
    </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'forklift-error-view': ForkliftErrorView;
  }
}
