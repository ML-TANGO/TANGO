/**
 Backend.AI base view page for Single-page application

@group Backend.AI Forklift UI
 */
import {get as _text, LanguageIdentifier, registerTranslateConfig} from 'lit-translate';
import {LitElement} from 'lit';
import {property} from 'lit/decorators.js';

/**
 Backend AI Page

@group Backend.AI Forklift UI
@element backend-ai-page
*/

registerTranslateConfig({
  loader: (lang: LanguageIdentifier) => {
    return fetch(`/resources/i18n/${lang}.json`).then((res: Response) => {
      return res.json();
    });
  }
});

export class BackendAIPage extends LitElement {
  public notification: any; // Global notification
  @property({type: Boolean}) active = false;

  constructor() {
    super();
    this.active = false;
  }

  public _viewStateChanged(param: boolean): void;

  // eslint-disable-next-line @typescript-eslint/no-empty-function
  public _viewStateChanged(param: any) {
  }

  shouldUpdate(): boolean {
    return this.active;
  }

  connectedCallback(): void {
    super.connectedCallback();
  }

  disconnectedCallback(): void {
    super.disconnectedCallback();
  }

  attributeChangedCallback(name: string, oldval: string|null, newval: string|null): void {
    if (name == 'active' && newval !== null) {
      this.active = true;
      this._viewStateChanged(true);
    } else if (name === 'active') {
      this.active = false;
      this._viewStateChanged(false);
    }
    super.attributeChangedCallback(name, oldval, newval);
  }
  /**
   * Hide the backend.ai dialog.
   *
   * @param {Event} e - Dispatches from the native input event each time the input changes.
   */
  _hideDialog(e: any) {
    const hideButton = e.target;
    const dialog = hideButton.closest('backend-ai-dialog');
    dialog.hide();
  }

  // Compatibility layer from here.
  _addInputValidator(obj: any) {
    if (!obj.hasAttribute('auto-validate')) {
      return;
    }
    let validationMessage: string;
    if (obj.validityTransform === null) {
      if (obj.getAttribute('error-message')) { // Support paper-component style attribute
        validationMessage = obj.getAttribute('error-message');
      } else if (obj.getAttribute('validationMessage')) { // Support standard attribute
        validationMessage = obj.getAttribute('validationMessage');
      } else {
        validationMessage = 'Validation failed';
      }
      obj.validityTransform = (value: any, nativeValidity: any) => {
        if (!nativeValidity.valid) {
          if (nativeValidity.patternMismatch) {
            obj.validationMessage = validationMessage;
            return {
              valid: nativeValidity.valid,
              patternMismatch: !nativeValidity.valid
            };
          } else if (nativeValidity.valueMissing) {
            obj.validationMessage = 'Value Required';
            return {
              valid: nativeValidity.valid,
              valueMissing: !nativeValidity.valid
            };
          } else if (nativeValidity.tooShort) {
            obj.validationMessage = 'Input value is too short';
            return {
              valid: nativeValidity.valid,
              valueMissing: !nativeValidity.valid
            };
          } else {
            obj.validationMessage = validationMessage;
            return {
              valid: nativeValidity.valid,
              patternMismatch: !nativeValidity.valid,
            };
          }
        } else {
          return {
            valid: nativeValidity.valid
          };
        }
      };
    }
  }
}
