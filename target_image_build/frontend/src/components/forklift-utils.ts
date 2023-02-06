/**
 * Collection of common utility methods.
 */
import toml from 'markty-toml';
import {store} from '../store';
import {navigate} from '../backend-ai-app';
import {fetchEventSource} from '@microsoft/fetch-event-source';

export class ForkliftUtils {
  static async getApiAddress(uri) {
    const config: any = await this._parseConfig('../../configs/forklift.toml');
    return `http://${config.server.bind_addr.bind_host}:${config.server.bind_addr.bind_port}${uri}`;
  }

  /**
   * Extended JavaScript's fetch method.
   *
   * - Parse / get response content before return.
   * - Shows notification text when error is catched (unless silent option give).
   * - Stop spinner when error occurs.
   *
   * @param {string} uri: address to send request.
   * @param {object} _opts: options for fetch method. Contains some keys which
   *                        does not exist in the original fetch.
   *   - {boolean} silent: If silent is true, do not show error notification text.
   * @param {object} caller: DOM element which calls fetch (will be deprecated).
   * @param {boolean} needAuthorization: whether adding Authorization to headers or not.
   * @return {object} body: response body or error object.
   */
  static async fetch(uri, _opts, caller=null, needAuthorization=false) {
    let body;
    const {silent = false, ...opts} = _opts;
    const d = new Date();
    const defaultOpts: RequestInit = {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': `Backend.AI Forklift.`,
        'X-BackendAI-Date': d.toISOString(),
      }
    };
    const mergedOpts = Object.assign(defaultOpts, opts);
    if (needAuthorization) {
      mergedOpts.headers['Authorization'] = sessionStorage.getItem('token');
    }
    try {
      uri = await this.getApiAddress(uri);
      const resp = await fetch(uri, mergedOpts);
      const contentType = resp.headers.get('Content-Type');
      if (resp.status === 204 || !contentType) return body; // No-Content
      if (contentType.startsWith('application/json') || contentType.startsWith('application/problem+json')) {
        body = await resp.json();
      } else if (contentType.startsWith('text/')) {
        const disposition = resp.headers.get('Content-Disposition');
        if (disposition) {
          body = {
            body: resp,
            disposition: disposition,
          };
          return body;
        }
        body = await resp.text();
      } else {
        body = await resp.blob();
      }
      if (!resp.ok) {
        throw body;
      }
      return body;
    } catch (err) {
      if (globalThis.spinner && globalThis.spinner.active) {
        globalThis.spinner.hide();
      }
      if (!silent && globalThis.notification) {
        let msg = err;
        if (err && 'detail' in err) {
          msg = err.detail[0].msg;
        } else if (body && 'message' in body) {
          msg = body.message;
        } else if (body && 'title' in body) {
          msg = body.title;
        } else if (body && 'error_msg' in body) {
          msg = body.error_msg;
        }
        globalThis.notification.show(msg, 'error');
      }
      throw body;
    }
  }

  static _parseConfig(fileName): Promise<void> {
    return fetch(fileName)
      .then((res) => {
        if (res.status == 200) {
          return res.text();
        }
        return '';
      })
      .then((res) => {
        return toml(res);
      }).catch(() => {
        console.log('Configuration file missing.');
      });
  }

  /**
   * Submit form with fetch API.
   *
   * @param {HTMLElement} e : HTML element under a `form` element.
   * @param {Object} fields : additional key-value object to update `formData`.
   */
  static async submitForm(e, fields) {
    const form = e.target.closest('form');
    // const formData = new FormData(form);
    const formData = new FormData();
    if (fields) {
      Object.keys(fields).forEach((key) => {
        formData.set(key, fields[key]);
      });
    }
    try {
      if (!form.reportValidity()) {
        return;
      }
      const method = form._method ? form._method : form.method;
      const headers = {};
      const d = new Date();
      if (method.toLowerCase().indexOf(['post', 'put', 'patch'])) {
        headers['accept'] = 'application/json';
        headers['User-Agent'] = `Backend.AI Forklift.`;
        headers['X-BackendAI-Date'] = d.toISOString();
      }
      const uri = await this.getApiAddress(form.getAttribute('uri'));
      const resp = await fetch(uri, {
        method: method,
        body: formData,
        headers: headers,
      });
      let respContent;
      const contentType = resp.headers.get('Content-Type');
      if (resp.status === 204 || !contentType) return respContent; // No-Content
      if (contentType.startsWith('application/json') || contentType.startsWith('application/problem+json')) {
        respContent = await resp.json();
      } else if (contentType.startsWith('text/')) {
        respContent = await resp.text();
      } else {
        respContent = await resp.blob();
      }
      if (!resp.ok) {
        throw respContent;
      }
      return respContent;
    } catch (err) {
      console.error(err);
      let errMsg = 'Form submit error';
      if (err && err.form && err.form.errors) {
        errMsg = err.fomr.errors[0] || 'Form submit error';
      } else if (err && 'message' in err) {
        errMsg = err.message;
      } else if (err && 'title' in err) {
        errMsg = err.title;
      } else if (err && 'error_msg' in err) {
        errMsg = err.error_msg;
      } else if (err && 'detail' in err) {
        errMsg = err.detail;
      }
      throw errMsg;
    }
  }

  static isJsonString(text) {
    if (typeof text !=='string') {
      return false;
    }
    try {
      const json = JSON.parse(text);
      return (typeof json === 'object');
    } catch (error) {
      return false;
    }
  }

  /**
   * Attach new event
   * @param {string} uri - uri of event api
   * @param {object} onMessage - callback function
   * @param {any} args - arguments for callback function (onMessage)
   */
  static async attachEventSource(uri, onMessage, ...args) {
    const url = await this.getApiAddress(uri);
    const hdrs = {
      'Authorization': String(sessionStorage.getItem('token')),
      'Accept': 'text/event-stream',
    };
    await fetchEventSource(url, {
      headers: hdrs,
      openWhenHidden: true,

      async onopen(resp) {
        if (resp.ok && resp.status === 200) {
          console.log('Connection made ', resp);
        } else if (resp.status >= 400 && resp.status < 500 && resp.status !== 429) {
          console.log('Client side error ', resp);
        }
      },
      onmessage(e) {
        onMessage(e.data, ...args);
      },
      onclose() {
        console.log('Connection closed by the server');
      },
      onerror(err) {
        throw new Error(`There was an error from server ${err}`);
      },
    }).catch((err) => {
      console.log(err);
      // this.closeEventSource(uri);
    });
  }

  static async closeEventSource(uri) {
    const url = await this.getApiAddress(uri);
    const hdrs = {
      'Authorization': String(sessionStorage.getItem('token')),
      'Accept': 'text/event-stream',
    };
    const ctrl = new AbortController();
    await fetchEventSource(url, {
      headers: hdrs,
      signal: ctrl.signal,
    });
  }

  static _observeElementVisibility(element, callback) {
    const options = {};
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        callback(entry.intersectionRatio > 0);
      });
    }, options);
    observer.observe(element);
  }

  /**
   * Move to input url.
   *
   * @param {string} url
   */
  static _moveTo(url: string) {
    globalThis.history.pushState({}, '', url);
    store.dispatch(navigate(decodeURIComponent(url), {}));
  }

  /**
   * Mask String with range
   *
   * @param {string} value - string to mask
   * @param {string} maskChar - character used for masking (default: '*')
   * @param {number} startFrom - exclusive index masking starts
   * @param {number} maskLength - range length to mask
   * @return {string} maskedString
   */
  static _maskString(value = '', maskChar = '*', startFrom = 0, maskLength = 0) {
    // clamp mask length
    maskLength = (startFrom + maskLength > value.length) ? value.length : maskLength;
    return value.substring(0, startFrom) + maskChar.repeat(maskLength) + value.substring(startFrom+maskLength, value.length);
  }

  static _checkLogin() {
    return sessionStorage.getItem('token') !== null;
  }

  /**
  * Convert elapsed seconds to "<passed date> ago"
  *
  * @param {Date} sec
  * @return {string} Human-readable date string.
  */
  static _humanReadableElapsedSeconds(sec) {
    const startDate = new Date(sec * 1000);
    const endDate = new Date();
    const elapsedSeconds = Math.floor((endDate.getTime() - startDate.getTime()) / 1000);
    if (elapsedSeconds < 1) {
      return 'now';
    }
    const qualifier = (num) => (num > 1 ? 's' : '');
    const numToStr = (num, unit) => num > 0 ? `${num} ${unit}${qualifier(num)} ago` : '';
    const oneMinute = 60;
    const oneHour = oneMinute * 60;
    const oneDay = oneHour * 24;
    const oneWeek = oneDay * 7;
    const oneMonth = oneDay * 30;
    const oneYear = oneDay * 365;
    const times = {
      year: ~~(elapsedSeconds / oneYear),
      month: ~~((elapsedSeconds % oneYear) / oneMonth),
      week: ~~((elapsedSeconds % oneYear) / oneWeek),
      day: ~~((elapsedSeconds % oneYear) / oneDay),
      hour: ~~((elapsedSeconds % oneDay) / oneHour),
      minute: ~~((elapsedSeconds % oneHour) / oneMinute),
      second: elapsedSeconds % oneMinute,
    };
    let str = '';
    for (const [key] of Object.entries(times)) {
      const convertedStr = numToStr(times[key], key);
      if (convertedStr !== '') {
        str += convertedStr;
        break;
      }
    }
    return str;
  }
}
