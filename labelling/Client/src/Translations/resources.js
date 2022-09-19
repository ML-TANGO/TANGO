/* eslint-disable camelcase */
import { _getLanguage } from "Config/Services/AdminApi"

export function _getAllLang(lang) {
  _getLanguage()
    .then(data => {
      let language = JSON.parse(data[0].LANG_INFR)
      switch (lang) {
        case "en":
          return language.en
        case "ko":
          return language.ko
      }
    })
    .catch(e => console.log(e))
}

export default {
  en: {
    common: _getAllLang("en")
  },
  ko: {
    common: _getAllLang("ko")
  },
  cn: {
    common: _getAllLang("en")
  },
  hu: {
    common: _getAllLang("en")
  },
  id: {
    common: _getAllLang("en")
  }
}
