import Vue from "vue";
import App from "./App.vue";
import router from "./router";
import store from "./store";
import vuetify from "./plugins/vuetify";
import VueKonva from "vue-konva";
import VueSweetalert2 from "vue-sweetalert2";
import VueClipboard from "vue-clipboard2";

import "sweetalert2/dist/sweetalert2.min.css";

Vue.config.productionTip = false;
Vue.use(VueKonva);
Vue.use(VueSweetalert2);
Vue.use(VueClipboard);

// EventBus 생성
Vue.prototype.$EventBus = new Vue();

new Vue({
  router,
  store,
  vuetify,
  render: h => h(App)
}).$mount("#app");
