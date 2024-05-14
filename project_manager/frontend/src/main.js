import Vue from "vue";
import App from "./App.vue";
import router from "./router";
import store from "./store";
import vuetify from "./plugins/vuetify";
import VueKonva from "vue-konva";
import VueClipboard from "vue-clipboard2";

Vue.config.productionTip = false;
Vue.use(VueKonva);
Vue.use(VueClipboard);

// EventBus 생성
Vue.prototype.$EventBus = new Vue();

new Vue({
  router,
  store,
  vuetify,
  render: h => h(App)
}).$mount("#app");
