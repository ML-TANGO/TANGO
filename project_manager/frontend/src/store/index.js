import Vue from "vue";
import Vuex from "vuex";

import userStore from "@/store/modules/user";
import projectStore from "@/store/modules/project";
import targetStore from "@/store/modules/target";

Vue.use(Vuex);

export default new Vuex.Store({
  state: {
    loading: false
  },
  getters: {},
  mutations: {
    setLoding(state, payload) {
      state.loading = payload;
    }
  },
  actions: {},
  modules: {
    user: userStore,
    project: projectStore,
    target: targetStore
  }
});
