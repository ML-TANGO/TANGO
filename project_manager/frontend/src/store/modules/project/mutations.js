export const Mutations = {
  SET_PROJECT: "SET_PROJECT",
  INIT_PROJECT: "INIT_PROJECT",
  SET_SELECTED_TARGET: "SET_SELECTED_TARGET",
  SET_SELECTED_IMAGE: "SET_SELECTED_IMAGE",
  SET_AUTO_NN_STATUS: "SET_AUTO_NN_STATUS"
};

const mutations = {
  [Mutations.SET_PROJECT](state, data) {
    state.project = { ...state.project, ...data };
  },

  [Mutations.INIT_PROJECT](state) {
    state.project = {};
    state.selectedTarget = {};
    state.selectedImage = {};
  },

  [Mutations.SET_SELECTED_TARGET](state, data) {
    state.selectedTarget = data;
  },

  [Mutations.SET_SELECTED_IMAGE](state, data) {
    state.selectedImage = data;
  },

  [Mutations.SET_AUTO_NN_STATUS](state, data) {
    state.autonn_status = data;
  }
};

export default mutations;
