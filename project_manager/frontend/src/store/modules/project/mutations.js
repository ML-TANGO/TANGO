export const Mutations = {
  SET_PROJECT: "SET_PROJECT",
  INIT_PROJECT: "INIT_PROJECT",
  SET_SELECTED_TARGET: "SET_SELECTED_TARGET",
  SET_SELECTED_IMAGE: "SET_SELECTED_IMAGE"
};

const mutations = {
  [Mutations.SET_PROJECT](state, data) {
    state.project = { ...state.project, ...data };
  },

  [Mutations.INIT_PROJECT](state) {
    state.project = {};
  },

  [Mutations.SET_SELECTED_TARGET](state, data) {
    state.selectedTarget = data;
  },

  [Mutations.SET_SELECTED_IMAGE](state, data) {
    state.selectedImage = data;
  }
};

export default mutations;
