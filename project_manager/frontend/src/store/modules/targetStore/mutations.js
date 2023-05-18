export const Mutations = {
  SET_TARGET: "SET_TARGET",
  INIT_TARGET: "INIT_TARGET"
};

const mutations = {
  [Mutations.SET_TARGET](state, data) {
    state.target = { ...state.target, ...data };
  },

  [Mutations.INIT_TARGET](state) {
    state.target = {
      host_ip: "",
      host_port: "",
      host_service_port: ""
    };
  }
};

export default mutations;
