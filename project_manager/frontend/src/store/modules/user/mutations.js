export const Mutations = {
  SET_USERINFO: "SET_USERINFO"
};

const mutations = {
  [Mutations.SET_USERINFO](state, data) {
    state.user = data;
  }
};

export default mutations;
