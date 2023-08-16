import createInitialState from "./state";
import getters from "./getters";
import mutations, { Mutations } from "./mutations";
import actions, { Actions } from "./actions";

export const UserNamespace = "user";
export const UserMutations = Mutations;
export const UserActions = Actions;

export default {
  state: () => createInitialState(),
  getters,
  mutations,
  actions,
  namespaced: true
};
