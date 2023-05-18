import createInitialState from "./state";
import getters from "./getters";
import mutations, { Mutations } from "./mutations";
import actions, { Actions } from "./actions";

export const TargetNamespace = "target";
export const TargetMutations = Mutations;
export const TargetActions = Actions;

export default {
  state: () => createInitialState(),
  getters,
  mutations,
  actions,
  namespaced: true
};
