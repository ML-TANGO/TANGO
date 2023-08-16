import createInitialState from "./state";
import getters from "./getters";
import mutations, { Mutations } from "./mutations";
import actions, { Actions } from "./actions";

export const ProjectNamespace = "project";
export const ProjectMutations = Mutations;
export const ProjectActions = Actions;

export default {
  state: () => createInitialState(),
  getters,
  mutations,
  actions,
  namespaced: true
};
