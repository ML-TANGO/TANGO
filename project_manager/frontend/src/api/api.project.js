import axios from "./axios";

export async function getProjectList() {
  let response = null;

  try {
    response = await axios.get("/api/project_list_get");
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}

export async function getProjectInfo(id) {
  let response = null;

  try {
    response = await axios.post("/api/project_info", { id: id });
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}

export async function createProject(name, description) {
  let response = null;

  try {
    response = await axios.post("/api/project_create", { project_name: name, project_description: description });
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}

export async function updateProjectInfo(data) {
  let response = null;

  try {
    response = await axios.post("/api/project_update", data);
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}

export async function updateProjectName(id, name) {
  let response = null;

  try {
    response = await axios.post("/api/project_rename/", { id: id, name: name });
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}

export async function updateProjectDescription(id, description) {
  let response = null;

  try {
    response = await axios.post("/api/project_description_update", { id: id, description: description });
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}

export async function deleteProject(id) {
  let response = null;

  try {
    response = await axios.post("/api/project_delete", { id: id });
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}
