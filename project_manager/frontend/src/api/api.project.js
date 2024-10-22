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

export async function createProject(name, description = "") {
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

export async function updateProjectType(id, type) {
  let response = null;

  try {
    response = await axios.post("/api/project_type", { id: id, type: type });
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}

export async function downloadNNModel(user_id, project_id) {
  let response = null;

  try {
    response = await axios.get("/api/download_nn_model", {
      params: { user_id: user_id, project_id: project_id },
      responseType: "blob"
    });
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}

export async function uploadNNModel(user_id, project_id, file) {
  let response = null;

  const formData = new FormData();
  formData.append("user_id", user_id);
  formData.append("project_id", project_id);
  formData.append("nn_model", file);

  try {
    response = await axios.post("/api/upload_nn_model", formData, {
      headers: { "Content-Type": "multipart/form-data" }
    });
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}

export async function setWorkflow(project_id, workflow) {
  let response = null;

  try {
    response = await axios.post("/api/set_workflow", { project_id: project_id, workflow: workflow });
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}

export async function get_autonn_status(project_id) {
  let response = null;

  try {
    response = await axios.post("/api/get_autonn_status", { project_id: project_id });
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}

export async function get_common_folder_structure() {
  let response = null;

  try {
    response = await axios.get("/api/get_common_folder_structure");
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}
