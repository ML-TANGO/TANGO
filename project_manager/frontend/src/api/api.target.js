import axios from "./axios";

export async function getTargetList() {
  let response = null;

  try {
    response = await axios.get("/api/targets/target_read");
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}

/**
 * target 생성
 * @param {formdata} formData
 * @returns
 */
export async function createTarget(param) {
  let response = null;
  const formData = new FormData();
  formData.append("name", param.name);
  formData.append("image", param.image);
  formData.append("info", param.info);
  formData.append("engine", param.engine);
  formData.append("os", param.os);
  formData.append("cpu", param.cpu);
  formData.append("acc", param.acc);
  formData.append("memory", param.memory);
  formData.append("nfs_ip", param.nfs_ip || "");
  formData.append("nfs_path", param.nfs_path || "");
  formData.append("host_ip", param.host_ip);
  formData.append("host_port", param.host_port);
  formData.append("host_service_port", param.host_service_port);

  try {
    response = await axios.post("/api/targets/target_create", formData, {
      headers: { "Content-Type": "multipart/form-data" }
    });
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}

/**
 * target 수정
 * @param {formdata} formData
 * @returns
 */
export async function updateTarget(param) {
  let response = null;
  const formData = new FormData();
  formData.append("id", param.id);
  formData.append("name", param.name);
  formData.append("image", param.image);
  formData.append("info", param.info);
  formData.append("engine", param.engine);
  formData.append("os", param.os);
  formData.append("cpu", param.cpu);
  formData.append("acc", param.acc);
  formData.append("memory", param.memory);
  formData.append("nfs_ip", param.nfs_ip || "");
  formData.append("nfs_path", param.nfs_path || "");
  formData.append("host_ip", param.host_ip);
  formData.append("host_port", param.host_port);
  formData.append("host_service_port", param.host_service_port);

  try {
    response = await axios.post("/api/targets/target_update", formData, {
      headers: { "Content-Type": "multipart/form-data" }
    });
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}

export async function deleteTarget(id) {
  let response = null;

  try {
    response = await axios.delete("/api/targets/target_delete", { data: { id: id } });
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}

export async function getTargetInfo(id) {
  let response = null;

  try {
    response = await axios.post("/api/targets/target_info", { id: id });
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}
