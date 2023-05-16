import axios from "axios";
import request from "./axios";
import { ContainerPort } from "@/shared/enums";

export async function startContainer(container, uid, pid) {
  let response = null;
  const host = window.location.hostname;
  const port = ContainerPort[container];

  try {
    response = await axios.get(
      "http://" + host + ":" + port + "/start",
      { params: { user_id: uid, project_id: pid } },
      { withCredentials: true }
    );
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}

export async function stopContainer(container, uid, pid) {
  let response = null;
  const host = window.location.hostname;
  const port = ContainerPort[container];

  try {
    response = await axios.get(
      "http://" + host + ":" + port + "/stop",
      { params: { user_id: uid, project_id: pid } },
      { withCredentials: true }
    );
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}

export async function checkContainerStatus(container, uid, pid) {
  let response = null;
  const host = window.location.hostname;
  const port = ContainerPort[container];

  try {
    response = await axios.get(
      "http://" + host + ":" + port + "/status_request",
      { params: { user_id: uid, project_id: pid } },
      { withCredentials: true }
    );
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}

export async function updateContainerStatus(param) {
  let response = null;

  try {
    response = await request.post("/api/status_update", param);
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}

/* 컨테이너 상태 요청 */
export async function getStatusResult(project_id) {
  let response = null;

  try {
    response = await request.post("/api/status_result", { project_id: project_id });
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}

/* 컨테이너 상태 요청 */
export async function postStatusRequest(data) {
  let response = null;

  try {
    response = await request.post("/api/status_request", data);
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}
