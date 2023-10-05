import axios from "axios";
import request from "./axios";
import { ContainerPort } from "@/shared/enums";

export async function startContainer(container, uid, pid) {
  let response = null;
  // const host = window.location.hostname;
  // const port = ContainerPort[container];

  const host = window.location.hostname;
  const port = 8888;

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

  const timeout = 4900;

  try {
    response = await request.post("/api/status_request", data, { timeout });
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}

/* 컨테이너 상태 요청 */
/**
 * user_id , project_id , container_id
 * @param {*} data
 * @returns
 */
export async function containerStart(container, user, project) {
  let response = null;

  try {
    response = await request.post("/api/container_start", {
      user_id: user,
      project_id: project,
      container_id: container
    });
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}
