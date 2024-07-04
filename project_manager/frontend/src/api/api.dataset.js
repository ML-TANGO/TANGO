import axios from "./axios";

export async function getDatasetList() {
  let response = null;
  const host = window.location.hostname;

  try {
    response = await axios.post("http://" + host + ":8095" + "/api/dataset/getDataSetList", {
      headers: ""
    });
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}

export async function getDatasetListTango() {
  let response = null;
  try {
    response = await axios.get("/api/datasets/get_dataset_list");
  } catch (error) {
    throw new Error(error);
  }

  return response.data.datasets;
}

export async function getDatasetInfo(name) {
  let response = null;
  try {
    response = await axios.get("/api/datasets/get_dataset_info", { params: { name: name } });
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}

export async function getDatasetFolderSize(folderList) {
  let response = null;
  try {
    response = await axios.post("/api/datasets/get_folders_size", { folder_list: folderList });
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}

export async function getDatasetFileCount(folderList) {
  let response = null;
  try {
    response = await axios.post("/api/datasets/get_folders_file_count", { folder_list: folderList });
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}

export async function cocoDatasetDownload() {
  let response = null;
  try {
    response = await axios.post("/api/datasets/download_coco", {
      isTrain: true,
      isVal: true,
      isTest: false,
      isSegments: false,
      isSama: false
    });
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}

export async function imagenetDatasetDownload() {
  let response = null;
  try {
    response = await axios.post("/api/datasets/download_imagenet", { isTrain: true, isVal: true });
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}

export async function vocDatasetDownload() {
  let response = null;
  try {
    response = await axios.post("/api/datasets/download_voc");
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}

export async function chestXrayDatasetDownload() {
  let response = null;
  try {
    response = await axios.post("/api/datasets/download_chest_xray_dataset");
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}

export async function checkExistKaggleInfo() {
  let response = null;
  try {
    response = await axios.get("/api/datasets/is_exist_user_kaggle_json");
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}

export async function createUserKaggleInfo(username, key) {
  let response = null;
  try {
    response = await axios.post("/api/datasets/setup_user_kaggle_api", { username, key });
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}
