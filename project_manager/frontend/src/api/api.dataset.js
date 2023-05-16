import axios from "axios";

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
