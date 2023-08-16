import axios from "./axios";

export async function userLoginAPI(id, pw) {
  let response = null;

  try {
    response = await axios.post("/api/login", { user_id: id, password: pw });
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}

export async function isDuplicateIDAPI(id) {
  let response = null;

  try {
    response = await axios.post("/api/user_id_check", { id: id });
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}

export async function createAccountAPI(id, email, password) {
  let response = null;

  try {
    response = await axios.post("/api/signup", {
      id: id,
      email: email,
      password: password
    });
  } catch (error) {
    throw new Error(error);
  }

  return response.data;
}
