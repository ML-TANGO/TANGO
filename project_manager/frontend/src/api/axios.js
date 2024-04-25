const Axios = require("axios");
import { refresh, refreshErrorHandle } from "./refreshToken";
import Cookies from "universal-cookie";

const baseURL = process.env.NODE_ENV === "production" ? "" : process.env.VUE_APP_ROOT_API;

const axios = Axios.create({
  baseURL: baseURL
});

axios.interceptors.request.use(refresh, refreshErrorHandle);
axios.interceptors.response.use(
  function (response) {
    return response;
  },
  function (error) {
    if (error.response.status === 401) {
      const cookie_info = new Cookies();
      cookie_info.remove("userinfo", { path: "/" });
      cookie_info.remove("TANGO_TOKEN", { path: "/" });
      window.location.reload();
    }
    return Promise.reject(error);
  }
);
export default axios;
