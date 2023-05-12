import Cookies from "universal-cookie";

const refresh = async config => {
  const token = getToken();
  if (token) config.headers["Authorization"] = `Bearer ${token}`;
  config.headers["Content-Type"] = `multipart/form-data`;
  return config;
};

const refreshErrorHandle = () => {
  const cookie_info = new Cookies();
  cookie_info.remove("userinfo", { path: "/" });
  cookie_info.remove("TANGO_TOKEN", { path: "/" });
};

function getToken() {
  const token = new Cookies().get("TANGO_TOKEN");
  return token ? token : null;
}

export { refresh, refreshErrorHandle };
