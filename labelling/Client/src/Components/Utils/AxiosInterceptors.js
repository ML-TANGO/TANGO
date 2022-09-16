import axios from "axios"
import { toast } from "react-toastify"

let isRefreshing = false
let failedQueue = []

const processQueue = (error, token = null) => {
  failedQueue.forEach(prom => {
    if (error) {
      prom.reject(error)
    } else {
      prom.resolve(token)
    }
  })

  failedQueue = []
}

axios.interceptors.response.use(
  function (response) {
    return response
  },
  async function (error) {
    const originalRequest = error.config
    // 권한 없음 에러일 경우
    if (error.response && error.response.status == 401 && !originalRequest._retry) {
      //갱신 중일때
      if (isRefreshing) {
        return new Promise(function (resolve, reject) {
          failedQueue.push({ resolve, reject })
        })
          .then(token => {
            originalRequest.headers["Authorization"] = "Bearer " + token
            return axios(originalRequest)
          })
          .catch(err => {
            return Promise.reject(err)
          })
      }

      originalRequest._retry = true
      isRefreshing = true

      return new Promise((resolve, reject) => {
        axios
          .post("/api/auth/refresh", { refreshToken: window.sessionStorage.getItem("refreshToken") })
          .then(result => {
            window.sessionStorage.setItem("userInfo", JSON.stringify(result.data.user))
            // window.sessionStorage.setItem("token", result.data.token)
            window.sessionStorage.setItem("refreshToken", result.data.refreshToken)

            axios.defaults.headers.common["Authorization"] = "Bearer " + result.data.token
            originalRequest.headers["Authorization"] = "Bearer " + result.data.token

            processQueue(null, result.data.token)

            resolve(axios(originalRequest))
          })
          // 토큰 받아오기 실패할 경우
          .catch(err => {
            if (err.response.status == 400) {
              toast.error("Login time has expired", {
                autoClose: 2000,
                onClose: () => {
                  window.location.replace("/")
                }
              })
              reject(err)
            } else {
              processQueue(err, null)
              reject(err)
            }
          })
          .finally(() => {
            isRefreshing = false
          })
      })
    } else {
      return Promise.reject(error)
    }
  }
)

const AxiosInterceptors = () => {
  // alert = useAlert()
}
export default AxiosInterceptors
