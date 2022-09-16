import React from "react"
import { useForm } from "react-hook-form"
import { FaLock, FaUser } from "react-icons/fa"
import axios from "axios"
import { toast } from "react-toastify"
import { MdInfo, MdError } from "react-icons/md"
import { useHistory } from "react-router-dom"

import CommonIconInput from "../Components/Common/CommonIconInput"
import * as AuthApi from "../Config/Services/AuthApi"
import CommonToast from "../Components/Common/CommonToast"
function LogInLayout() {
  const { register, handleSubmit, reset, errors } = useForm()
  const history = useHistory()

  const handleLogin = async data => {
    try {
      const result = await AuthApi._login(data)
      if (result.STATUS === 1) {
        window.sessionStorage.setItem("userInfo", JSON.stringify(result.user))
        // window.sessionStorage.setItem("token", result.token)
        window.sessionStorage.setItem("refreshToken", result.refreshToken)
        axios.defaults.headers.common["Authorization"] = "Bearer " + result.token
        toast.info(<CommonToast Icon={MdInfo} text={"login success"} />)
        history.push({
          pathname: "/dashboard"
        })
      }
    } catch (e) {
      reset({ USER_ID: "", USER_PW: "" })
      toast.error(<CommonToast Icon={MdError} text={e.data.msg ? e.data.msg : e.data} />)
      console.log(e)
    }
  }

  return (
    <div className="theme-light ltr-support without-y-scrollbar">
      <div className="account" style={{ background: "radial-gradient( #364261, #2e2e2e, #000)" }}>
        <div className="account__wrapper">
          <div className="account__card box_shadow">
            <div className="account__head">
              <h3>Welcome To BluAi</h3>
              <h4 className="subhead">Weda Ai Platform</h4>
            </div>
            <div className="form form--horizontal">
              <CommonIconInput
                name={"USER_ID"}
                className="mb-0"
                inputClassName="account__input"
                register={register({ required: true })}
                placeholder={"User ID"}
                icon={<FaUser />}
              />
              {errors.USER_ID?.type === "required" && <span className="form-error no-panel-error mt-1">Please enter your ID</span>}
              <CommonIconInput
                name={"USER_PW"}
                className="mb-0 mt-3"
                inputClassName="account__input"
                register={register({ required: true })}
                placeholder={"Password"}
                isPassword={true}
                icon={<FaLock />}
                onKeyDown={e => {
                  if (e.key === "Enter") {
                    handleSubmit(handleLogin)()
                  }
                }}
              />
              {errors.USER_PW?.type === "required" && <span className="form-error no-panel-error mt-1">Please enter a password</span>}
              <div className="w-100 mt-3" style={{ background: "none" }}>
                <button type="button" className={`form__form-group-button account__login`} onClick={handleSubmit(handleLogin)}>
                  LOGIN
                </button>
              </div>
            </div>
            {/* <h5 style={{ color: "#888", textAlign: "center", marginTop: "10px" }}>
              Don't have an account?{" "}
              <span className="count__register" style={{ color: "#1777CB" }}>
                Register
              </span>
            </h5> */}
          </div>
        </div>
      </div>
    </div>
  )
}

LogInLayout.propTypes = {}

export default LogInLayout
