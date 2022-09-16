import React from "react"
import { RiKey2Line } from "react-icons/ri"
import { useHistory } from "react-router-dom"
import { useForm } from "react-hook-form"
import { toast } from "react-toastify"
import { MdInfo, MdError } from "react-icons/md"

import * as AuthApi from "../../../Config/Services/AuthApi"

import CommonIconInput from "../../../Components/Common/CommonIconInput"
import CommonToast from "../../../Components/Common/CommonToast"

function RegisterLicense() {
  const history = useHistory()
  const { register, handleSubmit, errors } = useForm()

  const handleClick = data => {
    AuthApi._setAuthentication(data)
      .then(result => {
        if (result.status === 1) {
          // 인증 완료
          toast.info(<CommonToast Icon={MdInfo} text={"License verification is complete"} />, {
            onClose: () => history.push({ pathname: "/downloadPretrain" })
          })
        } else {
          toast.error(<CommonToast Icon={MdError} text={"Invalid License Key"} />)
          history.push({ pathname: "/downloadPretrain" })
        }
      })
      .catch(e => {
        console.log(e)
        toast.error(<CommonToast Icon={MdError} text={"Invalid License Key"} />)
      })
  }

  const handleApplyClick = () => {
    history.push({ pathname: "/applyLicense" })
  }

  const handleKeyDown = e => {
    if (e.keyCode === 13) {
      handleSubmit(handleClick)()
    }
  }

  return (
    <div className="theme-light ltr-support without-y-scrollbar">
      <div className="account" style={{ background: "radial-gradient( #364261, #2e2e2e, #000)" }}>
        <div className="account__wrapper" style={{ width: "500px" }}>
          <div className="account__card box_shadow">
            <div className="license-header">
              <h3>Welcome to BluAi</h3>
              <h4 className="subhead">Enter your a license</h4>
            </div>
            <div className="form form--horizontal">
              <CommonIconInput
                className="mt-1 mb-1"
                name="LICENSE_CODE"
                placeholder="License Key"
                icon={<RiKey2Line style={{ fontSize: "17px" }} />}
                register={register({ required: true })}
                onKeyDown={handleKeyDown}
              />
              {errors.license?.type === "required" && <div className="form__form-group-label form-error mt-1">License is Required</div>}
              <div className="w-100 mt-2" style={{ background: "none" }}>
                <button type="button" className={`form__form-group-button account__login`} onClick={handleSubmit(handleClick)}>
                  ENTER
                </button>
              </div>
            </div>
            <h5 style={{ color: "#888", textAlign: "center", marginTop: "10px" }}>
              Don&apos;t have a License?{" "}
              <span className="count__register" style={{ color: "#1777CB" }} onClick={handleApplyClick}>
                Apply
              </span>
            </h5>
          </div>
        </div>
      </div>
    </div>
  )
}

RegisterLicense.propTypes = {}

export default RegisterLicense
