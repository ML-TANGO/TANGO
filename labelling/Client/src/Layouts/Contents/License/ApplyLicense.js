import React, { useState } from "react"
import { useForm } from "react-hook-form"
import { RiMailLine, RiUser3Line, RiBuilding4Line, RiLuggageDepositLine, RiBriefcaseLine } from "react-icons/ri"
import PrivacyPolicyModal from "./PrivacyPolicyModal"
import { useHistory } from "react-router-dom"
import { toast } from "react-toastify"
import { MdInfo, MdError } from "react-icons/md"

import * as AuthApi from "../../../Config/Services/AuthApi"
import CommonIconInput from "../../../Components/Common/CommonIconInput"
import CommonToast from "../../../Components/Common/CommonToast"

function ApplyLicense() {
  const [isModal, setIsModal] = useState(false)
  const { register, handleSubmit, errors } = useForm()
  const history = useHistory()

  const handleClick = data => {
    data.BUILD = process.env.BUILD
    AuthApi._setNewLicense(data)
      .then(result => {
        if (result.status === 1) {
          // 발급 성공
          toast.info(<CommonToast Icon={MdInfo} text={"License key issuance completed.\n Check your mail"} />, {
            onClose: () => history.push({ pathname: "/registerLicense" })
          })
        } else if (result.status === 3) {
          // 이미 발급받았다 컴퓨터에서
          toast.error(<CommonToast Icon={MdError} text={"License Key has already been issued"} />)
        } else {
          toast.error(<CommonToast Icon={MdError} text={"License Key issuance failed"} />)
        }
      })
      .catch(e => {
        toast.error(<CommonToast Icon={MdError} text={"License Key issuance failed"} />)
        console.log(e)
      })
  }

  const toggle = () => {
    setIsModal(isModal => !isModal)
  }

  return (
    <>
      <PrivacyPolicyModal toggle={toggle} modal={isModal} />
      <div className="theme-light ltr-support without-y-scrollbar">
        <div className="account" style={{ background: "radial-gradient( #364261, #2e2e2e, #000)" }}>
          <div className="account__wrapper" style={{ width: "500px" }}>
            <div className="account__card box_shadow">
              <div className="license-header">
                <h3>Welcome to BluAi</h3>
                <h4 className="subhead">Apply for a license</h4>
              </div>
              <div className="form form--horizontal">
                <span className="form__form-group-label">Name</span>
                <CommonIconInput
                  className="mt-1 mb-1"
                  name="NAME"
                  placeholder="User Name"
                  register={register({ required: true })}
                  icon={<RiUser3Line style={{ fontSize: "17px" }} />}
                />
                {errors.NAME?.type === "required" && <div className="form__form-group-label form-error mt-1">Name is Required</div>}
                <span className="form__form-group-label mt-2">E-mail</span>
                <CommonIconInput
                  className="mt-1 mb-1"
                  name="EMAIL"
                  placeholder="example@weda.kr"
                  register={register({
                    required: true,
                    pattern: /^[0-9a-zA-Z]([-_.]?[0-9a-zA-Z])*@[0-9a-zA-Z]([-_.]?[0-9a-zA-Z])*\.[a-zA-Z]{2,3}$/i
                  })}
                  icon={<RiMailLine style={{ fontSize: "17px" }} />}
                />
                {errors.EMAIL?.type === "required" && <div className="form__form-group-label form-error mt-1">E-mail is Required</div>}
                {errors.EMAIL?.type === "pattern" && (
                  <div className="form__form-group-label form-error mt-1">E-mail format is not valid</div>
                )}
                <span className="form__form-group-label mt-2">Company</span>
                <CommonIconInput
                  className="mt-1 mb-1"
                  name="COMPANY"
                  placeholder="Company (회사)"
                  register={register({ required: true })}
                  icon={<RiBuilding4Line style={{ fontSize: "17px" }} />}
                />
                {errors.COMPANY?.type === "required" && <div className="form__form-group-label form-error mt-1">Company is Required</div>}
                <span className="form__form-group-label mt-2">Position</span>
                <CommonIconInput
                  className="mt-1 mb-1"
                  name="JOB_POSITION"
                  placeholder="Position (직책)"
                  register={register({ required: true })}
                  icon={<RiLuggageDepositLine style={{ fontSize: "17px" }} />}
                />
                {errors.JOB_POSITION?.type === "required" && (
                  <div className="form__form-group-label form-error mt-1">Position is Required</div>
                )}
                <span className="form__form-group-label mt-2">Rank</span>
                <CommonIconInput
                  className="mt-1 mb-1"
                  name="JOB_RANK"
                  placeholder="Rank (직급)"
                  register={register({ required: true })}
                  icon={<RiBriefcaseLine style={{ fontSize: "17px" }} />}
                />
                {errors.JOB_RANK?.type === "required" && <div className="form__form-group-label form-error mt-1">Rank is Required</div>}
                <div className="w-100 d-flex mb-2 mt-2" style={{ justifyContent: "space-between" }}>
                  <div className="d-flex">
                    <input
                      type="checkbox"
                      name="CHECKAGREE"
                      className="mr-2"
                      style={{ width: "15px", height: "15px" }}
                      ref={register({ validate: value => value === true })}
                      readOnly
                    />
                    <h5 style={{ color: "#888", textAlign: "center", marginTop: "-2px" }}>개인정보 수집 및 이용 동의 (필수)</h5>
                  </div>
                  <div
                    className="float-right hover-cursor"
                    style={{ color: "white", marginTop: "-2px", textAlign: "right" }}
                    onClick={toggle}
                  >
                    보기
                  </div>
                </div>
                {errors.CHECKAGREE?.type === "validate" && (
                  <div className="form__form-group-label form-error mt-1">개인정보 수집 및 이용 동의는 필수입니다</div>
                )}
                <div className="w-100" style={{ background: "none" }}>
                  <button type="button" className={`form__form-group-button account__login`} onClick={handleSubmit(handleClick)}>
                    APPLY
                  </button>
                </div>
                <div className="mt-2">
                  <h6 style={{ color: "#888", textAlign: "center", marginTop: "-2px" }}>신청한 라이센스 키는 이메일로 발송됩니다</h6>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}

ApplyLicense.propTypes = {}

export default ApplyLicense
