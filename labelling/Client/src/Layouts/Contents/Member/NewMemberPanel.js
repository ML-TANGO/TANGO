import React, { useState, useEffect } from "react"
import { useForm } from "react-hook-form"
import { toast } from "react-toastify"

import { MdError, MdSave } from "react-icons/md"

import CommonPanel from "../../../Components/Panel/CommonPanel"
import { FormText, FormSelect } from "../../../Components/Form/FormComponent"
import CommonToast from "../../../Components/Common/CommonToast"

import * as AuthApi from "Config/Services/AuthApi"

import { RiSendPlaneFill } from "react-icons/ri"

import CommonButton from "../../../Components/Common/CommonButton"

const textCount = str => {
  let byte = 0
  for (let i = 0; i < str.length; i += 1) {
    str.charCodeAt(i) > 127 ? (byte += 3) : byte++
  }
  return byte
}

const roleOptions = [
  { label: "ADMIN", value: "ADMIN" },
  { label: "USER", value: "USER" }
]

const useOptions = [
  { label: "Y", value: "1" },
  { label: "N", value: "2" }
]

function NewMemberPanel(props) {
  const { panelToggle, springProps, editData, initMember } = props
  const [pageState, setPageState] = useState({
    title: "New Member",
    isSave: false,
    pageMode: "NEW"
  })

  const [panelPos, setPanelPos] = useState({ top: 0, bottom: 0 })
  const { register, handleSubmit, control, setValue, errors, setError } = useForm()

  useEffect(() => {
    async function init() {
      try {
        setPageState(prevState => ({ ...prevState, pageMode: editData.pageMode }))
        switch (editData.pageMode) {
          case "NEW":
            break
          case "EDIT":
            setPageState(prevState => ({ ...prevState, title: "Edit Service" }))
            const result = await AuthApi._getUsers({ USER_ID: editData.dataInfo.USER_ID })
            const userInfo = result[0]
            setValue("USER_ID", userInfo.USER_ID)
            setValue("USER_NM", userInfo.USER_NM)
            setValue("ROLE", userInfo.ROLE)
            setValue("USE", userInfo.USE !== "2" ? "1" : "2")
            break
        }
      } catch (e) {
        console.log(e)
      }
    }
    init()
  }, [])

  const onSubmit = async data => {
    setPageState(prevState => ({ ...prevState, isSave: true }))
    switch (editData.pageMode) {
      case "NEW":
        const count = await AuthApi._checkUser({ USER_ID: data.USER_ID })
        if (count[0]?.COUNT !== 0) {
          setError("USER_ID", "validateIdCheck", "id check")
          setPageState(prevState => ({ ...prevState, isSave: false }))
          return
        }
        AuthApi._setUser(data)
          .then(async result => {
            console.log(result)
            if (result.status === 1) {
              toast.info(<CommonToast Icon={MdSave} text={"Create User Success"} />)
              setPageState(prevState => ({ ...prevState, isSave: false }))
              initMember()
              panelToggle()
            } else {
              throw { err: "status 0" }
            }
          })
          .catch(e => {
            toast.error(<CommonToast Icon={MdError} text={"Create User Fail"} />)
            setPageState(prevState => ({ ...prevState, isSave: false }))
            console.log(e)
          })
        break
      case "EDIT":
        AuthApi._updateUser(data)
          .then(result => {
            if (result.status === 1) {
              toast.info(<CommonToast Icon={MdSave} text={"Update User Success"} />)
              setPageState(prevState => ({ ...prevState, isSave: false }))
              initMember()
              panelToggle()
            } else {
              throw { err: "status 0" }
            }
          })
          .catch(e => {
            toast.error(<CommonToast Icon={MdError} text={"Update User Fail"} />)
            setPageState(prevState => ({ ...prevState, isSave: false }))
            console.log(e)
          })
        break
    }
  }

  const top = (
    <div className="form pr-2">
      <FormText
        title="USER ID"
        titleClassName={"mr-4 mt-2"}
        name="USER_ID"
        register={register({
          required: true,
          validate: {
            validateTrim: value => String(value).trim().length !== 0,
            validateLength: value => textCount(value) < 100
          }
        })}
        errors={errors}
        disabled={editData.pageMode === "EDIT"}
      />
      <FormText
        title="USER NAME"
        titleClassName={"mr-4 mt-2"}
        name="USER_NM"
        register={register({
          required: true,
          validate: {
            validateTrim: value => String(value).trim().length !== 0,
            validateLength: value => textCount(value) < 100
          }
        })}
        errors={errors}
      />
      <FormText
        title="PASSWORD"
        titleClassName={"mr-4 mt-2"}
        name="USER_PW"
        type="password"
        register={register(
          editData.pageMode !== "EDIT" && {
            required: true,
            validate: {
              validateTrim: value => String(value).trim().length !== 0,
              validateLength: value => textCount(value) < 256
              // pwCheck: value => value.length >= 8 && /(?=.*[a-z])(?=.*[A-Z])(?=.*[0-9])(?=.*[^\w\s])/.test(value),
              // continuNumber: value => !/(012|123|234|345|456|567|678|789)/.test(value),
              // continuString: value =>
              //   !/(abc|bcd|cdf|def|efg|fgh|ghi|hij|ijk|jkl|klm|lmn|mno|noq|opq|pqr|qrs|rst|stu|tuv|uvw|vwx|wxy|xyz)/.test(value),
              // sameString: value => !/(\w)\1\1/.test(value)
            }
          }
        )}
        errors={errors}
      />
      <FormSelect
        title="ROLE"
        titleClassName={"mr-4 mt-2"}
        name="ROLE"
        control={control}
        options={roleOptions}
        isDefault={true}
        onChange={([selected]) => selected}
      />
      <FormSelect
        title="USE"
        titleClassName={"mr-4 mt-2"}
        name="USE"
        control={control}
        options={useOptions}
        isDefault={true}
        onChange={([selected]) => selected}
      />
    </div>
  )

  const tail = (
    <>
      <div className="line-separator mx-2 mt-2" />
      <CommonButton
        ButtonIcon={RiSendPlaneFill}
        className="bg-green float-right"
        text="Apply"
        onClick={handleSubmit(onSubmit)}
        disabled={pageState.isUpload}
      />
    </>
  )

  return (
    <>
      <CommonPanel
        title={pageState.title}
        isSave={pageState.isSave}
        loadingText="Create Member"
        panelToggle={panelToggle}
        springProps={springProps}
        panelPos={panelPos}
        setPanelPos={setPanelPos}
        top={top}
        tail={tail}
      />
    </>
  )
}

NewMemberPanel.propTypes = {}

export default NewMemberPanel
