import React, { useState, useEffect, useMemo } from "react"
import { Container, Row, Col, Modal, ModalHeader, ModalBody, Input } from "reactstrap"
import PropTypes from "prop-types"
import { FaSave } from "react-icons/fa"
import { useForm, Controller } from "react-hook-form"
import { HuePicker } from "react-color"
import randomcolor from "randomcolor"
import { toast } from "react-toastify"
import { MdSave, MdError } from "react-icons/md"
import { useDispatch } from "react-redux"

import { RiPriceTag3Line } from "react-icons/ri"
import { BsFillCircleFill } from "react-icons/bs"

import * as ImageAnnoApi from "Config/Services/ImageAnnoApi"
import CommonToast from "../../../../../Components/Common/CommonToast"
import CommonSelect from "../../../../../Components/Common/CommonSelect"
import CommonButton from "../../../../../Components/Common/CommonButton"

import * as ImageLabelActions from "Redux/Actions/ImageLabelActions"
import * as VideoLabelActions from "Redux/Actions/VideoLabelActions"

function validObject(obj, t, f) {
  if (obj !== "" || obj !== null || obj !== undefined) {
    return t
  } else {
    return f
  }
}

function TagModal(props) {
  const [color, setColor] = useState(props.tagInfo !== null ? props.tagInfo.COLOR : randomcolor())
  const [category1, setCategory1] = useState([])
  const [category2, setCategory2] = useState([])
  const [category3, setCategory3] = useState([])
  const [isSave, setIsSave] = useState(false)

  // DataType Image, Video 구분
  const dispatch = useDispatch()
  const action = useMemo(() => (props.dataType === "I" ? ImageLabelActions : VideoLabelActions), [props.dataType])

  const { register, handleSubmit, setValue, getValues, control, errors } = useForm({
    defaultValues: {
      NAME: props.tagInfo !== null ? props.tagInfo.NAME : "",
      DESC_TXT: props.tagInfo !== null ? props.tagInfo.DESC_TXT : "",
      COLOR: props.tagInfo !== null ? props.tagInfo.COLOR : color,
      CATEGORY1: props.tagInfo !== null ? validObject(props.tagInfo.CATEGORY1, Number(props.tagInfo.CATEGORY1), "") : "",
      CATEGORY2: props.tagInfo !== null ? validObject(props.tagInfo.CATEGORY2, Number(props.tagInfo.CATEGORY2), "") : "",
      CLASS_CD: props.tagInfo !== null ? validObject(props.tagInfo.CLASS_CD, Number(props.tagInfo.CLASS_CD), "") : ""
    }
  })

  useEffect(() => {
    if (props.mode === "I") {
      ImageAnnoApi._getCategory({ DATA_TYPE: props.dataType, OBJECT_TYPE: props.objectType })
        .then(data => {
          setCategory1(data)
        })
        .catch(e => console.log(e))
    } else if (props.mode === "U") {
      setCategory1(props.tagInfo.CATEGORY1_LIST)
      if (validObject(props.tagInfo.CATEGORY2, true, false)) setCategory2(props.tagInfo.CATEGORY2_LIST)
      if (validObject(props.tagInfo.CLASS_CD, true, false)) setCategory3(props.tagInfo.CATEGORY3_LIST)
    }
  }, [])

  useEffect(() => {
    register({ name: "COLOR" })
  }, [register])

  useEffect(() => {
    setValue("COLOR", color)
  }, [color])

  const changeColor = color => {
    setColor(color.hex)
  }

  const onSubmit = data => {
    if (isSave) return
    setIsSave(true)
    data.DATASET_CD = props.dataSetCd
    if (props.mode === "I") {
      ImageAnnoApi._setDataTag(data)
        .then(result => {
          if (result.status === 1) {
            toast.info(<CommonToast Icon={MdSave} text={"Tag Create Success"} />)
            setIsSave(false)
            props._successModal()
          } else {
            throw { err: "status 0" }
          }
        })
        .catch(e => {
          console.log(e)
          setIsSave(false)
          toast.error(<CommonToast Icon={MdError} text={"Tag Create Fail"} />)
        })
    } else if (props.mode === "U") {
      data.TAG_CD = props.tagInfo.TAG_CD
      dispatch(action._setPreStatus(true))
      ImageAnnoApi._updateDataTag(data)
        .then(result => {
          if (result.status === 1) {
            toast.info(<CommonToast Icon={MdSave} text={"Tag Update Success"} />)
            setIsSave(false)
            props._successModal(props.updateIndex)
            dispatch(action._setPreStatus(false))
          } else {
            throw { err: "status 0" }
          }
        })
        .catch(e => {
          console.log(e)
          setIsSave(false)
          dispatch(action._setPreStatus(false))
          toast.error(<CommonToast Icon={MdError} text={"Tag Update Fail"} />)
        })
    }
  }

  const _changeCategory1 = value => {
    if (value === "") {
      setValue("CATEGORY1", "")
      setValue("CATEGORY2", "")
      setValue("CLASS_CD", "")
      setCategory2([])
      setCategory3([])
    } else {
      ImageAnnoApi._getCategory({ CATEGORY1: value, DATA_TYPE: props.dataType, OBJECT_TYPE: props.objectType })
        .then(category2 => {
          setCategory2(category2)
          setValue("CATEGORY2", category2[0]?.value)

          return ImageAnnoApi._getCategory({
            CATEGORY1: getValues("CATEGORY1"),
            CATEGORY2: category2[0].value,
            DATA_TYPE: props.dataType,
            OBJECT_TYPE: props.objectType
          })
        })
        .then(category3 => {
          setCategory3(category3)
          setValue("CLASS_CD", category3[0]?.value)
        })
        .catch(e => console.log(e))
    }
  }

  const _changeCategory2 = value => {
    setValue("CLASS_CD", "")
    ImageAnnoApi._getCategory({
      CATEGORY1: getValues("CATEGORY1"),
      CATEGORY2: value,
      DATA_TYPE: props.dataType,
      OBJECT_TYPE: props.objectType
    })
      .then(data => {
        setCategory3(data)
        setValue("CLASS_CD", data[0]?.value)
      })
      .catch(e => console.log(e))
  }

  return (
    <Modal
      isOpen={props.modal}
      toggle={props.toggle}
      modalClassName={"ltr-support"}
      className={"modal-dialog--primary modal-dialog--header"}
      style={{ marginTop: "10rem" }}
      size={"sm"}
    >
      <ModalHeader toggle={props.toggle}>
        <RiPriceTag3Line /> Tag
      </ModalHeader>
      <ModalBody>
        <Container>
          <Row>
            <Col md={12} xl={12}>
              <div className="form form--horizontal">
                <span className="form__form-group-label mt-1">Tag Title</span>
                <Input
                  type="text"
                  name="NAME"
                  className="mt-1"
                  innerRef={register({
                    required: true,
                    validate: {
                      validateTrim: value => String(value).trim().length !== 0
                    }
                  })}
                  onKeyDown={e => {
                    e.stopPropagation()
                  }}
                  style={{ color: "black", backgroundColor: "white" }}
                />
                {errors.NAME?.type === "validateTrim" && <div className="form__form-group-label form-error mt-1">Name is Required</div>}
                {errors.NAME?.type === "required" && <div className="form__form-group-label form-error mt-1">Name is Required</div>}
                <span className="form__form-group-label mt-1">Description</span>
                <Input
                  type="textarea"
                  name="DESC_TXT"
                  className="mt-1"
                  innerRef={register}
                  onKeyDown={e => {
                    e.stopPropagation()
                  }}
                  style={{ color: "black", backgroundColor: "white", resize: "none" }}
                />
                <span className="form__form-group-label mt-1">
                  Tag Color
                  <BsFillCircleFill className="ml-1" style={{ color: color, fontSize: "15px" }} />
                </span>
                <HuePicker className="mt-1" width={"100%"} color={color} onChange={changeColor} />
                <span className="form__form-group-label mt-1">Category</span>
                <Controller
                  className="mt-1"
                  as={CommonSelect}
                  name="CATEGORY1"
                  valueName="selected"
                  control={control}
                  options={category1}
                  onChange={([selected]) => {
                    _changeCategory1(selected)
                    return selected
                  }}
                  isMulti={false}
                  isDefault={false}
                  menuPortalTarget={false}
                  isClearable={true}
                />
                <span className="form__form-group-label mt-1">Sub Category</span>
                <Controller
                  className="mt-1"
                  as={CommonSelect}
                  name="CATEGORY2"
                  valueName="selected"
                  control={control}
                  options={category2}
                  onChange={([selected]) => {
                    _changeCategory2(selected)
                    return selected
                  }}
                  isMulti={false}
                  isDefault={false}
                  menuPortalTarget={false}
                  isClearable={true}
                />
                <span className="form__form-group-label mt-1">Class</span>
                <Controller
                  className="mt-1"
                  as={CommonSelect}
                  name="CLASS_CD"
                  valueName="selected"
                  control={control}
                  options={category3}
                  onChange={([selected]) => selected}
                  isMulti={false}
                  isDefault={false}
                  menuPortalTarget={false}
                  isClearable={true}
                />
              </div>
            </Col>
            <Col xl={12}>
              <div className="mt-3 mb-2 float-right">
                <CommonButton
                  ButtonIcon={FaSave}
                  className="bg-green z-0"
                  // text={props.mode === "U" ? "수정" : "저장"}
                  text={"Save"}
                  onClick={() => {
                    handleSubmit(onSubmit)()
                  }}
                />
              </div>
            </Col>
          </Row>
        </Container>
      </ModalBody>
    </Modal>
  )
}

TagModal.propTypes = {
  mode: PropTypes.string,
  tagInfo: PropTypes.object,
  dataSetCd: PropTypes.string,
  _successModal: PropTypes.func
}

export default TagModal
