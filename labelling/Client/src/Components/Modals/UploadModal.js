import React, { useEffect, useState } from "react"
import { Row, Col, Modal, ModalHeader, ModalBody, ModalFooter } from "reactstrap"
import { toast } from "react-toastify"
import { FaUpload } from "react-icons/fa"
import { FormDropZone } from "../Form/FormComponent"
import { MdError } from "react-icons/md"

import CommonButton from "../Common/CommonButton"
import useEnterpriseDivision from "../Utils/useEnterpriseDivision"
import { bytesToSize } from "../Utils/Utils"
import CommonToast from "../Common/CommonToast"

function UploadModal(props) {
  const { modal, toggle, control, pageState, typeState, fileState, setFileState, onChange, accept, multiple, maxFiles } = props
  const [fileList, setFileList] = useState([])

  const dataSetUpload = useEnterpriseDivision(process.env.BUILD, "dataSet", "dataSetUpload")

  useEffect(() => {
    if (fileList.length !== 0) {
      handleUpload()
    }
  }, [fileList])

  const handleUpload = () => {
    let flist = fileState.fileList

    if (typeState.dataType === "V" && dataSetUpload[typeState.dataType].SIZE !== Infinity) {
      const flag = fileList.some(ele => ele.size > dataSetUpload[typeState.dataType].SIZE)
      if (flag) {
        toast.warn(
          <CommonToast
            Icon={MdError}
            text={`The maximum uploads Size in the CE version is ${bytesToSize(dataSetUpload[typeState.dataType].SIZE)}`}
          />
        )
        return flist
      }
    }

    if (fileList.length + flist.length > dataSetUpload[typeState.dataType].COUNT) {
      toast.warn(
        <CommonToast
          Icon={MdError}
          text={`The maximum number of uploads in the CE version is ${dataSetUpload[typeState.dataType].COUNT}`}
        />
      )
      return
    }

    let obj = {}
    for (let i = 0; i < flist.length; i++) {
      obj[flist[i].path] = flist[i].path
    }

    let cSelected = []
    let count = 0
    for (let i = 0; i < fileList.length; i++) {
      if (obj[fileList[i].path]) {
        if (count < 5) {
          toast.error(<CommonToast Icon={MdError} text={`Cannot upload same file name. \n${fileList[i].name}`} />, {
            autoClose: 4000
          })
        }
        count++
      } else {
        cSelected.push(fileList[i])
      }
    }
    if (count > 5) {
      toast.error(<CommonToast Icon={MdError} text={`Cannot upload same file name. \n more than ${count - 5} files`} />, {
        autoClose: 4000
      })
    }

    setFileState(prevState => ({ ...prevState, fileList: [...prevState.fileList, ...cSelected] }))
    setFileList([])

    toggle()
  }

  const getAccept = () => {
    if (typeState.dataType === "I") {
      if (typeState.importType !== "N") {
        return "image/jpeg, image/png, application/JSON, .json"
      } else {
        return "image/jpeg, image/png"
      }
    } else if (typeState.dataType === "V") {
      return "video/mp4, video/webm, video/ogg"
    } else {
      return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet, text/plain, text/csv, application/vnd.ms-excel"
    }
  }

  const accepts = getAccept()

  return (
    <Modal
      isOpen={modal}
      toggle={toggle}
      className={"modal-dialog--primary modal-dialog--header"}
      style={{ marginTop: "10rem" }}
      size={"lg"}
    >
      <ModalHeader toggle={toggle}>
        <FaUpload /> File Upload
      </ModalHeader>
      <ModalBody>
        <Row>
          <Col xl={12}>
            <FormDropZone
              control={control}
              pageState={pageState}
              typeState={typeState}
              fileList={fileList}
              setFileList={setFileList}
              accept={accepts}
              multiple={multiple}
              onChange={onChange}
              maxFiles={maxFiles}
            />
          </Col>
        </Row>
      </ModalBody>
      <ModalFooter>
        <CommonButton className="bg-red" text="Cancel" onClick={toggle} />
      </ModalFooter>
    </Modal>
  )
}

UploadModal.propTypes = {}

export default UploadModal
