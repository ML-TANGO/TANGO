import React, { useEffect, useState } from "react"
import { Row, Col, Modal, ModalHeader, ModalBody, ModalFooter } from "reactstrap"
import { FaUpload } from "react-icons/fa"
import { FormDropZone } from "../Form/FormComponent"
import CommonButton from "../Common/CommonButton"

function UploadModal(props) {
  const { modal, toggle, control, pageState, setPageState, typeState, setFileState, onChange, multiple, maxFiles } = props
  const [fileList, setFileList] = useState([])

  useEffect(() => {
    if (fileList.length !== 0) {
      handleUpload()
    }
  }, [fileList])

  const handleUpload = () => {
    setFileState(prevState => ({ ...prevState, fileList: [...prevState.fileList, ...fileList] }))
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
      backdrop={"static"}
      onDrop={() => {
        setPageState(prevState => ({ ...prevState, isUpload: true }))
      }}
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
