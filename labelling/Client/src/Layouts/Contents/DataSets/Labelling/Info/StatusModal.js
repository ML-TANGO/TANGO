import React, { useState, useEffect, useMemo } from "react"
import PropTypes from "prop-types"
import { Container, Row, Col, Modal, ModalHeader, ModalBody } from "reactstrap"
import { GoDashboard } from "react-icons/go"
import { FaRegStopCircle } from "react-icons/fa"
import { toast } from "react-toastify"
import { MdDeleteForever, MdError } from "react-icons/md"

import * as AiModelApi from "Config/Services/AimodelApi"

import VirtualTable from "../../../../../Components/Common/VirtualTable"
import CommonToast from "../../../../../Components/Common/CommonToast"

function StatusModal(props) {
  const [hoverIndex, setHoverIndex] = useState(null)
  const [fileList, setFileList] = useState([])

  useEffect(() => {
    let arr = []
    props.activeModel.map(model => {
      let obj = {}
      let modelTag = []
      props?.tagList?.map(tag => {
        if (tag.CLASS_CD !== null) {
          model?.USEABLE_CLASS.some(cls => {
            if (cls.CLASS_CD === tag.CLASS_CD) {
              modelTag.push(tag.NAME)
              return true
            }
          })
        }
      })
      obj.modelTag = modelTag
      obj.modelName = model.AI_CD
      obj.aiType = model.AI_TYPE
      obj.pid = model.PID
      obj.port = model.PORT
      obj.epoch = model.EPOCH
      arr.push(obj)
    })
    setFileList(arr)
  }, [props.activeModel])

  const _onRowMouseOver = ({ index }) => {
    setHoverIndex(index)
  }

  const _onRowMouseOut = () => {
    setHoverIndex(null)
  }

  const _onRowClick = () => {}

  const _rowStyle = ({ index }) => {
    if (hoverIndex === index) {
      return { backgroundColor: "#2f2f2f" }
    }
    return
  }

  const _stopModel = rowData => () => {
    // stop
    const param = { PID: rowData.pid }
    AiModelApi._stopActiveModel(param)
      .then(result => {
        if (result.status === 1) {
          props.getActiveStatus()
          toast.info(<CommonToast Icon={MdDeleteForever} text={"Model Delete Success"} />)
        } else toast.error(<CommonToast Icon={MdError} text={"Model Delete Fail"} />)
      })
      .catch(e => {
        props.getActiveStatus()
        toast.error(<CommonToast Icon={MdError} text={"Model Delete Fail"} />)
        console.log(e)
      })
  }

  const columns = useMemo(
    () => [
      {
        label: "Model",
        width: 150,
        className: "text-center",
        disableSort: true,
        dataKey: "modelName"
      },
      {
        label: "Epoch",
        width: 100,
        className: "text-center",
        disableSort: true,
        dataKey: "epoch"
      },
      {
        label: "Tag",
        width: 250,
        className: "text-center",
        disableSort: true,
        dataKey: "modelTag",
        cellRenderer: ({ cellData }) => cellData.join(" , ")
      },
      {
        label: "Stop",
        width: 100,
        className: "text-center",
        disableSort: true,
        dataKey: "",
        cellRenderer: ({ rowData }) => {
          return <FaRegStopCircle className="icon-pointer hover-red" style={{ fontSize: "13px" }} onClick={_stopModel(rowData)} />
        }
      }
    ],
    [fileList]
  )

  return (
    <Modal
      isOpen={props.modal}
      toggle={props.toggle}
      modalClassName={"ltr-support"}
      className={"modal-dialog--primary modal-dialog--header"}
      style={{ marginTop: "10rem" }}
      size={"md"}
    >
      <ModalHeader toggle={props.toggle}>
        <GoDashboard className="mb-1" style={{ color: "#379a37" }} /> Active Mini Predictor
      </ModalHeader>
      <ModalBody>
        <Container>
          <Row>
            <Col md={12} xl={12}>
              <VirtualTable
                className="vt-table"
                rowClassName="vt-header"
                height="400px"
                headerHeight={40}
                rowHeight={50}
                columns={columns}
                data={fileList}
                onRowMouseOver={_onRowMouseOver}
                onRowMouseOut={_onRowMouseOut}
                onRowClick={_onRowClick}
                rowStyle={_rowStyle}
                isNoRowRender={false}
              />
            </Col>
          </Row>
        </Container>
      </ModalBody>
    </Modal>
  )
}

StatusModal.propTypes = {
  toggle: PropTypes.func,
  modal: PropTypes.bool,
  activeModel: PropTypes.array,
  tagList: PropTypes.array,
  getActiveStatus: PropTypes.func
}

export default StatusModal
