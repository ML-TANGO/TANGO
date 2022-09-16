import React from "react"
import { Row, Col, Modal, ModalHeader, ModalBody, ModalFooter, Input } from "reactstrap"
import { FaColumns } from "react-icons/fa"
import { cloneDeep } from "lodash-es"

import VirtualTable from "../Common/VirtualTable"
import CommonButton from "../Common/CommonButton"

function ColumnModal(props) {
  const { modal, toggle, fileState, setFileState, tableState, setTableState } = props

  const columns = [
    {
      label: "#",
      width: 200,
      className: "text-center",
      disableSort: true,
      dataKey: "index",
      cellRenderer: ({ rowIndex }) => rowIndex + 1
    },
    {
      label: "Column",
      dataKey: "COLUMN_NM",
      className: "text-center",
      disableSort: true,
      width: 250
    },
    {
      label: "Defalut Value",
      dataKey: "DEFAULT_VALUE",
      className: "text-center",
      disableSort: true,
      width: 200,
      cellRenderer: ({ rowIndex, cellData }) => {
        return (
          <div className="form">
            <Input
              type="text"
              value={cellData}
              className="mt-1"
              style={{ color: "black", backgroundColor: "white" }}
              onChange={handleChange(rowIndex)}
            />
          </div>
        )
      }
    }
  ]

  const handleChange = idx => e => {
    const cColList = cloneDeep(fileState !== undefined ? fileState.colList : tableState.colList)
    cColList[idx].DEFAULT_VALUE = e.target.value
    fileState !== undefined
      ? setFileState(prevState => ({ ...prevState, colList: cColList }))
      : setTableState(prevState => ({ ...prevState, colList: cColList }))
  }

  return (
    <Modal
      isOpen={modal}
      toggle={toggle}
      className={"modal-dialog--primary modal-dialog--header"}
      style={{ marginTop: "10rem" }}
      size={"lg"}
    >
      <ModalHeader toggle={toggle}>
        <FaColumns /> Column List
      </ModalHeader>
      <ModalBody>
        <Row>
          <Col xl={12}>
            <VirtualTable
              className="vt-table"
              rowClassName="vt-header"
              height={`270px`}
              headerHeight={40}
              rowHeight={50}
              columns={columns}
              data={fileState !== undefined ? fileState.colList : tableState.colList}
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

ColumnModal.propTypes = {}

export default ColumnModal
