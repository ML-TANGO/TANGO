import React, { useState } from "react"
import { Input, Modal, ModalHeader, ModalBody, ModalFooter } from "reactstrap"
import { RiExchangeLine } from "react-icons/ri"
import CommonButton from "../Common/CommonButton"

export const LabelModal = props => {
  const [newTagName, setNewTagName] = useState("")

  const changeLabel = () => {
    props.changeLabel(newTagName)
    setNewTagName("New tag")
  }

  return (
    <Modal
      isOpen={props.modal}
      toggle={props.toggle}
      className={"modal-dialog--primary modal-dialog--header"}
      style={{ marginTop: "10rem" }}
      size={"sm"}
    >
      <ModalHeader toggle={props.toggle}>
        <RiExchangeLine /> Change label name
      </ModalHeader>
      <ModalBody>
        <div style={{ textAlign: "left" }}>선택된 레이블 명을 새로운 레이블 명으로 바꿀 수 있습니다. </div>
        <div style={{ textAlign: "left" }}>
          아래에 <span style={{ color: "#08f" }}>새로운 레이블 명</span>을 입력해주세요.
        </div>
        <div style={{ textAlign: "left", marginBottom: "20px" }}>
          You can rename the selected label name to a new label name. Please enter <span style={{ color: "#08f" }}>a new label name </span>
          below.
        </div>
        <div>
          <Input
            type="text"
            onChange={e => setNewTagName(e.target.value)}
            placeholder="New Label"
            value={newTagName}
            onKeyDown={e => {
              if (e.keyCode === 13) changeLabel()
            }}
          />
        </div>
      </ModalBody>
      <ModalFooter>
        <CommonButton className="bg-green" text="Apply" onClick={changeLabel} />
        <CommonButton className="bg-red" text="Cancel" onClick={props.toggle} />
      </ModalFooter>
    </Modal>
  )
}
