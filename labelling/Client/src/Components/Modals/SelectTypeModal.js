import React, { useState } from "react"
import { Button, Modal, ModalHeader, ModalBody, ModalFooter } from "reactstrap"
import { FcEditImage, FcVideoCall, FcDataSheet } from "react-icons/fc"

const getIcon = type => {
  switch (type) {
    case "IMAGE":
      return <FcEditImage />
    case "VIDEO":
      return <FcVideoCall />
    case "TEXT":
      return <FcDataSheet />
    default:
      return null
  }
}

const LinkComp = ({ type, desc, onClick, width }) => {
  return (
    <div className="data-type-modal-card" style={{ width: width }} onClick={onClick}>
      <div className="datatype">
        <div className="image">{getIcon(type)}</div>
        <div className="text">
          <div className="main">{type}</div>
          <div className="sub">{desc}</div>
        </div>
      </div>
    </div>
  )
}

export const DataTypeModal = ({ modal, toggle, dataTypeList }) => {
  return (
    <Modal isOpen={modal} toggle={toggle} size={"lg"} style={{ marginTop: "10rem" }}>
      <ModalHeader toggle={toggle}>Select Data Type</ModalHeader>
      <ModalBody>
        {dataTypeList.map((el, i) => (
          <LinkComp key={i} type={el.type} desc={el.desc} onClick={el.onClick} width={el.width} />
        ))}
      </ModalBody>
    </Modal>
  )
}
