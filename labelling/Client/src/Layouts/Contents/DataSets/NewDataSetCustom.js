import React, { useRef } from "react"
import { toast } from "react-toastify"

// icons
import { MdSave, MdError } from "react-icons/md"

// api
import * as DataSetApi from "Config/Services/DataSetApi"

// pre-defined components
import CommonButton from "Components/Common/CommonButton"
import CommonToast from "Components/Common/CommonToast"

export const CustomConfirmAlert = ({ title, data, param, setPageState, onClose, initDataSet }) => {
  const confirmTextRef = useRef("")
  return (
    <div className="react-confirm-alert-custom">
      <h1>
        <MdError />
        DataSet Edit
      </h1>
      <div className="custom-modal-body">
        <div className="text-warning">Warning. This action is irreversible.</div>
        <div className="explain">Auto Labeling option is Yes</div>
        <div className="explain">Saved labeling information is deleted</div>
        <div className="explain">
          Please type <strong>[ {title} ]</strong> to avoid unexpected action.
        </div>
        <input
          type="text"
          className="react-confirm-alert-input"
          onChange={e => {
            confirmTextRef.current = e.target.value
          }}
        />
      </div>
      <div className="custom-buttons">
        <CommonButton
          className="bg-green"
          text="Apply"
          onClick={() => {
            if (confirmTextRef.current.trim() === data.TITLE.trim()) {
              DataSetApi._setUpdateDataset(param)
                .then(result => {
                  if (result.status === 1) {
                    toast.info(<CommonToast Icon={MdSave} text={"DataSet Update Success"} />)
                    setPageState(prevState => ({ ...prevState, isSave: false, isRedirect: true }))
                    initDataSet()
                  } else {
                    throw { err: "status 0" }
                  }
                })
                .catch(err => {
                  console.log(err)
                  setPageState(prevState => ({ ...prevState, isSave: false }))
                  toast.error(<CommonToast Icon={MdError} text={"DataSet Update Fail"} />)
                })
              onClose()
            } else alert("Not matched.")
          }}
        />
        <CommonButton
          className="bg-red"
          text="Cancel"
          onClick={() => {
            setPageState(prevState => ({ ...prevState, isSave: false }))
            onClose()
          }}
        />
      </div>
    </div>
  )
}

export const CustomColumnAlert = ({ onClose, onCancel, setFileState, colList, diffCols }) => {
  return (
    <div className="react-confirm-alert-custom">
      <h1>
        <MdError />
        Mismatch column shape
      </h1>
      <div className="custom-modal-body">
        <div className="text-warning">Warning. This action is irreversible.</div>
        <div className="explain">{diffCols}</div>
      </div>
      <div className="custom-buttons">
        <CommonButton
          className="bg-green"
          text="Apply"
          onClick={() => {
            setFileState(prevState => ({ ...prevState, colList: colList }))
            onClose()
          }}
        />
        <CommonButton
          className="bg-red"
          text="Cancel"
          onClick={() => {
            onCancel()
            onClose()
          }}
        />
      </div>
    </div>
  )
}
