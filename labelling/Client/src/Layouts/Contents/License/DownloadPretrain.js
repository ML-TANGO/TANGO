import React, { useEffect, useState } from "react"
import { Progress } from "reactstrap"
import { toast } from "react-toastify"
import { MdInfo, MdError } from "react-icons/md"
import { useHistory } from "react-router-dom"
import BarLoader from "react-spinners/BarLoader"
import * as CommonApi from "./../../../Config/Services/CommonApi"
import CommonToast from "../../../Components/Common/CommonToast"

function DownloadPretrain() {
  const [successCount, setSuccessCount] = useState(0)
  const [errMsg, setErrMsg] = useState({})
  const history = useHistory()

  useEffect(() => {
    download()
  }, [])

  const download = async () => {
    try {
      const redirectUrl = history.location?.state?.redirectUrl ? history.location?.state?.redirectUrl : "/"
      let param = { MDL_KIND: "detection" }
      let dResult = await CommonApi._getPretrainedModel(param)
      if (dResult?.status === 1) setSuccessCount(successCount => successCount + 1)
      else throw dResult

      param.MDL_KIND = "segmentation"
      let sResult = await CommonApi._getPretrainedModel(param)
      if (sResult?.status === 1) setSuccessCount(successCount => successCount + 1)
      else throw sResult
      param.MDL_KIND = "classification"
      let cResult = await CommonApi._getPretrainedModel(param)
      if (cResult?.status === 1) setSuccessCount(successCount => successCount + 1)
      else throw cResult

      toast.info(<CommonToast Icon={MdInfo} text={"Download is complete"} />, {
        onClose: () => history.push({ pathname: redirectUrl, state: { result: { status: 1, msg: "Success" } } })
      })
    } catch (e) {
      setErrMsg(e)
      toast.error(<CommonToast Icon={MdError} text={"Failed to download Pretrain Model"} />)
      history.push({ pathname: "/" })
      console.log(e)
    }
  }

  const handleClick = async () => {
    setSuccessCount(0)
    setErrMsg({})
    download()
  }

  return (
    <div className="theme-light ltr-support without-y-scrollbar">
      <div className="account" style={{ background: "radial-gradient( #364261, #2e2e2e, #000)" }}>
        <div className="account__wrapper" style={{ width: "500px" }}>
          <div className="account__card box_shadow">
            <div className="license-header">
              <h3>DownLoad Pretrain Model</h3>
              <h4 className="subhead">Download the default Ai Model</h4>
            </div>
            <Progress barClassName="progress-color" value={successCount} max={3} />
            {Object.keys(errMsg).length === 0 && <BarLoader width="100%" color="#4b5b85" />}
            <div className="text-center mt-1">{successCount} / 3</div>
            {Object.keys(errMsg).length !== 0 && (
              <>
                <div className="text-center mt-1" style={{ color: "red" }}>
                  {errMsg?.msg}{" "}
                </div>
                <button type="button" className={`form__form-group-button download_retry mt-1`} onClick={handleClick}>
                  RETRY
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default DownloadPretrain
