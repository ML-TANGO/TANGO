import React, { useState, useEffect, useMemo, useCallback, useRef } from "react"
import { Row } from "reactstrap"
import { useDispatch, useSelector } from "react-redux"

import Toolbar from "./Toolbar/Toolbar"
import Label from "./Label/Label"
import Info from "./Info/Info"

import * as ImageLabelActions from "../../../../Redux/Actions/ImageLabelActions"
import * as VideoLabelActions from "../../../../Redux/Actions/VideoLabelActions"
import * as ImageAnnoApi from "Config/Services/ImageAnnoApi"
import * as VideoAnnoApi from "Config/Services/VideoAnnoApi"

import useEnterpriseDivision from "../../../../Components/Utils/useEnterpriseDivision"

function LabelLayout(props) {
  const { history } = props
  const [dataSet, setDataSet] = useState({})
  const dispatch = useDispatch()

  const labelTracker = useEnterpriseDivision(process.env.BUILD, "dataSet", "labelTracker")

  const imageLabel = useSelector(
    state => state.imageLabel,
    (prevItem, item) => {
      if (item === prevItem) return true
      else return false
    }
  )
  const videoLabel = useSelector(
    state => state.videoLabel,
    (prevItem, item) => {
      if (item === prevItem) return true
      else return false
    }
  )

  if (history.location && history.location.state) {
    const state = { ...history.location.state }
    if (state.dataInfo) {
      setDataSet(state.dataInfo)
      delete state.dataInfo
      history.replace({ ...history.location, state })
    }
  }

  const label = useMemo(() => (dataSet.DATA_TYPE === "I" ? imageLabel : videoLabel), [dataSet.DATA_TYPE, imageLabel, videoLabel])
  const action = useMemo(() => (dataSet.DATA_TYPE === "I" ? ImageLabelActions : VideoLabelActions), [dataSet.DATA_TYPE])

  const labelRef = useRef(null)

  useEffect(() => {
    labelRef.current = label
  }, [label])

  const _checkSave = () => {
    if (labelRef.current.isDrawAction) {
      if (dataSet.DATA_TYPE === "I") {
        if (dataSet.OBJECT_TYPE === "D") {
          const param = {
            DATASET_CD: labelRef.current.curImage.DATASET_CD,
            DATA_CD: labelRef.current.curImage.DATA_CD,
            FILE_PATH: labelRef.current.curImage.FILE_PATH,
            ANNO_DATA: { POLYGON_DATA: labelRef.current.objectList, BRUSH_DATA: labelRef.current.brushList },
            OBJECT_TYPE: dataSet.OBJECT_TYPE
          }
          ImageAnnoApi._setImageAnnotation(param)
            .then(() => {
              // if (result.status === 1) {
              //   toast.info(<CommonToast Icon={MdSave} text={"Save Success"} />)
              // } else {
              //   throw { err: "status 0" }
              // }
            })
            .catch(e => {
              // toast.error(<CommonToast Icon={MdError} text={"Save Fail"} />)
              console.log(e)
            })
        } else {
          // 여기는 안 됨.... ㅜㅜ
          dispatch(ImageLabelActions._saveImage(true))
        }
      } else {
        const param = {
          DATASET_CD: labelRef.current.curVideo.DATASET_CD,
          DATA_CD: labelRef.current.curVideo.DATA_CD,
          ANNO_DATA: { POLYGON_DATA: labelRef.current.objectList, BRUSH_DATA: labelRef.current.brushList },
          OBJECT_TYPE: dataSet.OBJECT_TYPE
        }
        VideoAnnoApi._setVideoAnnotation(param)
          .then(() => {
            // if (result.status === 1) {
            //   toast.info(<CommonToast Icon={MdSave} text={"Save Success"} />)
            // } else {
            //   throw { err: "status 0" }
            // }
          })
          .catch(e => {
            // toast.error(<CommonToast Icon={MdError} text={"Save Fail"} />)
            console.log(e)
          })
      }
      dispatch(ImageLabelActions._isDrawAction(false))
    }
  }

  useEffect(() => {
    return () => {
      _checkSave()
      dispatch(ImageLabelActions._initLabel())
      dispatch(VideoLabelActions._initVideoLabel())
    }
  }, [])

  const _handleKeyDown = useCallback(
    e => {
      if (label.modalCheck) {
        return
      }

      switch (e.keyCode) {
        // case 49:
        //   label.btnSts === "isMove" ? dispatch(action._setBtnSts("none")) : dispatch(action._setBtnSts("isMove"))
        //   break
        // case 50:
        //   label.btnSts === "isEdit" ? dispatch(action._setBtnSts("none")) : dispatch(action._setBtnSts("isEdit"))
        //   break
        case 49:
          if (dataSet.OBJECT_TYPE === "D") {
            label.btnSts === "isRect" ? dispatch(action._setBtnSts("none")) : dispatch(action._setBtnSts("isRect"))
          } else if (dataSet.OBJECT_TYPE === "S") {
            label.btnSts === "isPolygon" ? dispatch(action._setBtnSts("none")) : dispatch(action._setBtnSts("isPolygon"))
          }
          break
        case 50:
          if (dataSet.OBJECT_TYPE === "S") {
            label.btnSts === "isMagic" ? dispatch(action._setBtnSts("none")) : dispatch(action._setBtnSts("isMagic"))
          }
          if (dataSet.OBJECT_TYPE === "D" && dataSet.DATA_TYPE === "V" && labelTracker) {
            label.btnSts === "isTracker" ? dispatch(action._setBtnSts("none")) : dispatch(action._setBtnSts("isTracker"))
          }
          break
        case 51:
          if (dataSet.OBJECT_TYPE === "S" && dataSet.DATA_TYPE === "I") {
            label.btnSts === "isBrush" ? dispatch(action._setBtnSts("none")) : dispatch(action._setBtnSts("isBrush"))
          }
          if (dataSet.OBJECT_TYPE === "S" && dataSet.DATA_TYPE === "V" && labelTracker) {
            label.btnSts === "isTracker" ? dispatch(action._setBtnSts("none")) : dispatch(action._setBtnSts("isTracker"))
          }
          break
        case 52:
          if (dataSet.OBJECT_TYPE === "S" && dataSet.DATA_TYPE === "I") {
            label.btnSts === "isEraser" ? dispatch(action._setBtnSts("none")) : dispatch(action._setBtnSts("isEraser"))
          }
          break
        default:
          break
      }
    },
    [label.btnSts, label.modalCheck, action, dataSet, dispatch]
  )

  useEffect(() => {
    window.addEventListener("keydown", _handleKeyDown)
    return () => {
      window.removeEventListener("keydown", _handleKeyDown)
    }
  }, [_handleKeyDown])

  return (
    <Row noGutters style={{ height: "100vh", position: "relative" }}>
      <Toolbar dataType={dataSet.DATA_TYPE} objectType={dataSet.OBJECT_TYPE} />
      <Label dataType={dataSet.DATA_TYPE} dataSet={dataSet} keymode={label.btnSts} />
      <Info dataSetCd={dataSet.DATASET_CD} objectType={dataSet.OBJECT_TYPE} dataType={dataSet.DATA_TYPE} />
    </Row>
  )
}

export default LabelLayout
