import React, { useState, useCallback } from "react"
import PropTypes from "prop-types"
import { useDispatch, useSelector } from "react-redux"
import { toast } from "react-toastify"
import { MdSave, MdError } from "react-icons/md"

import ImageList from "./ImageList"
import VideoList from "./VideoList"
import TagList from "./TagList"

import * as ImageLabelActions from "Redux/Actions/ImageLabelActions"
import * as VideoLabelActions from "Redux/Actions/VideoLabelActions"
import * as ImageAnnoApi from "Config/Services/ImageAnnoApi"
import * as VideoAnnoApi from "Config/Services/VideoAnnoApi"

import { FaSave } from "react-icons/fa"
import CommonButton from "../../../../../Components/Common/CommonButton"
import CommonToast from "../../../../../Components/Common/CommonToast"

function Info(props) {
  const [renderLazy, setRenderLazy] = useState(false)
  const [saveDataCd, setSaveDataCd] = useState(null)

  const dispatch = useDispatch()
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

  const _handleImageClick = useCallback(() => {
    if (props.objectType === "D") {
      const param = {
        DATASET_CD: imageLabel.curImage.DATASET_CD,
        DATA_CD: imageLabel.curImage.DATA_CD,
        FILE_PATH: imageLabel.curImage.FILE_PATH,
        ANNO_DATA: { POLYGON_DATA: imageLabel.objectList, BRUSH_DATA: imageLabel.brushList },
        OBJECT_TYPE: props.objectType
      }
      ImageAnnoApi._setImageAnnotation(param)
        .then(result => {
          if (result.status === 1) {
            setSaveDataCd(imageLabel.curImage.DATA_CD)
            toast.info(<CommonToast Icon={MdSave} text={"Save Success"} />)
          } else {
            throw { err: "status 0" }
          }
        })
        .catch(e => {
          console.log(e)
          toast.error(<CommonToast Icon={MdError} text={"Save Fail"} />)
        })
    } else if (props.objectType === "S") {
      dispatch(ImageLabelActions._saveImage(true))
    }
    dispatch(ImageLabelActions._isDrawAction(false))
  }, [props.objectType, imageLabel.curImage, imageLabel.objectList, imageLabel.brushList])

  const _handleVideoClick = useCallback(() => {
    const param = {
      DATASET_CD: videoLabel.curVideo.DATASET_CD,
      DATA_CD: videoLabel.curVideo.DATA_CD,
      ANNO_DATA: { POLYGON_DATA: videoLabel.objectList, BRUSH_DATA: videoLabel.brushList },
      OBJECT_TYPE: props.objectType
    }
    VideoAnnoApi._setVideoAnnotation(param)
      .then(result => {
        if (result.status === 1) {
          setSaveDataCd(videoLabel.curVideo.DATA_CD)
          toast.info(<CommonToast Icon={MdSave} text={"Save Success"} />)
        } else {
          throw { err: "status 0" }
        }
      })
      .catch(e => {
        console.log(e)
        toast.error(<CommonToast Icon={MdError} text={"Save Fail"} />)
      })
    dispatch(VideoLabelActions._isDrawAction(false))
  }, [videoLabel.curVideo, videoLabel.objectList, videoLabel.brushList, props.objectType])

  const _setRenderLazy = useCallback(bool => {
    setRenderLazy(bool)
  }, [])

  return (
    <div className="info_wrap h-100 p-2">
      <TagList dataSetCd={props.dataSetCd} objectType={props.objectType} dataType={props.dataType} _setRenderLazy={_setRenderLazy} />
      {props.dataType === "I" && (
        <ImageList
          dataSetCd={props.dataSetCd}
          objectType={props.objectType}
          renderLazy={renderLazy}
          setSaveDataCd={setSaveDataCd}
          saveDataCd={saveDataCd}
        />
      )}
      {props.dataType === "V" && (
        <VideoList
          dataSetCd={props.dataSetCd}
          objectType={props.objectType}
          renderLazy={renderLazy}
          setSaveDataCd={setSaveDataCd}
          saveDataCd={saveDataCd}
        />
      )}
      <CommonButton
        className="bg-green float-right mt-2 mr-0"
        ButtonIcon={FaSave}
        text="Save"
        onClick={props.dataType === "I" ? _handleImageClick : _handleVideoClick}
      />
    </div>
  )
}

Info.propTypes = {
  dataSetCd: PropTypes.string,
  objectType: PropTypes.string
}

export default React.memo(Info)
