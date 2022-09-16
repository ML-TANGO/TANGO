import React, { useEffect, useState, useCallback, useMemo } from "react"
import { useSelector, useDispatch } from "react-redux"
import { IoIosArrowBack, IoIosArrowForward } from "react-icons/io"
import { BsFillCircleFill } from "react-icons/bs"
import { RiPriceTag3Line, RiBrush2Line, RiDeleteBin4Line, RiPushpin2Line, RiPushpinLine, RiPictureInPictureExitFill } from "react-icons/ri"
import { MdOpacity } from "react-icons/md"
import InputNumber from "rc-input-number"
import { cloneDeep } from "lodash-es"

import Slider from "../../../../../Components/Common/Slider"
import * as ImageLabelActions from "../../../../../Redux/Actions/ImageLabelActions"
import * as VideoLabelActions from "../../../../../Redux/Actions/VideoLabelActions"

function LabelTopBar(props) {
  const [fileName, setFileName] = useState()
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

  // DataType Image, Video 구분
  const label = useMemo(() => (props.dataType === "I" ? imageLabel : videoLabel), [props.dataType, imageLabel, videoLabel])
  const action = useMemo(() => (props.dataType === "I" ? ImageLabelActions : VideoLabelActions), [props.dataType])

  useEffect(() => {
    if (props.dataType === "V") {
      if (videoLabel.curVideo.length !== 0) {
        const videoInfo = videoLabel.curVideo
        setFileName(`No.${videoLabel.curIndex + 1} - ${videoInfo.FILE_NAME}${videoInfo.FILE_EXT}`)
      }
    } else {
      if (Object.keys(imageLabel.curImage).length !== 0) {
        const imageInfo = imageLabel.curImage
        setFileName(`No.${imageLabel.curIndex + 1} - ${imageInfo.FILE_NAME}${imageInfo.FILE_EXT}`)
      }
    }
  }, [imageLabel.curImage, videoLabel.curVideo])

  const _handleZoomFixClick = () => {
    dispatch(ImageLabelActions._isZoomFix(!imageLabel.isZoomFix))
  }

  const _handleReturnImgSize = () => {
    dispatch(ImageLabelActions._isResetImageSize(!imageLabel.isResetImageSize))
  }

  const _handleKeyDown = useCallback(
    e => {
      if (label.modalCheck) {
        return
      }
      switch (e.keyCode) {
        case 65:
          if (props.dataType === "V") {
            if (!label.videoStatus) {
              dispatch(action._setVideoStatus(true))
              if (e.shiftKey) dispatch(action._prevVideo(10))
              else dispatch(action._prevVideo(1))
            }
          } else {
            if (!label.imageStatus) {
              dispatch(action._setImageStatus(true))
              if (e.shiftKey) dispatch(action._prevImage(10))
              else dispatch(action._prevImage(1))
            }
          }
          break
        case 68:
          if (props.dataType === "V") {
            if (!label.videoStatus) {
              dispatch(action._setVideoStatus(true))
              if (e.shiftKey) dispatch(action._nextVideo(10))
              else dispatch(action._nextVideo(1))
            }
          } else {
            if (!label.imageStatus) {
              dispatch(action._setImageStatus(true))
              if (e.shiftKey) dispatch(action._nextImage(10))
              else dispatch(action._nextImage(1))
            }
            break
          }
          break
        case 82:
          if (props.dataType === "I") {
            _handleReturnImgSize()
          }
          break
        case 70:
          if (props.dataType === "I") {
            _handleZoomFixClick()
          }
          break
        default:
          break
      }
    },
    [props.dataType, label.imageStatus, label.videoStatus, label.modalCheck, imageLabel.isResetImageSize, imageLabel.isZoomFix]
  )

  useEffect(() => {
    window.addEventListener("keydown", _handleKeyDown)
    return () => {
      window.removeEventListener("keydown", _handleKeyDown)
    }
  }, [_handleKeyDown])

  const _handleSizeSlider = useCallback(
    value => {
      dispatch(ImageLabelActions._setBrushSize(value))
    },
    [dispatch]
  )

  const _deleteAll = useCallback(() => {
    const tag = imageLabel.curTag
    const cBrushList = cloneDeep(imageLabel.brushList)
    const filter = cBrushList.filter(ele => tag.TAG_CD !== ele.TAG_CD)
    dispatch(ImageLabelActions._setBrushList(filter))
  }, [imageLabel.curTag, imageLabel.brushList, dispatch])

  const prevVideo = useCallback(() => {
    if (!videoLabel.videoStatus) {
      dispatch(VideoLabelActions._setVideoStatus(true))
      dispatch(VideoLabelActions._prevVideo(1))
    }
  }, [videoLabel.videoStatus, dispatch])

  const nextVideo = useCallback(() => {
    if (!videoLabel.videoStatus) {
      dispatch(VideoLabelActions._setVideoStatus(true))
      dispatch(VideoLabelActions._nextVideo(1))
    }
  }, [videoLabel.videoStatus, dispatch])

  const prevImage = useCallback(() => {
    if (!imageLabel.imageStatus) {
      dispatch(ImageLabelActions._setImageStatus(true))
      dispatch(ImageLabelActions._prevImage(1))
    }
  }, [imageLabel.imageStatus, dispatch])

  const nextImage = useCallback(() => {
    if (!imageLabel.imageStatus) {
      dispatch(ImageLabelActions._setImageStatus(true))
      dispatch(ImageLabelActions._nextImage(1))
    }
  }, [imageLabel.imageStatus, dispatch])

  return (
    <div className="label_top_wrap" style={{ display: "flex", alignItems: "center" }}>
      {props?.dataType === "V" && (
        <div className="w-100 grid-container">
          <div className="label-slider d-plex"></div>
          <div className="label-fileName">
            <div className="mb-1">
              <IoIosArrowBack className="label-arrow" data-tip={"Prev Video [a]"} onClick={prevVideo} />
              <span className="label-title">{fileName}</span>
              <IoIosArrowForward className="label-arrow" data-tip={"Next Video [d]"} onClick={nextVideo} />
            </div>
          </div>
          {Object.keys(videoLabel?.curTag).length !== 0 && (
            <div className="label-tag mr-2" data-tip={`Current Tag : ${videoLabel.curTag.NAME}`}>
              <RiPriceTag3Line className="mr-2 font-25" />
              <BsFillCircleFill className="font-25" color={videoLabel.curTag.COLOR} />
            </div>
          )}
        </div>
      )}

      {props?.dataType === "I" && (
        <div className="w-100 grid-container">
          <div className="label-slider d-flex">
            {(imageLabel.btnSts === "isBrush" || imageLabel.btnSts === "isEraser") && (
              <>
                <div className="d-flex" style={{ width: "30%" }}>
                  <MdOpacity className="font-25" data-tip={"Opacity"} />
                  <InputNumber
                    min={0}
                    max={100}
                    step={5}
                    value={imageLabel.opacityValue}
                    autoFocus={false}
                    focusOnUpDown={false}
                    className="ml-1"
                    style={{ width: "90%" }}
                    formatter={value => `${value}%`}
                    parser={value => value.replace("%", "")}
                    onChange={value => {
                      dispatch(ImageLabelActions._setOpacityValue(value))
                    }}
                    onFocus={e => {
                      e.target.blur()
                    }}
                  />
                </div>
                <div className="d-flex pl-1" style={{ width: "60%" }}>
                  <RiBrush2Line className="font-25" data-tip={"Brush Size [Ctrl + Wheel]"} />
                  <Slider className="pl-1" min={1} max={30} step={1} value={imageLabel.brushSize} _handleSlider={_handleSizeSlider} />
                </div>
                {imageLabel.btnSts === "isEraser" && (
                  <div style={{ width: "10%" }}>
                    <RiDeleteBin4Line
                      data-tip={"Current Tag Brush Delete All"}
                      className="ml-1 icon-pointer"
                      size={"20px"}
                      onClick={_deleteAll}
                    />
                  </div>
                )}
              </>
            )}
          </div>
          <div className="label-fileName">
            <div className="mb-1">
              <IoIosArrowBack data-tip={"Prev Image [a]"} className="label-arrow" onClick={prevImage} />
              <span className="label-title stop-dragging">{fileName}</span>
              <IoIosArrowForward data-tip={"Next Image [d]"} className="label-arrow" onClick={nextImage} />
            </div>
          </div>
          <div className="label-image-control">
            <div className="d-flex float-right"></div>
          </div>
          {Object.keys(imageLabel?.curTag).length !== 0 && (
            <div className="label-tag mr-2">
              <RiPictureInPictureExitFill data-tip="Reset Scale [r]" className="icon-pointer mr-2 font-25" onClick={_handleReturnImgSize} />
              {imageLabel.isZoomFix ? (
                <RiPushpin2Line data-tip="Zoom Fix Cancel [f]" className="icon-pointer mr-2 font-25" onClick={_handleZoomFixClick} />
              ) : (
                <RiPushpinLine data-tip="Zoom Fix [f]" className="icon-pointer mr-2 font-25" onClick={_handleZoomFixClick} />
              )}
              <span data-tip={`Current Tag : ${imageLabel.curTag.NAME}`}>
                <RiPriceTag3Line className="mr-2 font-25" />
                <BsFillCircleFill className="font-25" color={imageLabel.curTag.COLOR} />
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

LabelTopBar.propTypes = {}

export default React.memo(LabelTopBar)
