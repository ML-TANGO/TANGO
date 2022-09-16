import React, { useState, useEffect, useRef, useCallback } from "react"
import { Stage, Layer, Text, Line } from "react-konva/"
import { useDispatch, useSelector } from "react-redux"
import { Row, Col } from "reactstrap"
import LoadingOverlay from "react-loading-overlay"
import { toast } from "react-toastify"
import { FaMagic, FaPauseCircle } from "react-icons/fa"
import { MdError, MdInfoOutline, MdFindReplace } from "react-icons/md"
import { RiSearchEyeLine } from "react-icons/ri"
import { cloneDeep } from "lodash-es"
import { confirmAlert } from "react-confirm-alert"
import { IoMdWalk } from "react-icons/io"

import * as VideoLabelActions from "Redux/Actions/VideoLabelActions"
import * as VideoAnnoApi from "Config/Services/VideoAnnoApi"

import KonvaComponents from "Components/Common/KonvaComponents"
import VideoPlayer from "Components/Common/VideoPlayer"
import VideoControlBar from "../../../../../Components/Common/VideoControlBar"

import CursorRect from "./Drawers/CursorRect"
import CursorPolygon from "./Drawers/CursorPolygon"
import DrawRect from "./Drawers/DrawRect"
import DrawPolygon from "./Drawers/DrawPolygon"

import { _getNeedCount } from "Components/Utils/Utils"
import CommonToast from "../../../../../Components/Common/CommonToast"
import CommonButton from "../../../../../Components/Common/CommonButton"
import useResizeListener from "../../../../../Components/Utils/useResizeListener"

/**
 * Video Annotation 영역
 *
 * @param {*} props
 * @returns
 */
function DrawVideo(props) {
  const stageCanvasRef = useRef(null)
  const stageRef = useRef(null)
  const segLayerRef = useRef(null)
  const [canvasWidth, canvasHeight] = useResizeListener(stageCanvasRef)
  // Konva Layout
  const [mouseX, setMouseX] = useState(null)
  const [mouseY, setMouseY] = useState(null)
  const [stageX, setStageX] = useState(0)
  const [stageY, setStageY] = useState(0)
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const [stageScale, setStageScale] = useState(1)
  const [stageSize, setStageSize] = useState({ w: 0, h: 0 })
  const [defaultVideoSize, setDefaultVideoSize] = useState({ w: 0, h: 0 })

  // video Player
  const [src, setSrc] = useState(null)
  const [control, setControl] = useState({ paused: true })
  const [paused, setPaused] = useState(true)
  const [replay, setReplay] = useState(false)
  const [isPlay, setIsPlay] = useState(false) // play 가능 상태
  const [duration, setDuration] = useState(0)
  const [frameBound, setFrameBound] = useState(1)
  const [videoInfo, setVideoInfo] = useState({
    currentTime: 0,
    bufferedTime: 0,
    curFrame: 0
  })

  // Draw
  const [isDraggable, setIsDraggable] = useState(false)
  const [isMultiLine, setIsMultiLine] = useState(false)
  const [objectList, setObjectList] = useState([])
  const [, setBrushList] = useState([])
  const [trackerList, setTrackerList] = useState([])
  const [curObject, setCurObject] = useState({})
  const [inImage, setInImage] = useState(false)
  const [clickObject, setClickObject] = useState([])
  const [clickTracker, setClickTracker] = useState([])
  const [isCursor, setIsCursor] = useState(true)

  const [isError, setIsError] = useState(false)
  const [errorMessage, setErrorMessage] = useState(null)

  const dispatch = useDispatch()
  const videoLabel = useSelector(
    state => state.videoLabel,
    (prevItem, item) => {
      if (item === prevItem) return true
      else return false
    }
  )

  useEffect(() => {
    setSrc(videoLabel.curVideo.FILE_URL)
    setIsError(false)
    setErrorMessage(null)
    setPaused(true)
    setReplay(false)
    setControl({ paused: true })
    setVideoInfo({ currentTime: 0, bufferedTime: 0, curFrame: 0 })
    dispatch(VideoLabelActions._setCurFrame(0))
    setCurObject({})
    setTrackerList([])
    setClickObject([])
    setClickTracker([])
  }, [videoLabel.curVideo])

  // Redux objectList change useEffect
  useEffect(() => {
    setObjectList(videoLabel.objectList)
  }, [videoLabel.objectList])

  useEffect(() => {
    setBrushList(videoLabel.brushList)
  }, [videoLabel.brushList])

  // 총 재생시간이 변경 될 때 (video가 변경됐다고 봄)
  // 총 프레임 개수만큼 objectList Array 생성
  useEffect(() => {
    if (duration !== 0) {
      let totalFrame = Math.floor(duration * videoLabel.curVideo?.FPS) || 0
      if (!videoLabel?.curVideo?.ANNO_DATA?.POLYGON_DATA?.length) {
        let arr = new Array(totalFrame).fill(null).map(() => new Array())
        // let cCurObject = cloneDeep(curObject)
        // cCurObject.POSITION = arr
        // setCurObject(arr)
        dispatch(VideoLabelActions._setObjectList(arr))
      }
      // if (props.dataSet.OBJECT_TYPE === "S" && !videoLabel?.curVideo?.ANNO_DATA?.BRUSH_DATA?.length) {
      //   let arr = new Array(totalFrame).fill(null).map(ele => new Array())
      //   setBrushList(arr)
      // }
    }
  }, [duration])

  // 버튼 상태 변경 useEffect
  useEffect(() => {
    setIsCursor(true)
    setIsDraggable(true)
    stageRef.current.container().style.cursor = "none"
    _setObject()
  }, [videoLabel.btnSts])

  useEffect(() => {
    if (src !== null && defaultVideoSize.w !== 0 && defaultVideoSize.h !== 0) {
      let width = defaultVideoSize.w
      let height = defaultVideoSize.h
      let scale, stageX, stageY
      if (width > canvasWidth || height > canvasHeight) {
        // 영상 원본 사이즈와 현재 canvas 사이즈 비례값 계산으로 현재 canvasHeight 기준으로 화면에 꽉 차는 예상 width 구함
        const x = (width * canvasHeight) / height
        // stage값 계산 0보다 작을 시 stage 변경 안 함
        stageX = (canvasWidth - x) / 2
        stageX = stageX < 0 ? 0 : stageX

        // 영상 원본 사이즈와 현재 canvas 사이즈 비례값 계산으로 현재 canvasWidth 기준으로 화면에 꽉 차는 예상 height 구함
        const y = (height * canvasWidth) / width
        // stage값 계산 0보다 작을 시 stage 변경 안 함
        stageY = (canvasHeight - y) / 2
        stageY = stageY < 0 ? 0 : stageY

        // stage 값 기준으로 scale 계산할 기준 구함
        if (stageX > stageY) {
          scale = canvasHeight / height
        } else {
          scale = canvasWidth / width
        }
        // setStageX(stageX)
        // setStageY(stageY)

        let stageW = defaultVideoSize.w * scale
        let stageH = defaultVideoSize.h * scale
        let offsetX = (canvasWidth - stageW) / 2
        let offsetY = (canvasHeight - stageH) / 2
        setStageX(0)
        setStageY(0)
        setOffset({ x: offsetX, y: offsetY })
        setStageSize({ w: stageW, h: stageH })
        setStageScale(scale)
      } else {
        let offsetX = (canvasWidth - width) / 2
        let offsetY = (canvasHeight - height) / 2
        setStageX(0)
        setStageY(0)
        setOffset({ x: offsetX, y: offsetY })
        setStageSize({ w: defaultVideoSize.w, h: defaultVideoSize.h })
        setStageScale(1)
      }
    }
  }, [src, defaultVideoSize.w, defaultVideoSize.h, canvasWidth, canvasHeight])

  // Tag가 변경됐을 때 현재 Object Color, Brush Color 와 Tag Color 비교
  useEffect(() => {
    let flag = false
    if (objectList.length !== 0) {
      const newObjectList = objectList.map(list => {
        return list.map(ele => {
          if (
            String(ele.TAG_CD) === String(videoLabel.curTag.TAG_CD) &&
            (ele.COLOR !== videoLabel.curTag.COLOR || ele.TAG_NAME !== videoLabel.curTag.NAME)
          ) {
            ele.COLOR = videoLabel.curTag.COLOR
            ele.TAG_NAME = videoLabel.curTag.NAME
            flag = true
            return ele
          } else {
            return ele
          }
        })
      })
      setObjectList(newObjectList)
    }
    if (flag) dispatch(VideoLabelActions._isDrawAction(true))
    _setObject()
  }, [videoLabel.curTag])

  // useEffect(() => {
  //   if (!isNaN(curFrame) || curFrame !== null) {
  //     if (trackerList.length !== 0) setTrackerList([])
  //     setClickObject([])
  //     setClickTracker([])
  //     // dispatch(VideoLabelActions._setCurFrame(curFrame))
  //     // _setObject()
  //   }
  // }, [curFrame])

  useEffect(() => {
    setTrackerList([])
    setClickObject([])
    setClickTracker([])
    _setObject()
  }, [paused])

  useEffect(() => {
    dispatch(VideoLabelActions._setFrameBound(frameBound))
  }, [frameBound])

  const dispatchFrame = frame => {
    dispatch(VideoLabelActions._setCurFrame(frame))
  }

  const _setObject = useCallback(() => {
    if (videoLabel.curVideo.length === 0) return
    let curObj = {
      DATASET_CD: videoLabel.curVideo.DATASET_CD,
      DATA_CD: videoLabel.curVideo.DATA_CD,
      TAG_CD: videoLabel.curTag.TAG_CD,
      TAG_NAME: videoLabel.curTag.NAME,
      CLASS_CD: videoLabel.curTag.CLASS_CD,
      COLOR: videoLabel.curTag.COLOR,
      CURSOR: videoLabel.btnSts,
      NEEDCOUNT: _getNeedCount(videoLabel.btnSts),
      POSITION: []
    }
    setCurObject(curObj)
  }, [videoLabel.curVideo, videoLabel.btnSts, videoLabel.curTag])

  /*
   * Konva Control function
   * _handleWheel
   * _handleDragMove
   * _handleDragEnd
   * _handleMouseMove
   * _handleMouseClick
   */

  const _handleWheel = useCallback(
    e => {
      // get target component info
      e.evt.preventDefault()
      const stage = e.target.getStage()
      const oldScale = stage.scaleX()
      // const oldScale = stageScale
      // set scale interval
      // jogoon 낮은 스케일 구간에서 스케일 변환값 변경
      const scaleBy = stage.scaleX() > 1 ? 1.1 : 1.2
      // move mouse pointer when scale has been changed
      const mousePointTo = {
        x: stage.getPointerPosition().x / oldScale - stage.x() / oldScale,
        y: stage.getPointerPosition().y / oldScale - stage.y() / oldScale
      }
      // check mouse wheel direction ( zoom in / out )
      let newScale = e.evt.deltaY < 0 ? oldScale * scaleBy : oldScale / scaleBy
      if (newScale > 2.5 || newScale < 0.1) {
        return
      }
      let width = defaultVideoSize.w
      let height = defaultVideoSize.h
      let w = width * newScale
      let h = height * newScale

      let offsetX = -(mousePointTo.x - stage.getPointerPosition().x / newScale) * newScale
      let offsetY = -(mousePointTo.y - stage.getPointerPosition().y / newScale) * newScale

      setStageScale(newScale)
      setStageSize({ w: w, h: h })
      setOffset({ x: offsetX + offset.x, y: offsetY + offset.y })
    },
    [defaultVideoSize, offset]
  )

  const _handleDragMove = useCallback(
    e => {
      const stage = e.target.getStage()
      const mousePointTo = {
        x: stage.getPointerPosition().x / stageScale - stage.x() / stageScale,
        y: stage.getPointerPosition().y / stageScale - stage.y() / stageScale
      }

      stage.x(0)
      stage.y(0)
      stage.scaleX(stageScale)
      stage.scaleY(stageScale)

      let offsetX = -(mousePointTo.x - stage.getPointerPosition().x / stageScale) * stageScale
      let offsetY = -(mousePointTo.y - stage.getPointerPosition().y / stageScale) * stageScale
      setOffset({ x: offsetX + offset.x, y: offsetY + offset.y })
      stage.getLayers()[0].draw()
      stage.getLayers()[1].draw()
    },
    [stageScale, offset]
  )

  const _handleDragEnd = useCallback(
    e => {
      const stage = e.target.getStage()
      stage.x(0)
      stage.y(0)
      stage.scaleX(stageScale)
      stage.scaleY(stageScale)
      stage.getLayers()[0].draw()
      stage.getLayers()[1].draw()
      const attrs = stage.getAttrs()
      const x = (stage.getPointerPosition().x - attrs.x) / stageScale
      const y = (stage.getPointerPosition().y - attrs.y) / stageScale
      setMouseX(x)
      setMouseY(y)
    },
    [stageScale]
  )

  const _handleMouseMove = useCallback(
    e => {
      const stage = e.currentTarget
      const attrs = stage.getAttrs()
      const x = (stage.getPointerPosition().x - attrs.x) / stageScale
      const y = (stage.getPointerPosition().y - attrs.y) / stageScale
      const w = defaultVideoSize.w
      const h = defaultVideoSize.h
      if (paused) {
        if (x < 0 || y < 0 || x > w || y > h) {
          setInImage(false)
        } else {
          setInImage(true)
        }
      }
      if (!paused) return
      setMouseX(x)
      setMouseY(y)
    },
    [stageScale, defaultVideoSize, paused]
  )

  const _handleMouseEnter = useCallback(e => {
    e.target.getStage().container().style.cursor = "none"
    setIsCursor(true)
  }, [])

  const _handleMouseLeave = useCallback(e => {
    setIsCursor(false)
    e.target.getStage().container().style.cursor = "auto"
  }, [])

  const _drawAction = useCallback(
    (x, y) => {
      const len = curObject.POSITION.length
      let cCurObject = cloneDeep(curObject)

      let multilineFlag
      if (curObject.NEEDCOUNT === -1) {
        setIsMultiLine(true)
        multilineFlag = true
      } else {
        setIsMultiLine(false)
        multilineFlag = false
      }

      if (len < curObject.NEEDCOUNT || (curObject.NEEDCOUNT === -1 && multilineFlag)) {
        const pos = { X: x, Y: y }
        cCurObject.POSITION.push(pos)
        setCurObject(cCurObject)
      }

      if (!multilineFlag && cCurObject.POSITION.length === curObject.NEEDCOUNT) {
        if (videoLabel.btnSts === "isMagic") {
          // Magic tool draw
          _setObject()
          dispatch(VideoLabelActions._setVideoStatus(true))
          const param = {
            DATASET_CD: videoLabel.curVideo.DATASET_CD,
            DATA_CD: videoLabel.curVideo.DATA_CD,
            TAG_CD: videoLabel.curTag.TAG_CD,
            COLOR: videoLabel.curTag.COLOR,
            CLASS_CD: videoLabel.curTag.CLASS_CD,
            OBJECT_TYPE: props.dataSet.OBJECT_TYPE,
            RECT: cCurObject.POSITION,
            IS_READY: true,
            AI_CD: videoLabel.curTag.BASE_MDL,
            START_FRAME: videoInfo.curFrame,
            END_FRAME: videoInfo.curFrame
          }
          if (videoLabel.curTag.CLASS_CD === null || videoLabel.curTag.CLASS_CD === "") {
            dispatch(VideoLabelActions._setVideoStatus(false))
            toast.error(<CommonToast Icon={MdError} text={"Current tag is unpredictable"} />)
            return
          }
          VideoAnnoApi._getPredictResult(param)
            .then(data => {
              // 모델 상태 체크
              if (data[0]?.status) {
                if (data[0].status === 2) {
                  // model full
                  toast.error(<CommonToast Icon={MdInfoOutline} text={"The maximum number of processes has been exceeded"} />)
                  dispatch(VideoLabelActions._setVideoStatus(false))
                  dispatch(VideoLabelActions._statusModal(true))
                  // _openStatusModal()
                  return
                } else if (data[0].status === 3) {
                  // model 안 올라옴 다시 올리기
                  toast.error(<CommonToast Icon={MdInfoOutline} text={"The model is not running"} />)
                  dispatch(VideoLabelActions._setVideoStatus(false))
                  confirmAlert({
                    customUI: ({ onClose }) => {
                      return (
                        <div className="react-confirm-alert-custom">
                          <h1>
                            <FaPauseCircle />
                            The model is not running
                          </h1>
                          <p className="taglist moving">
                            Start Mini Predict Model
                            <IoMdWalk />
                          </p>

                          <div className="custom-buttons">
                            <CommonButton
                              className="bg-green"
                              text="Apply"
                              onClick={() => {
                                param.IS_READY = false
                                onClose()
                                dispatch(VideoLabelActions._setVideoStatus(true))
                                VideoAnnoApi._getPredictResult(param)
                                  .then(() => {
                                    dispatch(VideoLabelActions._checkModel(true))
                                    dispatch(VideoLabelActions._setVideoStatus(false))
                                    toast.info(<CommonToast Icon={MdFindReplace} text={"Model Start!!"} />)
                                  })
                                  .catch(e => {
                                    dispatch(VideoLabelActions._checkModel(true))
                                    dispatch(VideoLabelActions._setVideoStatus(false))
                                    console.log(e)
                                    toast.error(<CommonToast Icon={MdInfoOutline} text={"Model Start Fail"} />)
                                  })
                              }}
                            />
                            <CommonButton
                              className="bg-red"
                              text="Cancel"
                              onClick={() => {
                                onClose()
                              }}
                            />
                          </div>
                        </div>
                      )
                    }
                  })
                  return
                }
              }

              if (data[0]?.ANNO_DATA?.length !== 0) {
                // 결과 저장
                const arr = data[0].ANNO_DATA.map(ele => {
                  let obj = {}
                  obj.DATASET_CD = videoLabel.curVideo.DATASET_CD
                  obj.DATA_CD = videoLabel.curVideo.DATA_CD
                  obj.CLASS_CD = videoLabel.curTag.CLASS_CD
                  obj.TAG_CD = videoLabel.curTag.TAG_CD
                  obj.COLOR = videoLabel.curTag.COLOR
                  obj.TAG_NAME = videoLabel.curTag.NAME
                  obj.CURSOR = ele.CURSOR
                  obj.NEEDCOUNT = _getNeedCount(obj.CURSOR)
                  obj.POSITION = ele.POSITION
                  return obj
                })
                let cObjectList = cloneDeep(objectList)
                let cObject = cObjectList[videoInfo.curFrame]
                cObject = [...cObject, ...arr]
                cObjectList[videoInfo.curFrame] = cObject
                dispatch(VideoLabelActions._setObjectList(cObjectList))
                toast.info(<CommonToast Icon={FaMagic} text={"Magic Predict Success"} />)
              } else {
                toast.error(<CommonToast Icon={MdInfoOutline} text={"Magic Predict No result"} />)
              }
              dispatch(VideoLabelActions._setVideoStatus(false))
            })
            .catch(e => {
              dispatch(VideoLabelActions._setVideoStatus(false))
              console.log(e)
              toast.error(<CommonToast Icon={MdError} text={"Magic Predict Fail"} />)
            })
        } else if (videoLabel.btnSts === "isTracker") {
          let cTrackerList = cloneDeep(trackerList)
          cTrackerList.push(cCurObject)
          setTrackerList(cTrackerList)
          _setObject()
        } else {
          let cObjectList = cloneDeep(objectList)
          cObjectList[videoInfo.curFrame]?.push(cCurObject)
          dispatch(VideoLabelActions._setObjectList(cObjectList))
          _setObject()
        }
      }
    },
    [
      videoInfo.curFrame,
      curObject,
      props.dataSet.OBJECT_TYPE,
      videoLabel.curVideo,
      videoLabel.btnSts,
      trackerList,
      objectList,
      videoLabel.curTag,
      _setObject,
      dispatch
    ]
  )

  const _handleMouseClick = useCallback(
    e => {
      e.evt.preventDefault()

      const stage = e.currentTarget
      const attrs = stage.getAttrs()
      // get mouse position
      const x = (stage.getPointerPosition().x - attrs.x) / stageScale
      const y = (stage.getPointerPosition().y - attrs.y) / stageScale
      if (x < 0 || y < 0 || x > defaultVideoSize.w || y > defaultVideoSize.h || !paused) {
        return
      }
      switch (videoLabel.btnSts) {
        case "isRect":
        case "isPolygon":
        case "isMagic":
        case "isTracker":
          if (videoLabel.curTag.TAG_CD === undefined) {
            dispatch(VideoLabelActions._openTag())
          } else {
            dispatch(VideoLabelActions._isDrawAction(true))
            _drawAction(x, y)
          }
          break
        default:
          break
      }
    },
    [stageSize, defaultVideoSize, videoLabel.curTag, videoLabel.btnSts, paused, stageScale, _drawAction, dispatch]
  )

  /**
   * default Object Contorl function
   * _setClickObject
   * _changePos
   * _deleteObject
   */

  const _setClickObject = useCallback(
    id => {
      const indexof = clickObject.indexOf(id)
      let arr = cloneDeep(clickObject)
      if (indexof === -1) {
        arr.push(id)
        setClickObject(arr)
      } else {
        const filter = arr.filter(ele => ele !== id)
        setClickObject(filter)
      }
    },
    [clickObject]
  )

  const _changePos = useCallback(
    async (id, pos, mode) => {
      const ids = id.split("_")
      const idx = Number(ids[0])
      const posIdx = Number(ids[1])
      let cObjectList = cloneDeep(objectList)
      let obj = cObjectList[videoInfo.curFrame]
      if (mode === "all") {
        obj[idx].POSITION = pos
      } else if (mode === "add") {
        let position = obj[idx].POSITION
        position.splice(posIdx, 0, pos)
        obj[idx].POSITION = position
      } else {
        obj[idx].POSITION[posIdx] = pos
      }
      dispatch(VideoLabelActions._setObjectList(cObjectList))
      dispatch(VideoLabelActions._isDrawAction(true))
      return true
    },
    [objectList, videoInfo.curFrame, dispatch]
  )

  const _deleteObject = useCallback(
    (id, mode) => {
      let cObjectList = cloneDeep(objectList)
      let obj = cObjectList[videoInfo.curFrame]
      if (mode === "point") {
        const ids = id.split("_")
        const idx = Number(ids[0])
        const posIdx = Number(ids[1])
        let points = obj[idx].POSITION
        let filter = points.filter((ele, index) => index !== posIdx)
        obj[idx].POSITION = filter
      } else {
        if (clickObject.length !== 0) {
          let filter = obj.filter((ele, index) => !clickObject.some(clicked => clicked === index))
          cObjectList[videoInfo.curFrame] = filter
          setClickObject([])
        }

        if (clickTracker.length !== 0) {
          let filter = trackerList.filter((ele, index) => !clickTracker.some(clicked => clicked === index))
          setTrackerList(filter)
          setClickTracker([])
        }
      }
      dispatch(VideoLabelActions._setObjectList(cObjectList))
      dispatch(VideoLabelActions._isDrawAction(true))
    },
    [objectList, videoInfo.curFrame, clickObject, trackerList, clickTracker, dispatch]
  )

  /**
   * tracker Object Contorl function
   * _setClickTracker
   * _changeTracker
   * _handleTracker
   */

  const _setClickTracker = useCallback(
    id => {
      const indexof = clickTracker.indexOf(id)
      let arr = clickTracker
      if (indexof === -1) {
        arr.push(id)
        setClickTracker(arr)
      } else {
        const filter = arr.filter(ele => ele !== id)
        setClickTracker(filter)
      }
    },
    [clickTracker]
  )

  const _changeTracker = useCallback(
    async (id, pos, mode) => {
      const ids = id.split("_")
      const idx = Number(ids[0])
      const posIdx = Number(ids[1])
      let cTrackerList = cloneDeep(trackerList)
      if (mode === "all") {
        cTrackerList[idx].POSITION = pos
      } else if (mode === "add") {
        let position = cTrackerList[idx].POSITION
        position.splice(posIdx, 0, pos)
        cTrackerList[idx].POSITION = position
      } else {
        cTrackerList[idx].POSITION[posIdx] = pos
      }
      setTrackerList(cTrackerList)
      return true
    },
    [trackerList]
  )

  const _handleTracker = useCallback(() => {
    if (!paused) return
    if (trackerList.length === 0) {
      toast.error(<CommonToast Icon={MdError} text={"Please Draw Tracker Rect"} />)
      return
    }

    const param = {
      DATASET_CD: videoLabel.curVideo.DATASET_CD,
      FILE_PATH: videoLabel.curVideo.FILE_PATH,
      START_FRAME: videoInfo.curFrame,
      END_FRAME: videoInfo.curFrame + frameBound,
      TRACKER_INFO: trackerList
    }
    dispatch(VideoLabelActions._setVideoStatus(true))
    VideoAnnoApi._getTrackResult(param)
      .then(result => {
        let cObjectList = cloneDeep(objectList)
        result.TRACKER_INFO?.forEach(tracker => {
          tracker.forEach(ele => {
            cObjectList[ele.FRAME_NUMBER].push(ele)
          })
        })
        dispatch(VideoLabelActions._setObjectList(cObjectList))
        setTrackerList([])
        dispatch(VideoLabelActions._setVideoStatus(false))
        toast.info(<CommonToast Icon={RiSearchEyeLine} text={"Find tracker Success"} />)
      })
      .catch(e => {
        toast.error(<CommonToast Icon={MdError} text={"Failed to find tracker"} />)
        dispatch(VideoLabelActions._setVideoStatus(false))
        console.log(e)
      })
  }, [trackerList, objectList, videoInfo.curFrame, videoLabel.curVideo, frameBound, paused, dispatch])

  const polygonClick = useCallback(() => {
    if (isMultiLine && videoLabel.btnSts === "isPolygon") {
      let cObjectList = cloneDeep(objectList)
      cObjectList[videoInfo.curFrame].push(curObject)
      dispatch(VideoLabelActions._setObjectList(cObjectList))
      setIsMultiLine(false)
      _setObject()
    }
  }, [objectList, curObject, isMultiLine, videoLabel.btnSts, _setObject, dispatch, videoInfo.curFrame])

  // key Event
  const _handleKeyDown = useCallback(
    e => {
      if (videoLabel.modalCheck) {
        return
      }
      switch (e.keyCode) {
        case 32:
          // space key
          if (isMultiLine && videoLabel.btnSts === "isPolygon") {
            let cObjectList = cloneDeep(objectList)
            cObjectList[videoInfo.curFrame].push(curObject)
            dispatch(VideoLabelActions._setObjectList(cObjectList))
            setIsMultiLine(false)
            _setObject()
          } else {
            if (isPlay) {
              if (paused) {
                setPaused(false)
                setControl({ k: "paused", v: false })
              } else {
                if (replay) {
                  setPaused(true)
                  setReplay(false)
                  setControl({ k: "replay", v: true })
                } else {
                  setPaused(true)
                  setControl({ k: "paused", v: true })
                }
              }
            }
          }
          break
        case 46:
          // delete key
          _deleteObject()
          break
        case 75:
          // k
          if (isPlay) {
            if (paused) {
              setPaused(false)
              setControl({ k: "paused", v: false })
            } else {
              if (replay) {
                setPaused(true)
                setReplay(false)
                setControl({ k: "replay", v: true })
              } else {
                setPaused(true)
                setControl({ k: "paused", v: true })
              }
            }
          }
          break
        case 78:
        case 39:
          // n or ->
          if (isPlay) {
            if (e.shiftKey) {
              // next 10 Frame
              setControl({ k: "move", v: 10 })
            } else {
              // next Frame
              setControl({ k: "move", v: 1 })
            }
          }
          break
        case 66:
        case 37:
          // b or <-
          if (isPlay) {
            if (e.shiftKey) {
              // prev 10 Frame
              setControl({ k: "move", v: -10 })
            } else {
              // prev Frame
              setControl({ k: "move", v: -1 })
            }
          }
          break
        case 27:
          // esc key
          _setObject()
          break
        default:
          break
      }
    },
    [
      isMultiLine,
      objectList,
      videoInfo.curFrame,
      curObject,
      isPlay,
      paused,
      videoLabel.modalCheck,
      videoLabel.btnSts,
      replay,
      _setObject,
      _deleteObject,
      dispatch
    ]
  )

  // // KeyDown useEffect
  useEffect(() => {
    window.addEventListener("keydown", _handleKeyDown)
    return () => {
      window.removeEventListener("keydown", _handleKeyDown)
    }
  }, [_handleKeyDown])

  const _switchCursor = useCallback(() => {
    const props = {
      mouseX: mouseX,
      mouseY: mouseY,
      color: curObject.COLOR,
      needCount: curObject.NEEDCOUNT,
      position: curObject.POSITION,
      scale: stageScale,
      polygonClick: polygonClick
    }

    switch (videoLabel.btnSts) {
      case "isRect":
      case "isMagic":
      case "isTracker":
        return <CursorRect {...props} />
      case "isPolygon":
        return <CursorPolygon {...props} />
      default:
        return null
    }
  }, [mouseX, mouseY, curObject, stageScale, videoLabel.btnSts, polygonClick])

  const _switchDraw = useCallback(
    (ele, index, tracker) => {
      const filter = videoLabel?.tagList?.filter(tag => Number(tag.TAG_CD) === Number(ele.TAG_CD))

      const isShow = filter[0]?.isShow
      const props = {
        key: index,
        id: index,
        cursor: ele.CURSOR,
        color: ele.COLOR,
        position: ele.POSITION,
        scale: stageScale,
        isMove: true,
        imageWidth: defaultVideoSize.w,
        imageHeight: defaultVideoSize.h,
        _changePos: tracker ? _changeTracker : _changePos,
        _deleteObject: _deleteObject,
        _setClickObject: tracker ? _setClickTracker : _setClickObject,
        btnSts: videoLabel.btnSts,
        tagName: ele.TAG_NAME,
        isDash: tracker ? true : false,
        tagCd: ele.TAG_CD,
        curTagCd: videoLabel.curTag.TAG_CD,
        isShow: isShow,
        acc: ele?.ACCURACY,
        isShowPoint: paused,
        setIsCursor: setIsCursor
      }
      switch (ele.CURSOR) {
        case "isRect":
        case "isTracker":
          return <DrawRect {...props} />
        case "isPolygon":
          return <DrawPolygon {...props} />
        default:
          return null
      }
    },
    [videoLabel, defaultVideoSize, stageScale, paused, _changeTracker, _changePos, _deleteObject, _setClickTracker, _setClickObject]
  )

  return (
    <Row className="w-100 h-100" noGutters>
      <Col md={12} className="h-100">
        <LoadingOverlay
          className="w-100 h-100"
          active={src == null || (!isPlay && !isError) || videoLabel.videoStatus || videoLabel.preStatus}
          spinner
          text="Loading Video..."
        >
          {isError ? (
            <div
              className="w-100 h-100"
              style={{
                position: "relative",
                background: "black",
                zIndex: "1"
              }}
            >
              <div className="w-100" style={{ position: "absolute", top: "50%" }}>
                <MdError className="mr-2 icon-error" />
                {errorMessage}
              </div>
            </div>
          ) : (
            <>
              <div
                id="drawVideo"
                style={{
                  overflow: "hidden",
                  position: "relative",
                  width: "100%",
                  height: "calc(100% - 60px)",
                  background: "black",
                  zIndex: "0"
                }}
                ref={stageCanvasRef}
              >
                <div
                  style={{
                    position: "absolute",
                    left: offset.x + "px",
                    top: offset.y + "px",
                    width: stageSize.w,
                    height: stageSize.h
                  }}
                >
                  <Stage
                    ref={stageRef}
                    width={stageSize.w}
                    height={stageSize.h}
                    x={stageX}
                    y={stageY}
                    scaleX={stageScale}
                    scaleY={stageScale}
                    onWheel={_handleWheel}
                    onClick={_handleMouseClick}
                    // onDragStart={_handleDragStart}
                    onDragMove={_handleDragMove}
                    onDragEnd={_handleDragEnd}
                    onMouseMove={_handleMouseMove}
                    onMouseEnter={_handleMouseEnter}
                    onMouseLeave={_handleMouseLeave}
                    draggable={isDraggable}
                  >
                    <Layer>
                      <KonvaComponents containerId="drawVideo">
                        <div
                          style={{
                            position: "absolute",
                            left: offset.x + "px",
                            top: offset.y + "px",
                            width: stageSize.w,
                            height: stageSize.h,
                            background: "black",
                            zIndex: "-1"
                            // transform: `scale(${stageScale})`
                          }}
                        >
                          <VideoPlayer
                            src={src}
                            fps={videoLabel.curVideo?.FPS}
                            scale={stageScale}
                            stageSize={stageSize}
                            control={control}
                            setReplay={setReplay}
                            setVideoInfo={setVideoInfo}
                            setDuration={setDuration}
                            setDefaultVideoSize={setDefaultVideoSize}
                            setIsPlay={setIsPlay}
                            setIsError={setIsError}
                            setErrorMessage={setErrorMessage}
                            dispatchFrame={dispatchFrame}
                          />
                        </div>
                      </KonvaComponents>
                      {!isNaN(videoInfo.curFrame) && videoInfo.curFrame !== null && (
                        <Text
                          x={5 / stageScale}
                          y={5 / stageScale}
                          fontSize={20 / stageScale}
                          fill={"red"}
                          text={videoInfo.curFrame}
                          fontStyle={"bold"}
                        />
                      )}
                      {isCursor && inImage && paused && (
                        <>
                          <Line
                            points={[0, mouseY, defaultVideoSize.w, mouseY]}
                            stroke={"red"}
                            strokeWidth={1 / stageScale}
                            opacity={0.7}
                          />
                          <Line
                            points={[mouseX, 0, mouseX, defaultVideoSize.h]}
                            stroke={"red"}
                            strokeWidth={1 / stageScale}
                            opacity={0.7}
                          />
                        </>
                      )}
                      {videoLabel.btnSts !== "none" && curObject.CURSOR !== undefined && _switchCursor()}
                      {trackerList.length !== 0 && trackerList.map((ele, index) => _switchDraw(ele, index, true))}
                    </Layer>
                    <Layer ref={segLayerRef}>
                      {isPlay &&
                        objectList[videoInfo.curFrame]?.length !== 0 &&
                        objectList[videoInfo.curFrame]?.map((ele, index) => _switchDraw(ele, index))}
                    </Layer>
                  </Stage>
                </div>
              </div>
              <VideoControlBar
                isPredict={true}
                duration={duration}
                videoInfo={videoInfo}
                replay={replay}
                paused={paused}
                setReplay={setReplay}
                setPaused={setPaused}
                setControl={setControl}
                showFrame={true}
                tracker={props.dataSet.OBJECT_TYPE === "D"}
                frameBound={frameBound}
                setFrameBound={setFrameBound}
                _handleTracker={_handleTracker}
              />
            </>
          )}
        </LoadingOverlay>
      </Col>
    </Row>
  )
}

DrawVideo.propTypes = {}

export default React.memo(DrawVideo)
