import React, { useState, useRef, useEffect, useCallback } from "react"
import { Row, Col } from "reactstrap"
import { Stage, Layer, Line, Circle, Image } from "react-konva"
import { useSelector, useDispatch } from "react-redux"
import { cloneDeep } from "lodash-es"
import { toast } from "react-toastify"
import { FaMagic, FaPauseCircle } from "react-icons/fa"
import { MdError, MdInfoOutline, MdFindReplace } from "react-icons/md"
import { IoMdWalk } from "react-icons/io"

import { confirmAlert } from "react-confirm-alert"

import * as ImageAnnoApi from "Config/Services/ImageAnnoApi"
import * as ImageLabelActions from "Redux/Actions/ImageLabelActions"
import CursorRect from "./Drawers/CursorRect"
import DrawRect from "./Drawers/DrawRect"
import DrawBrush from "./Drawers/DrawBrush"
import LoadingOverlay from "react-loading-overlay"
import CursorPolygon from "./Drawers/CursorPolygon"
import DrawPolygon from "./Drawers/DrawPolygon"
import CommonToast from "../../../../../Components/Common/CommonToast"
import { _getNeedCount, _sampling } from "Components/Utils/Utils"
import CommonButton from "../../../../../Components/Common/CommonButton"
import useResizeListener from "../../../../../Components/Utils/useResizeListener"

function DrawArea(props) {
  const stageRef = useRef(null)
  const segLayerRef = useRef(null)
  const imageRef = useRef(null)

  const stageCanvasRef = useRef(null)
  const [canvasWidth, canvasHeight] = useResizeListener(stageCanvasRef)

  // konva Layout
  const [image, setImage] = useState(null)
  const [stageX, setStageX] = useState(0)
  const [stageY, setStageY] = useState(0)
  const [defaultStageX, setDefaultStageX] = useState(0)
  const [defaultStageY, setDefaultStageY] = useState(0)
  const [defaultScale, setDefaultScale] = useState(1)
  const [stageScale, setStageScale] = useState(1)
  const [mouseX, setMouseX] = useState(null)
  const [mouseY, setMouseY] = useState(null)

  // Draw
  const [objectList, setObjectList] = useState([]) // Rect 및 Ploygon 정보 list
  const [brushList, setBrushList] = useState([]) // Brush 정보 List
  const [curObject, setCurObject] = useState({}) // 현재 작업 중인 Object Info
  const [isDraggable, setIsDraggable] = useState(false)
  const [isDrawing, setIsDrawing] = useState(false)
  const [inImage, setInImage] = useState(false)
  const [isMultiLine, setIsMultiLine] = useState(false)
  const [clickObject, setClickObject] = useState([])
  const [isCursor, setIsCursor] = useState(true)

  const [isLoadError, setIsLoadError] = useState(false)

  const dispatch = useDispatch()
  const imageLabel = useSelector(
    state => state.imageLabel,
    (prevItem, item) => {
      if (item === prevItem) return true
      else return false
    }
  )

  // Tag가 변경됐을 때 현재 Object Color, Brush Color 와 Tag Color 비교
  useEffect(() => {
    let flag = false
    if (objectList.length !== 0) {
      const newObjectList = objectList.map(ele => {
        if (
          String(ele.TAG_CD) === String(imageLabel.curTag.TAG_CD) &&
          (ele.COLOR !== imageLabel.curTag.COLOR || ele.TAG_NAME !== imageLabel.curTag.NAME)
        ) {
          ele.COLOR = imageLabel.curTag.COLOR
          ele.TAG_NAME = imageLabel.curTag.NAME
          flag = true
          return ele
        } else {
          return ele
        }
      })
      setObjectList(newObjectList)
    }

    if (props.dataSet.OBJECT_TYPE === "S") {
      if (brushList.length !== 0) {
        const newBrushList = brushList.map(ele => {
          if (
            String(ele.TAG_CD) === String(imageLabel.curTag.TAG_CD) &&
            (ele.COLOR !== imageLabel.curTag.COLOR || ele.TAG_NAME !== imageLabel.curTag.NAME)
          ) {
            ele.COLOR = imageLabel.curTag.COLOR
            ele.TAG_NAME = imageLabel.curTag.TAG_NAME
            flag = true
            return ele
          } else {
            return ele
          }
        })
        setBrushList(newBrushList)
      }
    }
    if (flag) dispatch(ImageLabelActions._isDrawAction(true))
    _setObject()
  }, [imageLabel.curTag])

  // 버튼 상태 변경 useEffect
  useEffect(() => {
    switch (imageLabel.btnSts) {
      case "isBrush":
      case "isEraser":
        setIsCursor(false)
        setIsDraggable(false)
        stageRef.current.container().style.cursor = "none"
        _setObject()
        break
      default:
        setIsCursor(true)
        setIsDraggable(true)
        stageRef.current.container().style.cursor = "none"
        _setObject()
        break
    }
  }, [imageLabel.btnSts])

  //이미지 원래 원본크기로 되돌리기
  useEffect(() => {
    if (imageLabel.isResetImageSize) {
      setStageScale(defaultScale)
      setStageX(defaultStageX)
      setStageY(defaultStageY)
      dispatch(ImageLabelActions._isResetImageSize(false))
    }
  }, [imageLabel.isResetImageSize])

  // 이미지 or canvas 크기가 변경됐을 때 호출 x,y,scale 재계산
  useEffect(() => {
    if (image !== null) {
      let width = image.width
      let height = image.height
      if (width > canvasWidth || height > canvasHeight) {
        let scale, stageX, stageY

        // 원본 이미지 width, height 같은 경우
        // canvas 크기가 작은 쪽으로 scale 계산
        if (canvasWidth > canvasHeight) {
          scale = canvasHeight / height
          stageX = (canvasWidth - width * scale) / 2
          stageX = stageX < 0 ? 0 : stageX
          stageY = 0
        } else {
          scale = canvasWidth / width
          stageY = (canvasHeight - height * scale) / 2
          stageY = stageY < 0 ? 0 : stageY
          stageX = 0
        }
        if (!imageLabel.isZoomFix) {
          setStageX(stageX)
          setStageY(stageY)
          setStageScale(scale)
        }
        setDefaultScale(scale)
        setDefaultStageX(stageX)
        setDefaultStageY(stageY)
      } else {
        if (!imageLabel.isZoomFix) {
          setStageX((canvasWidth - width) / 2)
          setStageY((canvasHeight - height) / 2)
          setStageScale(1)
        }
        setDefaultScale(1)
        setDefaultStageX((canvasWidth - width) / 2)
        setDefaultStageY((canvasHeight - height) / 2)
      }
    }
  }, [image, canvasWidth, canvasHeight])

  // redux imageLabel.curImage 변경 시 이미지 로드
  useEffect(() => {
    if (Object.keys(imageLabel.curImage).length !== 0) {
      setImage(null)
      const imageInfo = imageLabel.curImage
      let img = new window.Image()
      // const ext = imageInfo.FILE_EXT.substring(1, imageInfo.FILE_EXT.length)
      // img.src = `data:image/${ext};base64, ` + base64Arraybuffer.encode(imageInfo.IMG_ORG.data)
      img.src = imageInfo.FILE_URL
      img.onload = () => {
        setImage(img)
        setIsLoadError(false)
      }
      img.onerror = () => {
        setIsLoadError(true)
      }
    }
  }, [imageLabel.curImage])

  // Redux objectList change useEffect
  useEffect(() => {
    setObjectList(imageLabel.objectList)
  }, [imageLabel.objectList])

  useEffect(() => {
    setBrushList(imageLabel.brushList)
  }, [imageLabel.brushList])

  // Segmentation 저장시에 Brush Mask Image Create
  useEffect(() => {
    if (imageLabel.saveImage) {
      // stage 이미지 크기로 초기화
      stageRef.current.setAttrs({
        x: 0,
        y: 0,
        scaleX: 1,
        scaleY: 1,
        width: image.width,
        height: image.height
      })
      // line만 그리기 위해서 Line이 아닌것은 hide
      segLayerRef.current.children.forEach(node => {
        if (node.getClassName() !== "Line" || node.name() === "delete") {
          node.hide()
        } else {
          // 투명도 1
          node.opacity(1)
          if (node.name() === "polygon") {
            node.setAttrs({ fill: node.attrs.stroke, prevFill: node.attrs.fill })
          }
        }
      })

      // image 영역만큼 layer 잘라서 dataUrl 생성
      const dataUrl = segLayerRef.current.toDataURL({
        x: 0,
        y: 0,
        width: image.width,
        height: image.height
      })

      // const file = _dataURLtoFile(dataUrl, "mask11.png")
      // fileDownload(file, "mask11.png")
      const maskImg = Buffer.from(dataUrl.split(",")[1], "base64")
      dispatch(ImageLabelActions._setMaskImg(maskImg))

      segLayerRef.current.children.forEach(node => {
        if (!node.isVisible()) {
          node.show()
        } else {
          if (node.name() === "polygon") {
            node.setAttrs({ fill: node.attrs.prevFill })
          } else if (node.name() === "brush") {
            node.opacity(imageLabel.opacityValue * 0.01)
          }
        }
      })

      // 원상 복구
      stageRef.current.setAttrs({
        x: stageX,
        y: stageY,
        scaleX: stageScale,
        scaleY: stageScale,
        width: canvasWidth,
        height: canvasHeight
      })
      segLayerRef.current.batchDraw()
    }
  }, [imageLabel.saveImage])

  const _setObject = useCallback(() => {
    let curObj = {
      DATASET_CD: imageLabel.curImage.DATASET_CD,
      DATA_CD: imageLabel.curImage.DATA_CD,
      TAG_CD: imageLabel.curTag.TAG_CD,
      TAG_NAME: imageLabel.curTag.NAME,
      CLASS_CD: imageLabel.curTag.CLASS_CD,
      COLOR: imageLabel.curTag.COLOR,
      CURSOR: imageLabel.btnSts,
      NEEDCOUNT: _getNeedCount(imageLabel.btnSts),
      POSITION: []
    }
    setCurObject(curObj)
  }, [imageLabel.curImage, imageLabel.curTag, imageLabel.btnSts])

  useEffect(() => {
    if (stageRef.current) {
      if (inImage) {
        stageRef.current.container().style.cursor = "none"
      } else {
        stageRef.current.container().style.cursor = "auto"
      }
    }
  }, [inImage])

  const _handleWheel = useCallback(
    e => {
      // get target component info
      e.evt.preventDefault()

      // Ctrl + Wheel brush mode cursor size change
      if (e.evt.ctrlKey && (imageLabel.btnSts === "isBrush" || imageLabel.btnSts === "isEraser")) {
        if (e.evt.deltaY < 0) {
          if (imageLabel.brushSize !== 30) {
            dispatch(ImageLabelActions._setBrushSize(Math.min(30, imageLabel.brushSize + 1)))
          }
        } else {
          if (imageLabel.brushSize !== 1) {
            dispatch(ImageLabelActions._setBrushSize(Math.max(0, imageLabel.brushSize - 1)))
          }
        }
        return
      }
      const stage = e.target.getStage()
      const oldScale = stage.scaleX()
      // set scale interval
      // jogoon 낮은 스케일 구간에서 스케일 변환값 변경
      const scaleBy = stage.scaleX() > 1 ? (stage.scaleX() >= 2 ? 1.4 : 1.1) : 1.2
      // move mouse pointer when scale has been changed
      const mousePointTo = {
        x: stage.getPointerPosition().x / oldScale - stage.x() / oldScale,
        y: stage.getPointerPosition().y / oldScale - stage.y() / oldScale
      }
      // check mouse wheel direction ( zoom in / out )
      let newScale = e.evt.deltaY < 0 ? oldScale * scaleBy : oldScale / scaleBy
      if (newScale > 3.5 || newScale < 0.1) {
        return
      }

      // for image movement
      let stageX = -(mousePointTo.x - stage.getPointerPosition().x / newScale) * newScale
      let stageY = -(mousePointTo.y - stage.getPointerPosition().y / newScale) * newScale

      setStageX(stageX)
      setStageY(stageY)
      setStageScale(newScale)
    },
    [imageLabel.btnSts, imageLabel.brushSize]
  )

  const _handleDragEnd = useCallback(
    e => {
      const stage = e.target.getStage()
      const mousePointTo = {
        x: stage.getPointerPosition().x / stageScale - stage.x() / stageScale,
        y: stage.getPointerPosition().y / stageScale - stage.y() / stageScale
      }

      let stageX = -(mousePointTo.x - stage.getPointerPosition().x / stageScale) * stageScale
      let stageY = -(mousePointTo.y - stage.getPointerPosition().y / stageScale) * stageScale
      setStageX(stageX)
      setStageY(stageY)

      const attrs = stage.getAttrs()
      const x = (stage.getPointerPosition().x - attrs.x) / stageScale
      const y = (stage.getPointerPosition().y - attrs.y) / stageScale

      setMouseX(x)
      setMouseY(y)
    },
    [stageScale]
  )

  const _handleSegMouseDown = useCallback(
    e => {
      if (imageLabel.btnSts === "isBrush" || imageLabel.btnSts === "isEraser") {
        if (imageLabel.curTag.TAG_CD === undefined) {
          dispatch(ImageLabelActions._openTag())
          return
        }
        setIsDrawing(true)
        const stage = e.target.getStage()
        /// 새로운 방법
        const attrs = stage.getAttrs()
        let pos = {
          x: (stage.getPointerPosition().x - attrs.x) / stageScale,
          y: (stage.getPointerPosition().y - attrs.y) / stageScale
        }

        let lastLine = {
          TAG_CD: imageLabel.curTag.TAG_CD,
          COLOR: imageLabel.curTag.COLOR,
          LINE_WIDTH: imageLabel.brushSize * 2,
          MODE: imageLabel.btnSts === "isBrush" ? "source-over" : "destination-out",
          POINTS: [pos.x, pos.y]
        }

        let cBrushList = cloneDeep(brushList)
        cBrushList.push(lastLine)
        setBrushList(cBrushList)
      }
    },
    [imageLabel.btnSts, brushList, stageScale, imageLabel.curTag, imageLabel.brushSize]
  )

  const _handleSegMouseUp = useCallback(() => {
    if (isDrawing && (imageLabel.btnSts === "isBrush" || imageLabel.btnSts === "isEraser")) {
      dispatch(ImageLabelActions._setBrushList(brushList))
      dispatch(ImageLabelActions._isDrawAction(true))
    }
    setIsDrawing(false)
  }, [isDrawing, imageLabel.btnSts, brushList])

  const _handleSegMouseMove = useCallback(
    e => {
      if (isDrawing && (imageLabel.btnSts === "isBrush" || imageLabel.btnSts === "isEraser")) {
        const stage = e.target.getStage()
        const attrs = stage.getAttrs()
        let pos = stage.getPointerPosition()
        let movePos = {
          x: (pos.x - attrs.x) / stageScale,
          y: (pos.y - attrs.y) / stageScale
        }

        let cBrushList = cloneDeep(brushList)
        let lastLine = cBrushList[cBrushList.length - 1]
        let points = lastLine.POINTS.concat([movePos.x, movePos.y])
        lastLine.POINTS = points
        cBrushList[cBrushList.length - 1] = lastLine
        setBrushList(cBrushList)
      }
    },
    [isDrawing, imageLabel.btnSts, brushList, stageScale]
  )

  const _handleMouseMove = useCallback(
    e => {
      const stage = e.currentTarget
      const attrs = stage.getAttrs()
      const x = (stage.getPointerPosition().x - attrs.x) / stageScale
      const y = (stage.getPointerPosition().y - attrs.y) / stageScale

      setMouseX(x)
      setMouseY(y)

      switch (imageLabel.btnSts) {
        case "isBrush":
        case "isEraser":
          _handleSegMouseMove(e)
          break
        default:
          const imageWidth = image.width
          const imageHeight = image.height
          if (x < 0 || y < 0 || x > imageWidth || y > imageHeight) {
            setInImage(false)
          } else {
            setInImage(true)
          }
          break
      }
    },
    [stageScale, imageLabel.btnSts, image, _handleSegMouseMove]
  )

  const _handleMouseEnter = useCallback(
    e => {
      const stage = e.currentTarget
      switch (imageLabel.btnSts) {
        case "isBrush":
        case "isEraser":
          stage.container().style.cursor = "none"
          break
        default:
          const imageWidth = image.width
          const imageHeight = image.height
          if (mouseX < 0 || mouseY < 0 || mouseX > imageWidth || mouseY > imageHeight) {
            setInImage(false)
            stage.container().style.cursor = "auto"
          } else {
            setInImage(true)
            stage.container().style.cursor = "none"
          }
          setIsCursor(true)
          break
      }
    },
    [imageLabel.btnSts, mouseX, mouseY, image]
  )

  const _handleMouseLeave = useCallback(e => {
    setIsDrawing(false)
    setInImage(false)
    setIsCursor(false)
    e.target.getStage().container().style.cursor = "auto"
  }, [])

  const _drawAction = (x, y) => {
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
      if (imageLabel.btnSts === "isMagic") {
        // Magic tool draw
        _setObject()
        dispatch(ImageLabelActions._setImageStatus(true))
        const param = {
          DATASET_CD: imageLabel.curImage.DATASET_CD,
          DATA_CD: imageLabel.curImage.DATA_CD,
          TAG_CD: imageLabel.curTag.TAG_CD,
          COLOR: imageLabel.curTag.COLOR,
          CLASS_CD: imageLabel.curTag.CLASS_CD,
          OBJECT_TYPE: props.dataSet.OBJECT_TYPE,
          RECT: cCurObject.POSITION,
          IS_READY: true,
          AI_CD: imageLabel.curTag.BASE_MDL
        }

        if (imageLabel.curTag.CLASS_CD === null || imageLabel.curTag.CLASS_CD === "") {
          dispatch(ImageLabelActions._setImageStatus(false))
          toast.error(<CommonToast Icon={MdError} text={"Current tag is unpredictable"} />)
          return
        }

        ImageAnnoApi._getImagePredict(param)
          .then(data => {
            if (data[0]?.status) {
              if (data[0].status === 2) {
                // model full
                toast.error(<CommonToast Icon={MdInfoOutline} text={"The maximum number of processes has been exceeded"} />)
                dispatch(ImageLabelActions._setImageStatus(false))
                dispatch(ImageLabelActions._statusModal(true))
                // _openStatusModal()
                return
              } else if (data[0].status === 3) {
                // model 안 올라옴 다시 올리기
                toast.error(<CommonToast Icon={MdInfoOutline} text={"The model is not running"} />)
                dispatch(ImageLabelActions._setImageStatus(false))
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
                              dispatch(ImageLabelActions._setImageStatus(true))
                              param.IS_READY = false
                              onClose()
                              ImageAnnoApi._getImagePredict(param)
                                .then(() => {
                                  dispatch(ImageLabelActions._checkModel(true))
                                  dispatch(ImageLabelActions._setImageStatus(false))
                                  toast.info(<CommonToast Icon={MdFindReplace} text={"Model Start!!"} />)
                                })
                                .catch(e => {
                                  dispatch(ImageLabelActions._checkModel(true))
                                  dispatch(ImageLabelActions._setImageStatus(false))
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

            if (data[0].ANNO_DATA.length !== 0) {
              const arr = data[0].ANNO_DATA.map(ele => {
                let obj = {}
                obj.DATASET_CD = imageLabel.curImage.DATASET_CD
                obj.DATA_CD = imageLabel.curImage.DATA_CD
                obj.CLASS_CD = imageLabel.curTag.CLASS_CD
                obj.TAG_CD = imageLabel.curTag.TAG_CD
                obj.COLOR = imageLabel.curTag.COLOR
                obj.TAG_NAME = imageLabel.curTag.NAME
                obj.CURSOR = ele.CURSOR
                obj.NEEDCOUNT = _getNeedCount(obj.CURSOR)
                obj.POSITION = _sampling(ele.POSITION)
                return obj
              })
              let cObjectList = cloneDeep(objectList)
              cObjectList = [...cObjectList, ...arr]

              dispatch(ImageLabelActions._setObjectList(cObjectList))
              toast.info(<CommonToast Icon={FaMagic} text={"Magic Predict Success"} />)
            } else {
              toast.error(<CommonToast Icon={MdInfoOutline} text={"Magic Predict No result"} />)
            }
            dispatch(ImageLabelActions._setImageStatus(false))
          })
          .catch(e => {
            dispatch(ImageLabelActions._setImageStatus(false))
            console.log(e)
            toast.error(<CommonToast Icon={MdError} text={"Magic Predict Fail"} />)
          })
      } else {
        let checkX = true
        let checkY = true
        cCurObject.POSITION.forEach(pos => {
          if (pos.X < 0) {
            pos.X = 0
          } else if (pos.X > image.width) {
            pos.X = image.width
          } else {
            checkX = false
          }

          if (pos.Y < 0) {
            pos.Y = 0
          } else if (pos.Y > image.height) {
            pos.Y = image.height
          } else {
            checkY = false
          }
        })

        if (checkX || checkY) {
          _setObject()
          return
        }
        if (cCurObject.CURSOR === "isRect") {
          getRectPoint(cCurObject)
        }
        let cObjectList = cloneDeep(objectList)
        cObjectList.push(cCurObject)
        dispatch(ImageLabelActions._setObjectList(cObjectList))
        _setObject()
      }
    }
  }

  const getRectPoint = obj => {
    const pos = obj.POSITION
    const hx = pos[0].X >= pos[1].X ? pos[0].X : pos[1].X
    const hy = pos[0].Y >= pos[1].Y ? pos[0].Y : pos[1].Y
    const lx = pos[0].X < pos[1].X ? pos[0].X : pos[1].X
    const ly = pos[0].Y < pos[1].Y ? pos[0].Y : pos[1].Y
    obj.POSITION = [
      { X: lx, Y: ly },
      { X: hx, Y: hy }
    ]
  }

  const _handleMouseClick = e => {
    e.evt.preventDefault()

    const stage = e.currentTarget
    const attrs = stage.getAttrs()
    // get mouse position
    const x = (stage.getPointerPosition().x - attrs.x) / stageScale
    const y = (stage.getPointerPosition().y - attrs.y) / stageScale
    // const imageWidth = image.width
    // const imageHeight = image.height

    // if (x < 0 || y < 0 || x > imageWidth || y > imageHeight) {
    //   return
    // }
    switch (imageLabel.btnSts) {
      case "isRect":
      case "isPolygon":
      case "isMagic":
        if (imageLabel.curTag.TAG_CD === undefined) {
          dispatch(ImageLabelActions._openTag())
        } else {
          dispatch(ImageLabelActions._isDrawAction(true))
          _drawAction(x, y)
        }
        break
      default:
        break
    }
  }

  const _setClickObject = useCallback(
    id => {
      // console.log(id, clickObject)
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
      if (mode === "all") {
        cObjectList[idx].POSITION = pos
      } else if (mode === "add") {
        let position = cObjectList[idx].POSITION
        position.splice(posIdx, 0, pos)
        cObjectList[idx].POSITION = position
      } else {
        cObjectList[idx].POSITION[posIdx] = pos
      }
      dispatch(ImageLabelActions._setObjectList(cObjectList))
      dispatch(ImageLabelActions._isDrawAction(true))
      return true
    },
    [objectList]
  )

  const _deleteObject = useCallback(
    (id, mode) => {
      let cObjectList = cloneDeep(objectList)
      if (mode === "point") {
        const ids = id.split("_")
        const idx = Number(ids[0])
        const posIdx = Number(ids[1])
        let points = cObjectList[idx].POSITION
        let filter = points.filter((ele, index) => index !== posIdx)
        cObjectList[idx].POSITION = filter
      } else {
        if (clickObject.length !== 0) {
          let filter = objectList.filter((ele, index) => !clickObject.some(clicked => clicked === index))
          cObjectList = filter
          setClickObject([])
        }
      }
      dispatch(ImageLabelActions._setObjectList(cObjectList))
      dispatch(ImageLabelActions._isDrawAction(true))
    },
    [objectList, clickObject]
  )

  const _switchCursor = () => {
    const props = {
      mouseX: mouseX,
      mouseY: mouseY,
      color: curObject.COLOR,
      needCount: curObject.NEEDCOUNT,
      position: curObject.POSITION,
      scale: stageScale,
      polygonClick: polygonClick
    }

    switch (imageLabel.btnSts) {
      case "isRect":
      case "isMagic":
        return <CursorRect {...props} />
      case "isPolygon":
        return <CursorPolygon {...props} />
      default:
        return null
    }
  }

  const _switchDraw = (ele, index) => {
    let isMove = false
    switch (imageLabel.btnSts) {
      case "isBrush":
      case "isEraser":
        isMove = false
        break
      default:
        isMove = true
        break
    }
    const filter = imageLabel?.tagList?.filter(tag => Number(tag.TAG_CD) === Number(ele.TAG_CD))
    const isShow = filter[0]?.isShow
    const props = {
      key: index,
      id: index,
      cursor: ele.CURSOR,
      color: ele.COLOR,
      position: ele.POSITION,
      scale: stageScale,
      isMove: isMove,
      imageWidth: image.width,
      imageHeight: image.height,
      _changePos: _changePos,
      _deleteObject: _deleteObject,
      _setClickObject: _setClickObject,
      btnSts: imageLabel.btnSts,
      tagName: ele.TAG_NAME,
      tagCd: ele.TAG_CD,
      curTagCd: imageLabel.curTag.TAG_CD,
      isShow: isShow,
      acc: ele?.ACCURACY,
      setIsCursor: setIsCursor
    }
    switch (ele.CURSOR) {
      case "isRect":
        return <DrawRect {...props} />
      case "isPolygon":
        return <DrawPolygon {...props} />
      default:
        return null
    }
  }

  // const _dataURLtoFile = (dataurl, filename) => {
  //   const arr = dataurl.split(",")
  //   const mime = arr[0].match(/:(.*?);/)[1]
  //   const bstr = atob(arr[1])
  //   let n = bstr.length
  //   let u8arr = new Uint8Array(n)
  //   while (n--) {
  //     u8arr[n] = bstr.charCodeAt(n)
  //   }
  //   return new File([u8arr], filename, { type: mime })
  // }

  const polygonClick = e => {
    if (Number(e.target.id().split("_")[1]) === 0) {
      if (isMultiLine && imageLabel.btnSts === "isPolygon") {
        let cObjectList = cloneDeep(objectList)
        let cCurObject = cloneDeep(curObject)
        let checkX = true
        let checkY = true
        cCurObject.POSITION.forEach(pos => {
          if (pos.X < 0) {
            pos.X = 0
          } else if (pos.X > image.width) {
            pos.X = image.width3
          } else {
            checkX = false
          }

          if (pos.Y < 0) {
            pos.Y = 0
          } else if (pos.Y > image.height) {
            pos.Y = image.height
          } else {
            checkY = false
          }
        })

        if (checkX || checkY) {
          _setObject()
          return
        }

        cObjectList.push(cCurObject)
        dispatch(ImageLabelActions._setObjectList(cObjectList))
        setIsMultiLine(false)
        _setObject()
      }
    }
  }

  // key event
  const _handleKeyDown = useCallback(
    e => {
      if (imageLabel.modalCheck) {
        return
      }
      switch (e.keyCode) {
        case 32:
          // space key
          if (isMultiLine && imageLabel.btnSts === "isPolygon") {
            let cObjectList = cloneDeep(objectList)
            let cCurObject = cloneDeep(curObject)
            let checkX = true
            let checkY = true
            cCurObject.POSITION.forEach(pos => {
              if (pos.X < 0) {
                pos.X = 0
              } else if (pos.X > image.width) {
                pos.X = image.width
              } else {
                checkX = false
              }

              if (pos.Y < 0) {
                pos.Y = 0
              } else if (pos.Y > image.height) {
                pos.Y = image.height
              } else {
                checkY = false
              }
            })

            if (checkX || checkY) {
              _setObject()
              return
            }

            cObjectList.push(cCurObject)
            dispatch(ImageLabelActions._setObjectList(cObjectList))
            setIsMultiLine(false)
            _setObject()
          }
          break
        case 46:
          // delete key
          _deleteObject()
          break
        case 27:
          // esc key
          _setObject()
          break
        default:
          break
      }
    },
    [curObject, isMultiLine, imageLabel.btnSts, imageLabel.modalCheck, objectList, image, _setObject, _deleteObject, dispatch]
  )

  // KeyDown useEffect
  useEffect(() => {
    window.addEventListener("keydown", _handleKeyDown)
    return () => {
      window.removeEventListener("keydown", _handleKeyDown)
    }
  }, [_handleKeyDown])

  return (
    <Row noGutters className="h-100">
      <Col md={12} className="h-100">
        <div className="w-100 h-100" style={{ background: "black" }} ref={stageCanvasRef}>
          <LoadingOverlay
            className="w-100 h-100"
            active={imageLabel.preStatus || (!image && !isLoadError) || imageLabel.imageStatus}
            spinner
            text="Loading ..."
          >
            {isLoadError ? (
              <div className="text-center w-100" style={{ position: "absolute", top: "50%" }}>
                <MdError className="mr-1 icon-error" />
                Image Load Fail
              </div>
            ) : (
              <Stage
                ref={stageRef}
                width={canvasWidth}
                height={canvasHeight}
                x={stageX}
                y={stageY}
                scaleX={stageScale}
                scaleY={stageScale}
                onWheel={_handleWheel}
                // onDblClick={this._handleMouseDoubleClick}
                onClick={_handleMouseClick}
                onMouseEnter={_handleMouseEnter}
                onMouseLeave={_handleMouseLeave}
                onDragEnd={_handleDragEnd}
                onMouseDown={_handleSegMouseDown}
                onMouseUp={_handleSegMouseUp}
                onMouseMove={_handleMouseMove}
                //onDragMove={_handleMouseMove}
                draggable={isDraggable}
              >
                <Layer>
                  <Image image={image} ref={imageRef} />
                  {isCursor && inImage && (
                    <>
                      <Line points={[0, mouseY, image.width, mouseY]} stroke={"red"} strokeWidth={1 / stageScale} opacity={0.7} />
                      <Line points={[mouseX, 0, mouseX, image.height]} stroke={"red"} strokeWidth={1 / stageScale} opacity={0.7} />
                    </>
                  )}
                  {(imageLabel.btnSts === "isBrush" || imageLabel.btnSts === "isEraser") && (
                    <Circle
                      x={mouseX}
                      y={mouseY}
                      radius={imageLabel.brushSize}
                      strokeWidth={1}
                      stroke={imageLabel.curTag.COLOR}
                      fill={imageLabel.curTag.COLOR}
                      opacity={0.5}
                    />
                  )}
                  {imageLabel.btnSts !== "none" && curObject.CURSOR !== undefined && _switchCursor()}
                </Layer>
                <Layer ref={segLayerRef}>
                  {image && brushList !== undefined && brushList.length !== 0 && (
                    <DrawBrush brushList={brushList} opacity={imageLabel.opacityValue} tagList={imageLabel.tagList} />
                  )}
                  {image && objectList !== undefined && objectList.length !== 0 && objectList.map((ele, index) => _switchDraw(ele, index))}
                </Layer>
              </Stage>
            )}
          </LoadingOverlay>
        </div>
      </Col>
    </Row>
  )
}

DrawArea.propTypes = {}

export default React.memo(DrawArea)
