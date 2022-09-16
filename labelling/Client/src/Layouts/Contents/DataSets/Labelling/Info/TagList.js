import React, { useState, useMemo, useEffect, useCallback, useRef } from "react"
import PropTypes from "prop-types"
import LoadingOverlay from "react-loading-overlay"
import { useDispatch, useSelector } from "react-redux"
import { toast } from "react-toastify"
import { confirmAlert } from "react-confirm-alert"
import hash from "object-hash"
import { cloneDeep, uniqBy } from "lodash-es"

import { BsFillCircleFill, BsTrash } from "react-icons/bs"
import { GoPlus, GoDashboard } from "react-icons/go"
import { MdBackspace, MdError, MdFindReplace, MdInfoOutline, MdDeleteForever } from "react-icons/md"
import { FaRunning, FaRegPlayCircle, FaRegPauseCircle } from "react-icons/fa"
import { RiDeleteBin3Line, RiPriceTag3Line } from "react-icons/ri"
import { IoIosArrowDropright, IoMdWalk } from "react-icons/io"
import ReactTooltip from "react-tooltip"

import VirtualTable from "Components/Common/VirtualTable"
import CommonButton from "../../../../../Components/Common/CommonButton"
import CommonToast from "../../../../../Components/Common/CommonToast"
import TagModal from "./TagModal"
import StatusModal from "./StatusModal"

import * as ImageAnnoApi from "Config/Services/ImageAnnoApi"
import * as ImageLabelActions from "Redux/Actions/ImageLabelActions"
import * as VideoAnnoApi from "Config/Services/VideoAnnoApi"
import * as VideoLabelActions from "Redux/Actions/VideoLabelActions"

import { _getNeedCount } from "Components/Utils/Utils"

function TagList(props) {
  const [isAddModal, setIsAddModal] = useState(false)
  const [isStatusModal, setIsStatusModal] = useState(false)
  const [mode, setMode] = useState()
  const [curTagIndex, setCurTagIndex] = useState(null)
  const [tagList, setTagList] = useState([])
  const [tagInfo, setTagInfo] = useState(null)
  const [hoverIndex, setHoverIndex] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [activeModel, setActiveModel] = useState([])
  const [updateIndex, setUpdateIndex] = useState(null)
  const confirmText = useRef(null)

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
  const api = useMemo(() => (props.dataType === "I" ? ImageAnnoApi : VideoAnnoApi), [props.dataType])
  const label = useMemo(() => (props.dataType === "I" ? imageLabel : videoLabel), [props.dataType, imageLabel, videoLabel])
  const action = useMemo(() => (props.dataType === "I" ? ImageLabelActions : VideoLabelActions), [props.dataType])

  // Tag List 불러오기
  useEffect(() => {
    if (props.dataSetCd) {
      let param = { DATASET_CD: props.dataSetCd, OBJECT_TYPE: props.objectType }

      setIsLoading(true)
      api
        ._getDataTags(param)
        .then(data => {
          data.map(ele => (ele.isShow = true))
          setTagList(data)
          setIsLoading(false)
          if (data.length !== 0) {
            setCurTagIndex(0)
          }
          props._setRenderLazy(true)
        })
        .catch(e => {
          setIsLoading(false)
          props._setRenderLazy(true)
          console.log(e)
        })
      _getActiveStatus()
    }
  }, [props.dataSetCd])

  useEffect(() => {
    dispatch(action._setTagList(tagList))
  }, [tagList])

  useEffect(() => {
    if (curTagIndex !== null) {
      dispatch(action._setCurTag(tagList[curTagIndex]))
    } else {
      dispatch(action._setCurTag({}))
    }
  }, [curTagIndex])

  useEffect(() => {
    if (label.openTag === true) {
      _openAddModal()
      dispatch(action._openTag())
    }
  }, [label.openTag])

  useEffect(() => {
    ReactTooltip.rebuild()
  }, [tagList, label, activeModel])

  // 모델 상태 삭제 modal useEffect
  useEffect(() => {
    if (label.statusModal) {
      _openStatusModal()
      dispatch(action._statusModal(false))
    }
  }, [label.statusModal])

  // 모델 상태 체크 useEffect
  useEffect(() => {
    if (label.statusModal) {
      _getActiveStatus()
      dispatch(action._checkModel(false))
    }
  }, [label.checkModel])

  const _handleKeyDown = useCallback(
    e => {
      if (label.modalCheck) {
        return
      }
      switch (e.keyCode) {
        case 87:
          if (curTagIndex !== null && curTagIndex !== 0) {
            setCurTagIndex(curTagIndex => curTagIndex - 1)
          } else if (curTagIndex !== null && curTagIndex === 0) setCurTagIndex(tagList.length - 1)
          break
        case 83:
          if (curTagIndex !== null && curTagIndex + 1 < tagList.length) {
            setCurTagIndex(curTagIndex => curTagIndex + 1)
          } else if (curTagIndex + 1 === tagList.length) {
            setCurTagIndex(0)
          }
          break
        default:
          break
      }
    },
    [curTagIndex, tagList.length, label.modalCheck]
  )

  useEffect(() => {
    window.addEventListener("keydown", _handleKeyDown)
    return () => {
      window.removeEventListener("keydown", _handleKeyDown)
    }
  }, [_handleKeyDown])

  const _tagAddModal = useCallback(() => {
    setIsAddModal(isAddModal => !isAddModal)
    dispatch(action._setModalCheck(!label.modalCheck))
  }, [label.modalCheck, action, dispatch])

  const _openAddModal = useCallback(() => {
    setMode("I")
    setTagInfo(null)
    setCurTagIndex(null)
    setUpdateIndex(null)
    _tagAddModal()
  }, [_tagAddModal])

  const _openStatusModal = useCallback(() => {
    setIsStatusModal(isStatusModal => !isStatusModal)
    dispatch(action._setModalCheck(!label.modalCheck))
  }, [label.modalCheck, action, dispatch])

  const _onRowClick = useCallback(({ index }) => {
    // if (curTagIndex === index) {
    //   setCurTagIndex(null)
    // } else {
    //   setCurTagIndex(index)
    // }
    setCurTagIndex(index)
  }, [])

  const _onRowDoubleClick = useCallback(
    ({ index }) => {
      setIsLoading(true)
      const param = { DATASET_CD: props.dataSetCd, NAME: tagList[index].NAME, OBJECT_TYPE: props.objectType }
      api
        ._getDataTags(param)
        .then(data => {
          setMode("U")
          setTagInfo(data[0])
          setUpdateIndex(index)
          _tagAddModal()
          setIsLoading(false)
        })
        .catch(e => {
          setUpdateIndex(null)
          setIsLoading(false)
          console.log(e)
        })
    },
    [props.dataSetCd, tagList, props.objectType, _tagAddModal, api]
  )

  const _rowStyle = useCallback(
    ({ index }) => {
      if (hoverIndex === index) {
        return { backgroundColor: "#2f2f2f" }
      }
      if (curTagIndex === index) {
        return { backgroundColor: "#015aa7af", color: "white" }
      }
      return
    },
    [hoverIndex, curTagIndex]
  )

  const _onRowMouseOver = useCallback(({ index }) => {
    setHoverIndex(index)
  }, [])

  const _onRowMouseOut = useCallback(() => {
    setHoverIndex(null)
  }, [])

  const _removeTag = useCallback(
    index => () => {
      confirmAlert({
        customUI: ({ onClose }) => {
          return (
            <div className="react-confirm-alert-custom">
              <h1>
                <BsTrash />
                Delete Tag
              </h1>
              {/* message: "Please type [" + data.TITLE + "] to avoid unexpected action.", */}
              <div className="custom-modal-body">
                <div className="text-warning">Warning. This action is irreversible.</div>
                <div className="explain">
                  <strong>{tagList[index].NAME}</strong> tag will be removed from all images.
                </div>
                <div>
                  Please type <strong>[ {tagList[index].NAME} ]</strong> to avoid unexpected action.
                </div>
                <input
                  type="text"
                  className="react-confirm-alert-input"
                  onKeyDown={e => {
                    e.stopPropagation()
                  }}
                  onChange={e => {
                    confirmText.current = e.target.value
                  }}
                />
              </div>
              <div className="custom-buttons">
                <CommonButton
                  className="bg-green"
                  text="Yes"
                  onClick={() => {
                    if (String(confirmText.current).trim() === tagList[index].NAME.trim()) {
                      const param = { DATASET_CD: tagList[index].DATASET_CD, NAME: tagList[index].NAME }
                      api
                        ._removeDataTag(param)
                        .then(data => {
                          if (data.status === 1) {
                            const cTagList = cloneDeep(tagList)
                            cTagList.splice(index, 1)
                            const objectList = cloneDeep(label.objectList)
                            const filter = objectList.filter(ele => cTagList.some(tag => String(tag.TAG_CD) === String(ele.TAG_CD)))
                            dispatch(action._setObjectList(filter))

                            if (props.objectType === "S") {
                              const cBrushList = cloneDeep(label.brushList)
                              const brushFilter = cBrushList.filter(ele => cTagList.some(tag => String(tag.TAG_CD) === String(ele.TAG_CD)))
                              dispatch(action._setBrushList(brushFilter))
                            }

                            setTagList(cTagList)
                            setCurTagIndex(null)
                            toast.info(<CommonToast Icon={MdDeleteForever} text={"Tag Delete Success"} />)
                            onClose()
                          } else {
                            throw { err: "status 0" }
                          }
                        })
                        .catch(e => {
                          toast.error(<CommonToast Icon={MdError} text={"Tag Delete Fail"} />)
                          console.log(e)
                        })
                    } else alert("Not matched.")
                  }}
                />
                <CommonButton className="bg-red" text="No" onClick={onClose} />
              </div>
            </div>
          )
        }
      })
    },
    [tagList, label.brushList, label.objectList, props.objectType, action, api, dispatch]
  )

  const _successModal = useCallback(
    tagIndex => {
      const param = { DATASET_CD: props.dataSetCd, OBJECT_TYPE: props.objectType }
      api
        ._getDataTags(param)
        .then(data => {
          data.forEach(ele => {
            let isShow = tagList.filter(tag => Number(tag.TAG_CD) === Number(ele.TAG_CD))[0]?.isShow
            if (isShow !== undefined) ele.isShow = isShow
            else ele.isShow = true
          })
          setTagList(data)
          if (tagIndex !== undefined) {
            dispatch(action._setCurTag(data[tagIndex]))
          } else {
            setCurTagIndex(data.length - 1)
          }
          setUpdateIndex(null)
          _tagAddModal()
        })
        .catch(e => {
          setUpdateIndex(null)
          console.log(e)
        })
    },
    [tagList, props.dataSetCd, props.objectType, _tagAddModal, action, api, dispatch]
  )

  const _tagShow = useCallback(
    rowIndex => () => {
      let cTagList = cloneDeep(tagList)
      cTagList[rowIndex].isShow = !cTagList[rowIndex].isShow
      setTagList(cTagList)
    },
    [tagList]
  )

  const _getActiveStatus = useCallback(() => {
    setIsLoading(true)
    api
      ._getActivePredictor(null)
      .then(data => {
        if (data?.status !== 0) {
          setActiveModel(data)
        }
        setIsLoading(false)
      })
      .catch(e => {
        setIsLoading(false)
        console.log(e)
      })
  }, [api])

  /**
   * Run MiniPredictor
   */
  const _runPreTrain = useCallback(
    (rowData, rowIndex) => async () => {
      try {
        dispatch(action._setPreStatus(true))
        setIsLoading(true)
        const param = {
          DATASET_CD: rowData.DATASET_CD,
          DATA_CD: props.dataType === "I" ? label.curImage.DATA_CD : label.curVideo.DATA_CD,
          CLASS_CD: rowData.CLASS_CD,
          COLOR: rowData.COLOR,
          OBJECT_TYPE: props.objectType,
          IS_READY: true,
          AI_CD: rowData.BASE_MDL
        }
        // 이미지 or 동영상 일 때 호출 API 분기
        let data
        if (props.dataType === "I") {
          data = await api._getImagePredict(param)
        }
        if (props.dataType === "V") {
          param.START_FRAME = label.curFrame
          param.END_FRAME = label.curFrame + label.frameBound
          data = await api._getPredictResult(param)
        }

        // python Model 상태 체크
        if (data[0]?.status) {
          if (data[0].status === 2) {
            // model full
            toast.error(<CommonToast Icon={MdInfoOutline} text={"The maximum number of processes has been exceeded"} />)
            setIsLoading(false)
            dispatch(action._setPreStatus(false))
            _onRowClick({ index: rowIndex })
            _openStatusModal()
            return
          } else if (data[0].status === 3) {
            // model not running
            toast.error(<CommonToast Icon={MdInfoOutline} text={"The model is not running"} />)
            setIsLoading(false)
            dispatch(action._setPreStatus(false))
            confirmAlert({
              customUI: ({ onClose }) => {
                return (
                  <div className="react-confirm-alert-custom">
                    <h1>
                      <FaRegPauseCircle />
                      The model is not running
                    </h1>
                    <p className="taglist">
                      Start Mini Predict Model
                      <IoMdWalk />
                    </p>
                    <div className="custom-buttons">
                      <CommonButton
                        className="bg-green"
                        text="Apply"
                        onClick={() => {
                          onClose()
                          _runStartModel()
                        }}
                      />
                      <CommonButton
                        className="bg-red"
                        text="Cancel"
                        onClick={() => {
                          _onRowClick({ index: rowIndex })
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
          // hash값을 사용 predict 결과 중복 체크
          const uniqList = data[0]?.ANNO_DATA?.map((ele, index) => {
            let obj = {}
            obj.hash = hash(ele)
            obj.index = index
            return obj
          })
          const u = uniqBy(uniqList, e => e.hash)
          const filterList = data[0]?.ANNO_DATA.filter((ele, i) => u.some(uniq => uniq.index === i))

          // predictor 결과 저장
          let cObjectList = cloneDeep(label.objectList)
          if (props.dataType === "I") {
            filterList.forEach(ele => {
              let obj = {}
              obj.DATASET_CD = rowData.DATASET_CD
              obj.DATA_CD = label.curImage.DATA_CD
              obj.CLASS_CD = rowData.CLASS_CD
              obj.TAG_CD = rowData.TAG_CD
              obj.COLOR = rowData.COLOR
              obj.TAG_NAME = rowData.NAME
              obj.CURSOR = props.objectType === "D" ? "isRect" : "isPolygon"
              obj.NEEDCOUNT = _getNeedCount(obj.CURSOR)
              obj.POSITION = ele.POSITION
              cObjectList.push(obj)
            })
          } else if (props.dataType === "V") {
            filterList.forEach(ele => {
              let obj = {}
              obj.DATASET_CD = rowData.DATASET_CD
              obj.DATA_CD = label.curVideo.DATA_CD
              obj.CLASS_CD = rowData.CLASS_CD
              obj.TAG_CD = rowData.TAG_CD
              obj.COLOR = rowData.COLOR
              obj.TAG_NAME = rowData.NAME
              obj.CURSOR = props.objectType === "D" ? "isRect" : "isPolygon"
              obj.NEEDCOUNT = _getNeedCount(obj.CURSOR)
              obj.POSITION = ele.POSITION
              cObjectList[ele.FRAME_NUMBER]?.push(obj)
            })
          }
          dispatch(action._setObjectList(cObjectList))
          dispatch(action._isDrawAction(true))
          toast.info(<CommonToast Icon={MdFindReplace} text={"Tag Predict Success"} />)
        } else {
          toast.error(<CommonToast Icon={MdInfoOutline} text={"Tag Predict No result"} />)
        }

        _getActiveStatus()
        setIsLoading(false)
        dispatch(action._setPreStatus(false))
        _onRowClick({ index: rowIndex })
      } catch (e) {
        console.log(e)
        _getActiveStatus()
        setIsLoading(false)
        dispatch(action._setPreStatus(false))
        _onRowClick({ index: rowIndex })
        toast.error(<CommonToast Icon={MdError} text={"Tag Predict Fail"} />)
      }
    },
    [
      label?.curFrame,
      label?.curVideo,
      label?.curImage,
      label.objectList,
      label?.frameBound,
      props.objectType,
      props.dataType,
      _getActiveStatus,
      _onRowClick,
      _runStartModel,
      _openStatusModal,
      action,
      api,
      dispatch
    ]
  )

  const _runStartModel = useCallback(
    async (rowData, rowIndex) => {
      dispatch(action._setPreStatus(true))
      setIsLoading(true)
      try {
        let data
        const param = {
          DATASET_CD: rowData.DATASET_CD,
          DATA_CD: props.dataType === "I" ? label.curImage.DATA_CD : label.curVideo.DATA_CD,
          CLASS_CD: rowData.CLASS_CD,
          COLOR: rowData.COLOR,
          OBJECT_TYPE: props.objectType,
          IS_READY: false,
          AI_CD: rowData.BASE_MDL
        }
        if (props.dataType === "I") {
          data = await api._getImagePredict(param)
        }
        if (props.dataType === "V") {
          param.START_FRAME = 1
          param.END_FRAME = 1
          data = await api._getPredictResult(param)
        }

        if (data[0]?.status) {
          if (data[0].status === 2) {
            // model full
            toast.error(<CommonToast Icon={MdInfoOutline} text={"The maximum number of processes has been exceeded"} />)
            setIsLoading(false)
            dispatch(action._setPreStatus(false))
            _onRowClick({ index: rowIndex })
            _openStatusModal()
            return
          }
        }
        _getActiveStatus()
        setIsLoading(false)
        dispatch(action._setPreStatus(false))
        _onRowClick({ index: rowIndex })
        toast.info(<CommonToast Icon={MdFindReplace} text={"Model Start!!"} />)
      } catch (e) {
        _getActiveStatus()
        setIsLoading(false)
        dispatch(action._setPreStatus(false))
        _onRowClick({ index: rowIndex })
        console.log(e)
        toast.error(<CommonToast Icon={MdInfoOutline} text={"Model Start Fail"} />)
      }
    },
    [
      label?.curImage,
      label?.curVideo,
      props.objectType,
      props.dataType,
      _openStatusModal,
      _getActiveStatus,
      _onRowClick,
      action,
      api,
      dispatch
    ]
  )

  const _startModel = useCallback(
    (rowData, rowIndex) => () => {
      confirmAlert({
        customUI: ({ onClose }) => {
          return (
            <div className="react-confirm-alert-custom">
              <h1>
                <FaRegPlayCircle />
                Start Model
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
                    _runStartModel(rowData, rowIndex)
                    onClose()
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
    },
    [_runStartModel]
  )

  const _getActiveModel = useCallback(
    (rowData, rowIndex) => {
      let flag = false
      flag = activeModel.some(model => {
        return model?.USEABLE_CLASS.some(cls => {
          if (cls.CLASS_CD === rowData.CLASS_CD) {
            return true
          }
        })
      })
      if (flag) {
        return (
          <IoIosArrowDropright
            className="icon-pointer icon-active-model"
            data-tip={"Run Predict"}
            onClick={_runPreTrain(rowData, rowIndex)}
          />
        )
      } else {
        return <IoMdWalk className="icon-pointer icon-active-model" data-tip={"Start Model"} onClick={_startModel(rowData, rowIndex)} />
      }
    },
    [activeModel, _runPreTrain, _startModel]
  )

  const _handleStatusCheck = useCallback(() => {
    _getActiveStatus()
    toast.info(<CommonToast Icon={GoDashboard} text={"Model Status Check"} />)
  }, [_getActiveStatus])

  const columns = useMemo(
    () => [
      {
        label: "Color",
        width: 100,
        className: "text-center",
        disableSort: true,
        dataKey: "COLOR",
        cellRenderer: ({ cellData, rowIndex }) => {
          if (tagList[rowIndex]?.isShow) {
            return (
              <BsFillCircleFill
                style={{ verticalAlign: "sub", color: cellData, fontSize: "15px", cursor: "pointer" }}
                onClick={_tagShow(rowIndex)}
              />
            )
          } else {
            return (
              <BsFillCircleFill
                style={{ verticalAlign: "sub", color: "grey", fontSize: "15px", cursor: "pointer" }}
                onClick={_tagShow(rowIndex)}
              />
            )
          }
        }
      },
      {
        label: "Tag",
        width: 200,
        className: "text-left",
        disableSort: false,
        dataKey: "NAME",
        cellRenderer: ({ cellData }) => {
          return <div style={{ fontSize: "12px" }}>{cellData}</div>
        }
      },
      {
        label: "-",
        dataKey: "",
        className: "text-center",
        disableSort: true,
        width: 50,
        headerRenderer: () => {
          return (
            <div>
              <FaRunning className="mr-1" />
            </div>
          )
        },
        // eslint-disable-next-line react/display-name, react/prop-types
        cellRenderer: ({ rowData, rowIndex }) => {
          if (rowData.CLASS_CD !== null) {
            return _getActiveModel(rowData, rowIndex)
          } else {
            return <div>-</div>
          }
        }
      },
      {
        label: "-",
        dataKey: "",
        className: "text-center",
        disableSort: true,
        width: 50,
        headerRenderer: () => {
          return (
            <div>
              <RiDeleteBin3Line className="mr-1" />
            </div>
          )
        },
        cellRenderer: ({ rowIndex }) => {
          return <MdBackspace className="icon-pointer font-17 ver-sub " onClick={_removeTag(rowIndex)} />
        }
      }
    ],
    [tagList, _tagShow, _removeTag, _getActiveModel]
  )

  return (
    <>
      {isAddModal && (
        <TagModal
          mode={mode}
          tagInfo={tagInfo}
          updateIndex={updateIndex}
          dataSetCd={props.dataSetCd}
          objectType={props.objectType}
          dataType={props.dataType}
          _successModal={_successModal}
          toggle={_tagAddModal}
          modal={isAddModal}
        />
      )}
      {isStatusModal && (
        <StatusModal
          toggle={_openStatusModal}
          modal={isStatusModal}
          activeModel={activeModel}
          tagList={tagList}
          getActiveStatus={_getActiveStatus}
        />
      )}
      <div className="card__title mt-1 mb-0">
        <h5 className="bold-text">
          <span data-tip={"Up Tag [w] Down Tag [s]"}>
            <RiPriceTag3Line className="mr-1 font-14 mb-1" />
            Tag
          </span>
          <GoPlus className="float-right mr-2 icon-pointer font-14" data-tip={"Add Tag"} onClick={_openAddModal} />
          <GoDashboard className="float-right mr-2 icon-pointer font-14" data-tip={"Mini Predictor Status"} onClick={_handleStatusCheck} />
        </h5>
      </div>
      <LoadingOverlay active={isLoading} spinner text="Predict images & recommend tags...">
        <VirtualTable
          className="vt-table image-table-font "
          rowClassName="vt-header image-table-header"
          height="150px"
          headerHeight={25}
          rowHeight={25}
          scrollIndex={curTagIndex}
          columns={columns}
          data={tagList}
          onRowClick={_onRowClick}
          onRowDoubleClick={_onRowDoubleClick}
          onRowMouseOver={_onRowMouseOver}
          onRowMouseOut={_onRowMouseOut}
          rowStyle={_rowStyle}
          isNoRowRender={false}
        />
      </LoadingOverlay>
    </>
  )
}

TagList.propTypes = {
  dataSetCd: PropTypes.string
}

export default React.memo(TagList)
