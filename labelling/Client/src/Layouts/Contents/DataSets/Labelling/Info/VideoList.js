import React, { useState, useMemo, useEffect, useRef, useCallback } from "react"
import PropTypes from "prop-types"
import { FaVideo } from "react-icons/fa"
import VirtualTable from "Components/Common/VirtualTable"
import { useDispatch, useSelector } from "react-redux"
import LoadingOverlay from "react-loading-overlay"
import { toast } from "react-toastify"
import { MdSave, MdError } from "react-icons/md"
import { cloneDeep } from "lodash-es"

import CommonToast from "../../../../../Components/Common/CommonToast"
import * as VideoAnnoApi from "Config/Services/VideoAnnoApi"
import * as VideoLabelActions from "Redux/Actions/VideoLabelActions"
import { _getNeedCount } from "Components/Utils/Utils"

// state prev value 저장
function usePrevious(value) {
  const ref = useRef()
  useEffect(() => {
    ref.current = value
  }, [value])
  return ref.current
}

const iconStyle = { fontSize: "14px", marginBottom: "0.1rem" }

function VideoList(props) {
  const [fileList, setFileList] = useState([])
  const [curIndex, setCurIndex] = useState(null)
  const [hoverIndex, setHoverIndex] = useState(null)
  const [onLoad, setOnLoad] = useState(false)

  const prevIndex = usePrevious(curIndex)
  const dispatch = useDispatch()
  const videoLabel = useSelector(
    state => state.videoLabel,
    (prevItem, item) => {
      if (item === prevItem) return true
      else return false
    }
  )

  useEffect(() => {
    if (props.dataSetCd) {
      let param = { DATASET_CD: props.dataSetCd }
      VideoAnnoApi._getVideoList(param)
        .then(data => {
          dispatch(VideoLabelActions._setVideoListLength(data.length))
          setFileList(data)
          setOnLoad(true)
        })
        .catch(e => {
          console.log(e)
        })
    }
  }, [props.dataSetCd])

  // 첫 화면 진입시 ImageList, TagList 모두 로딩되야 현재 이미지 불러오기 시작
  useEffect(() => {
    if (props.renderLazy && onLoad) {
      setCurIndex(0)
      setOnLoad(false)
    }
  }, [props.renderLazy, onLoad])

  // Label top 영역에서 화살표로 누를 때 변경되는 Redux curIndex 감지
  useEffect(() => {
    if (fileList.length !== 0 && videoLabel.curIndex !== curIndex) {
      setCurIndex(videoLabel.curIndex)
    }
  }, [videoLabel.curIndex])

  useEffect(() => {
    if (props.saveDataCd !== null) {
      props.setSaveDataCd(null)
      _updateLastFrame()
    }
  }, [props.saveDataCd])

  // curIndex 변경 될 때 해당 Video 정보 가져옴
  useEffect(() => {
    // 이전 Index가 있으면 이전 결과 저장
    if (prevIndex !== undefined && prevIndex !== null) {
      if (videoLabel.isDrawAction) {
        const param = {
          DATASET_CD: videoLabel.curVideo.DATASET_CD,
          DATA_CD: videoLabel.curVideo.DATA_CD,
          ANNO_DATA: { POLYGON_DATA: videoLabel.objectList, BRUSH_DATA: videoLabel.brushList },
          OBJECT_TYPE: props.objectType
        }
        VideoAnnoApi._setVideoAnnotation(param)
          .then(result => {
            if (result.status === 1) {
              _updateLastFrame()
              toast.info(<CommonToast Icon={MdSave} text={"Save Success"} />)
            } else {
              throw { err: "status 0" }
            }
          })
          .catch(e => {
            toast.error(<CommonToast Icon={MdError} text={"Save Fail"} />)
            console.log(e)
          })
        dispatch(VideoLabelActions._isDrawAction(false))
      }
    }

    // 이동한 index 영상 정보 불러오기
    if (fileList.length !== 0 && curIndex !== null) {
      const param = { DATASET_CD: fileList[curIndex].DATASET_CD, DATA_CD: fileList[curIndex].DATA_CD, OBJECT_TYPE: props.objectType }
      VideoAnnoApi._getVideo(param)
        .then(data => {
          // 현재 비디오 정보 redux dispatch
          dispatch(VideoLabelActions._setCurVideo(curIndex, data))

          // polygon 정보 저장
          let objectList = new Array()
          if (data?.ANNO_DATA?.POLYGON_DATA && data.ANNO_DATA.POLYGON_DATA.length !== 0) {
            data.ANNO_DATA.POLYGON_DATA.forEach(frameData => {
              let frameArray = new Array()
              if (frameData.length !== 0) {
                frameData.forEach(ele => {
                  // 현재 불러온 tag 정보들이 현재 TagList에 있는지 확인
                  const hasTag = videoLabel.tagList.some(tag => String(tag.TAG_CD) === String(ele.TAG_CD))
                  let tagColor = ele.COLOR
                  // 현재 Tag의 색상이 변경 됐을 경우 Object Tag Color 변경
                  videoLabel.tagList.some(tag => {
                    if (String(tag.TAG_CD) === String(ele.TAG_CD) && ele.COLOR !== tag.COLOR) {
                      tagColor = tag.COLOR
                      return true
                    }
                  })
                  if (hasTag) {
                    let obj = {}
                    obj.DATASET_CD = ele.DATASET_CD
                    obj.DATA_CD = ele.DATA_CD
                    obj.TAG_CD = ele.TAG_CD
                    obj.TAG_NAME = ele.TAG_NAME
                    obj.COLOR = tagColor
                    obj.CURSOR = ele.CURSOR
                    obj.NEEDCOUNT = _getNeedCount(obj.CURSOR)
                    obj.POSITION = ele.POSITION
                    obj.ACCURACY = ele.ACCURACY
                    frameArray.push(obj)
                  }
                })
              }
              objectList.push(frameArray)
            })
          }
          dispatch(VideoLabelActions._setObjectList(objectList))
          dispatch(VideoLabelActions._setVideoStatus(false))
        })
        .catch(err => {
          dispatch(VideoLabelActions._setObjectList([]))
          dispatch(VideoLabelActions._setVideoStatus(false))
          console.log(err)
        })
    }
  }, [curIndex])

  const _updateLastFrame = useCallback(() => {
    let cFileList = cloneDeep(fileList)
    let index = fileList.findIndex(file => file.DATA_CD === videoLabel.curVideo.DATA_CD)
    cFileList[index].IS_ANNO = videoLabel.objectList.length !== 0 ? 1 : 0
    let maxCnt = 0
    videoLabel.objectList.forEach((ele, idx) => {
      if (ele.length > 0) maxCnt = idx
    })
    cFileList[index].ANNO_CNT = maxCnt
    setFileList(cFileList)
  }, [fileList, videoLabel.objectList, videoLabel.curVideo])

  const _onRowMouseOver = useCallback(({ index }) => {
    setHoverIndex(index)
  }, [])

  const _onRowMouseOut = useCallback(() => {
    setHoverIndex(null)
  }, [])

  const _onRowClick = useCallback(
    ({ index }) => {
      if (curIndex !== index) {
        if (!videoLabel.imageStatus) {
          dispatch(VideoLabelActions._setVideoStatus(true))
          setCurIndex(index)
        }
      }
    },
    [curIndex, videoLabel.imageStatus]
  )

  const _rowStyle = useCallback(
    ({ index }) => {
      if (curIndex === index) {
        return { backgroundColor: "#015aa7af", color: "white" }
      }
      if (hoverIndex === index) {
        return { backgroundColor: "#2f2f2f" }
      }
      return {}
    },
    [hoverIndex, curIndex]
  )

  const columns = useMemo(() => {
    return [
      {
        label: "#",
        width: 50,
        className: "text-center",
        disableSort: true,
        dataKey: "index",
        cellRenderer: ({ rowIndex }) => {
          return <div className="videoList-index">{rowIndex === curIndex && videoLabel.isDrawAction ? <span>*</span> : rowIndex + 1}</div>
        }
      },
      {
        label: "Video",
        width: 80,
        className: "text-center",
        disableSort: true,
        dataKey: "THUM",
        cellRenderer: ({ cellData }) => {
          return <img className="list-thumbnail" src={cellData} />
        }
      },
      {
        label: "Name",
        dataKey: "FILE_NAME",
        className: "text-left",
        disableSort: true,
        width: 200,
        cellRenderer: ({ rowData }) => {
          return (
            <div>
              <div className="videoList-fileName">
                {rowData.FILE_NAME}
                {rowData.FILE_EXT}
              </div>
              <div className="videoList-fileName-sub">LastFrame: {rowData.ANNO_CNT === null ? 0 : rowData.ANNO_CNT}</div>
            </div>
          )
        }
      }
    ]
  }, [curIndex, videoLabel.isDrawAction])

  return (
    <>
      <div className="card__title mt-1">
        <h5 className="bold-text">
          <span data-tip={"Prev [a] 10 Prev [shift+a] Next [d] 10 Next [shift+d]"}>
            <FaVideo className="mr-1" style={iconStyle} />
            Video list
          </span>
        </h5>
      </div>
      <LoadingOverlay active={onLoad} spinner text="Load data...">
        <VirtualTable
          className="vt-table image-table-font "
          rowClassName="vt-header image-table-header"
          height="calc(100vh - 280px)"
          headerHeight={25}
          rowHeight={60}
          columns={columns}
          data={fileList}
          // scrollIndex={curIndex}
          onRowMouseOver={_onRowMouseOver}
          onRowMouseOut={_onRowMouseOut}
          onRowClick={_onRowClick}
          rowStyle={_rowStyle}
        />
      </LoadingOverlay>
    </>
  )
}

VideoList.propTypes = {
  dataSetCd: PropTypes.string
}

export default React.memo(VideoList)
