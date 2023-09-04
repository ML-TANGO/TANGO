import React, { useState, useMemo, useEffect, useRef, useCallback } from "react"
import PropTypes from "prop-types"
import { FaImages } from "react-icons/fa"
import { useDispatch, useSelector } from "react-redux"
import LoadingOverlay from "react-loading-overlay"
import { toast } from "react-toastify"
import { MdSave, MdError } from "react-icons/md"
import { cloneDeep } from "lodash-es"

import CommonToast from "../../../../../Components/Common/CommonToast"
import { _getNeedCount } from "Components/Utils/Utils"
import VirtualTable from "Components/Common/VirtualTable"
import * as ImageAnnoApi from "Config/Services/ImageAnnoApi"
import * as ImageLabelActions from "Redux/Actions/ImageLabelActions"

// state prev value 저장
function usePrevious(value) {
  const ref = useRef()
  useEffect(() => {
    ref.current = value
  }, [value])
  return ref.current
}

function ImageList(props) {
  const [fileList, setFileList] = useState([])
  const [curIndex, setCurIndex] = useState(null)
  const [hoverIndex, setHoverIndex] = useState(null)
  const [onLoad, setOnLoad] = useState(false)

  const prevIndex = usePrevious(curIndex)
  const dispatch = useDispatch()
  const imageLabel = useSelector(
    state => state.imageLabel,
    (prevItem, item) => {
      if (item === prevItem) return true
      else return false
    }
  )

  useEffect(() => {
    if (props.dataSetCd) {
      let param = { DATASET_CD: props.dataSetCd }
      ImageAnnoApi._getImageList(param)
        .then(data => {
          dispatch(ImageLabelActions._setImageListLength(data.length))
          setFileList(data)
          setOnLoad(true)
        })
        .catch(e => {
          setOnLoad(false)
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
    if (fileList.length !== 0 && imageLabel.curIndex !== curIndex) {
      setCurIndex(imageLabel.curIndex)
    }
  }, [imageLabel.curIndex])

  useEffect(() => {
    if (props.saveDataCd !== null) {
      props.setSaveDataCd(null)
      _updateAnnoCount()
    }
  }, [props.saveDataCd])

  // curIndex 변경 될 때 해당 Image 정보 가져옴
  useEffect(() => {
    // 이전 Index가 있으면 이전 결과 저장
    if (prevIndex !== undefined && prevIndex !== null) {
      if (imageLabel.isDrawAction) {
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
                _updateAnnoCount()
                toast.info(<CommonToast Icon={MdSave} text={"Save Success"} />)
              } else {
                throw { err: "status 0" }
              }
            })
            .catch(e => {
              toast.error(<CommonToast Icon={MdError} text={"Save Fail"} />)
              console.log(e)
            })
        } else if (props.objectType === "S") {
          dispatch(ImageLabelActions._saveImage(true))
        }
      }
      dispatch(ImageLabelActions._isDrawAction(false))
    }

    // 이동한 index 이미지 정보 불러오기
    if (fileList.length !== 0 && curIndex !== null) {
      const param = { DATASET_CD: fileList[curIndex].DATASET_CD, DATA_CD: fileList[curIndex].DATA_CD, OBJECT_TYPE: props.objectType }
      ImageAnnoApi._getImage(param)
        .then(data => {
          // 현재 이미지 정보 redux dispatch
          dispatch(ImageLabelActions._setCurImage(curIndex, data))

          // polygon 정보 저장
          let objectList = new Array()
          if (data?.ANNO_DATA?.POLYGON_DATA && data?.ANNO_DATA?.POLYGON_DATA.length !== 0) {
            data.ANNO_DATA.POLYGON_DATA.forEach(ele => {
              // 현재 불러온 tag 정보들이 현재 TagList에 있는지 확인
              const hasTag = imageLabel.tagList.some(tag => String(tag.TAG_CD) === String(ele.TAG_CD))
              let tagColor = ele.COLOR
              // 현재 Tag의 색상이 변경 됐을 경우 Object Tag Color 변경
              imageLabel.tagList.some(tag => {
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
                obj.CURSOR = props.objectType === "D" ? "isRect" : "isPolygon"
                obj.NEEDCOUNT = _getNeedCount(obj.CURSOR)
                obj.POSITION = ele.POSITION
                obj.ACCURACY = ele.ACCURACY
                objectList.push(obj)
              }
            })
          }

          // brush 정보 저장
          let brushList = new Array()
          if (data?.ANNO_DATA?.BRUSH_DATA && data?.ANNO_DATA?.BRUSH_DATA.length !== 0 && props.objectType === "S") {
            data.ANNO_DATA.BRUSH_DATA.forEach(ele => {
              // 현재 불러온 tag 정보들이 현재 TagList에 있는지 확인
              const hasTag = imageLabel.tagList.some(tag => String(tag.TAG_CD) === String(ele.TAG_CD))
              let tagColor = ele.COLOR
              // 현재 Tag의 색상이 변경 됐을 경우 Object Tag Color 변경
              imageLabel.tagList.some(tag => {
                if (String(tag.TAG_CD) === String(ele.TAG_CD) && ele.COLOR !== tag.COLOR) {
                  tagColor = tag.COLOR
                  return true
                }
              })

              if (hasTag) {
                let obj = {
                  TAG_CD: ele.TAG_CD,
                  COLOR: tagColor,
                  LINE_WIDTH: ele.LINE_WIDTH,
                  MODE: ele.MODE,
                  POINTS: ele.POINTS
                }
                brushList.push(obj)
              }
            })
          }

          dispatch(ImageLabelActions._setObjectList(objectList))
          dispatch(ImageLabelActions._setBrushList(brushList))
          dispatch(ImageLabelActions._setImageStatus(false))
        })
        .catch(err => {
          dispatch(ImageLabelActions._setBrushList([]))
          dispatch(ImageLabelActions._setObjectList([]))
          dispatch(ImageLabelActions._setImageStatus(false))
          console.log(err)
        })
    }
  }, [curIndex])

  useEffect(() => {
    if (imageLabel.maskImg.length !== 0 && imageLabel.saveImage) {
      const param = {
        DATASET_CD: imageLabel.curImage.DATASET_CD,
        DATA_CD: imageLabel.curImage.DATA_CD,
        FILE_PATH: imageLabel.curImage.FILE_PATH,
        ANNO_DATA: { POLYGON_DATA: imageLabel.objectList, BRUSH_DATA: imageLabel.brushList },
        MASK_IMG: imageLabel.maskImg,
        OBJECT_TYPE: props.objectType
      }
      ImageAnnoApi._setImageAnnotation(param)
        .then(result => {
          if (result.status === 1) {
            _updateAnnoCount()

            dispatch(ImageLabelActions._saveImage(false))
            dispatch(ImageLabelActions._setMaskImg([]))
            toast.info(<CommonToast Icon={MdSave} text={"Save Success"} />)
          } else {
            throw { err: "status 0" }
          }
        })
        .catch(e => {
          console.log(e)
          dispatch(ImageLabelActions._saveImage(false))
          dispatch(ImageLabelActions._setMaskImg([]))
          toast.error(<CommonToast Icon={MdError} text={"Save Fail"} />)
        })
    }
  }, [imageLabel.maskImg])

  const _updateAnnoCount = useCallback(() => {
    let cFileList = cloneDeep(fileList)
    let index = fileList.findIndex(file => file.DATA_CD === imageLabel.curImage.DATA_CD)
    cFileList[index].IS_ANNO = imageLabel.objectList.length !== 0 ? 1 : 0
    cFileList[index].ANNO_CNT = imageLabel.objectList.length
    const tagCds = imageLabel.objectList.map(ele => String(ele.TAG_CD))
    const unique = Array.from(new Set(tagCds))
    cFileList[index].TAG_CNT = unique.length
    setFileList(cFileList)
  }, [fileList, imageLabel.objectList, imageLabel.curImage])

  const _onRowMouseOver = useCallback(({ index }) => {
    setHoverIndex(index)
  }, [])

  const _onRowMouseOut = useCallback(() => {}, [])

  const _onRowClick = useCallback(
    ({ index }) => {
      if (curIndex !== index) {
        if (!imageLabel.imageStatus) {
          dispatch(ImageLabelActions._setImageStatus(true))
          setCurIndex(index)
        }
      }
    },
    [curIndex, imageLabel.imageStatus]
  )

  const _rowStyle = useCallback(
    ({ index }) => {
      if (curIndex === index) {
        return { backgroundColor: "#015aa7af", color: "white" }
      }
      if (hoverIndex === index) {
        return { backgroundColor: "#2f2f2f" }
      }
      return
    },
    [hoverIndex, curIndex]
  )

  const columns = useMemo(
    () => [
      {
        label: "Thumbnail",
        width: 100,
        className: "text-center",
        disableSort: true,
        dataKey: "THUM",
        cellRenderer: ({ cellData }) => {
          return <img className="list-thumbnail" src={cellData} />
        }
      },
      {
        label: "No.",
        width: 60,
        className: "text-center",
        disableSort: true,
        dataKey: "index",
        cellRenderer: ({ rowIndex }) => {
          // return <div style={{ color: "#888888", fontStyle: "italic" }}>{rowIndex + 1}</div>
          return (
            <div style={{ color: "#888888", fontStyle: "italic" }}>
              {rowIndex === curIndex && imageLabel.isDrawAction ? (
                <span style={{ fontSize: "1rem", fontWeight: "bold" }}>*</span>
              ) : (
                rowIndex + 1
              )}
            </div>
          )
        }
      },
      {
        label: "Description",
        dataKey: "FILE_NAME",
        className: "text-left",
        disableSort: true,
        width: 200,
        cellRenderer: ({ rowData }) => {
          return (
            <div>
              <div style={{ fontSize: "14px" }}>
                {rowData.FILE_NAME}
                {rowData.FILE_EXT}
              </div>
              <div style={{ fontSize: "8px", marginTop: "-3px", color: "#888888" }}>
                Tagged: {rowData.TAG_CNT === null ? 0 : rowData.TAG_CNT}, Boxes: {rowData.ANNO_CNT === null ? 0 : rowData.ANNO_CNT}{" "}
              </div>
            </div>
          )
        }
      }
    ],
    [fileList, curIndex, imageLabel.isDrawAction]
  )

  return (
    <>
      <div className="card__title mt-3">
        <h5 className="bold-text">
          <span data-tip={"Prev [a] 10 Prev [shift+a] Next [d] 10 Next [shift+d]"}>
            <FaImages className="mr-1 mb-1 font-14" />
            Image list
          </span>
        </h5>
      </div>
      <LoadingOverlay active={onLoad} spinner text="Load data...">
        <VirtualTable
          className="vt-table image-table-font"
          rowClassName="vt-header image-table-header"
          height="calc(100vh - 280px)"
          headerHeight={25}
          rowHeight={60}
          columns={columns}
          data={fileList}
          scrollIndex={curIndex}
          onRowMouseOver={_onRowMouseOver}
          onRowMouseOut={_onRowMouseOut}
          onRowClick={_onRowClick}
          rowStyle={_rowStyle}
        />
      </LoadingOverlay>
    </>
  )
}

ImageList.propTypes = {
  dataSetCd: PropTypes.string
}

export default React.memo(ImageList)
