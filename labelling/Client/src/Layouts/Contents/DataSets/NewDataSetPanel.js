import React, { useState, useEffect, useLayoutEffect, useRef } from "react"
import axios from "axios"
import stc from "string-to-color"
import moment from "moment"
import path from "path"
import LoadingOverlay from "react-loading-overlay"
import { Col, Row } from "reactstrap"
import { useForm } from "react-hook-form"
import { v1 as uuidv1 } from "uuid"
import { toast } from "react-toastify"
import { confirmAlert } from "react-confirm-alert"
import { cloneDeep, transform, isEqual, isObject } from "lodash-es"
import ReactTooltip from "react-tooltip"
import { CircularProgressbar, buildStyles } from "react-circular-progressbar"

import { MdPauseCircleOutline, MdCheckCircle, MdError, MdSave } from "react-icons/md"
import { RiPriceTag3Fill, RiSendPlaneFill } from "react-icons/ri"
import { FaTrashAlt, FaUpload, FaColumns, FaDatabase } from "react-icons/fa"
import { FiImage, FiVideo } from "react-icons/fi"
import { VscSymbolClass } from "react-icons/vsc"
import { BsBoundingBoxCircles } from "react-icons/bs"

// api
import * as DataSetApi from "Config/Services/DataSetApi"
import * as TabDataSetApi from "Config/Services/TabDataSetApi"
import * as AimodelApi from "Config/Services/AimodelApi"

import { FormText, FormNumber, FormSelect, FormTextArea, FormIcon } from "../../../Components/Form/FormComponent"
import { bytesToSize } from "Components/Utils/Utils"
import useEnterpriseDivision from "Components/Utils/useEnterpriseDivision"
import BeatLoader from "react-spinners/BeatLoader"
import styled from "styled-components"

import CommonToast from "Components/Common/CommonToast"
import { CustomColumnAlert, CustomConfirmAlert } from "./NewDataSetCustom"
import { LabelModal } from "../../../Components/Modals/LabelModal"
import UploadModal from "../../../Components/Modals/UploadModal"
import DbModal from "../../../Components/Modals/DbModal"
import ColumnModal from "../../../Components/Modals/ColumnModal"
import CommonButton from "../../../Components/Common/CommonButton"
import VirtualTable from "../../../Components/Common/VirtualTable"
import CommonPanel from "../../../Components/Panel/CommonPanel"
import IconDrawPloygon from "../../../Components/Form/IconDrawPloygon"

const StyledLoader = styled(LoadingOverlay)`
  .MyLoader_overlay {
    background: none;
  }
  .MyLoader_content {
    top: 50%;
    right: 50%;
  }
  &.MyLoader_wrapper--active {
    overflow: hidden;
    width: 100%;
    height: 100%;
  }
`

const dataTypeList = [
  { title: "Image", type: "I", icon: <FiImage size="50" /> },
  { title: "Video", type: "V", icon: <FiVideo size="50" /> }
]

const purposeList = [
  { title: "Classfication", type: "C", icon: <VscSymbolClass size="50" />, viewType: ["I", "T"] },
  { title: "Detection", type: "D", icon: <BsBoundingBoxCircles size="49" />, viewType: ["I", "V"] },
  { title: "Segmentation", type: "S", icon: <IconDrawPloygon size="48" fill="#fff" />, viewType: ["I", "V"] }
]

const importOptions = [
  { value: "N", label: "NO" },
  { label: "COCO", value: "COCO" }
]

const filterModel = (list, objectType) =>
  list
    .filter(el => el.OBJECT_TYPE === objectType)
    .map(el => ({
      label: `${el.TITLE} : ${el.AI_CD}`,
      value: el.AI_CD,
      type: el.TYPE
    }))

const TableIcon = ({ ButtonIcon, className, tooltip, disabled, onClick }) => {
  return (
    <ButtonIcon
      className={`${className} ${disabled ? "ds-table-icon-disabled" : "ds-table-icon"} `}
      data-tip={disabled ? null : tooltip}
      onClick={disabled ? undefined : onClick}
    />
  )
}
function NewDataSetPanel(props) {
  const { panelToggle, springProps, editData, initDataSet, isPanelRender } = props
  const [source] = useState(axios.CancelToken.source)
  const [uuid] = useState(uuidv1())
  const [pageState, setPageState] = useState({
    title: "New Dataset",
    isSave: false,
    isUpload: false,
    modal: false,
    pageMode: "NEW",
    fileHoverIndex: null,
    columnHoverIndex: null,
    uploadModal: false,
    columnModal: false,
    dbModal: false,
    dbColumnModal: false
  })
  const [typeState, setTypeState] = useState({ dataType: "I", objectType: "D", importType: "N" })
  const [tableState, setTableState] = useState({
    tableData: [],
    tableInfo: {},
    colList: [],
    columns: []
  })
  const [fileState, setFileState] = useState({
    fileList: [],
    removeFiles: [],
    selectedFiles: [],
    totalCount: 0,
    successCount: 0,
    failCount: 0,
    checkAllFiles: false,
    colList: [],
    selectedColumns: [],
    checkAllColumns: false
  })

  const autoUserModel = useEnterpriseDivision(process.env.BUILD, "dataSet", "autoUserModel")
  const dataSetUpload = useEnterpriseDivision(process.env.BUILD, "dataSet", "dataSetUpload")
  const { register, handleSubmit, control, setValue, errors, watch } = useForm()

  const [optionState, setOptionState] = useState({ tagOption: [], baseModelOption: [], filteredModelOption: [], modelEpochs: [] })
  const [autoModelList, setAutoModelList] = useState([])
  const [uploadType, setUploadType] = useState("FILE")
  const [panelPos, setPanelPos] = useState({ top: 0, bottom: 0 })
  const watchAutoType = watch("AUTO_TYPE", "N")
  const resultRef = useRef(null)
  const toggle = () => setPageState(prevState => ({ ...prevState, modal: !prevState.modal }))

  const beforeunload = function (e) {
    // 경고 메시지 설정 (사용자에게 표시할 내용)
    const confirmationMessage = "변경 사항이 저장되지 않을 수 있습니다. 페이지를 나가시겠습니까?"

    // 이벤트 객체에 경고 메시지를 추가하여 얼럿 또는 확인 다이얼로그를 표시
    e.returnValue = confirmationMessage
    return confirmationMessage
  }
  useEffect(() => {
    window.addEventListener("beforeunload", beforeunload)
    return () => {
      window.removeEventListener("beforeunload", beforeunload)
    }
  }, [])

  useLayoutEffect(() => {
    async function init() {
      try {
        const result = await AimodelApi._getSelectedModelList({ DATA_TYPE: "I" })
        const filter = result.filter(ele => {
          if (!autoUserModel) {
            if (ele.EPOCH !== -1) return false
            else return true
          } else {
            return true
          }
        })
        setAutoModelList(filter)
        setOptionState(prevState => ({ ...prevState, baseModelOption: filter, filteredModelOption: filterModel(filter, "D") }))
        setPageState(prevState => ({ ...prevState, pageMode: editData.pageMode }))
        const dataInfo = editData.dataInfo
        switch (editData.pageMode) {
          case "EDIT":
            setPageState(prevState => ({ ...prevState, title: "Edit Dataset" }))
            Object.keys(dataInfo).map(ok => {
              setValue(ok, dataInfo[ok])
            })

            setTypeState(prevState => ({ ...prevState, objectType: dataInfo.OBJECT_TYPE, dataType: dataInfo.DATA_TYPE }))
            setOptionState(prevState => ({
              ...prevState,
              filteredModelOption: filterModel(filter, dataInfo.OBJECT_TYPE)
            }))

            if (dataInfo.AUTO_ACC !== null && dataInfo.AUTO_MODEL !== null) {
              setValue("AUTO_TYPE", "Y")
              setValue("AUTO_ACC", dataInfo.AUTO_ACC * 100)
              if (dataInfo.AUTO_MODEL !== null) {
                setValue("AUTO_MODEL", dataInfo.AUTO_MODEL)
              }
            }

            let fileList, tableInfo, tableData
            if (dataInfo.DATA_TYPE === "T") {
              setUploadType(dataInfo.UPLOAD_TYPE)
              if (dataInfo.UPLOAD_TYPE === "FILE") {
                fileList = await TabDataSetApi._getFileList({ DATASET_CD: dataInfo.DATASET_CD })
              } else {
                tableInfo = await TabDataSetApi._getDBInfo({ DATASET_CD: dataInfo.DATASET_CD })
                tableData = await TabDataSetApi._getDbConnectionInfo(tableInfo.table)
              }
            } else {
              fileList = await DataSetApi._getFileList({ DATASET_CD: dataInfo.DATASET_CD })
            }
            if (dataInfo?.UPLOAD_TYPE !== "DB") {
              const list = fileList.map(file => ({
                name: file.FILE_NAME + file.FILE_EXT,
                size: file.FILE_SIZE,
                type: file.FILE_EXT.substring(1),
                path: file.FILE_PATH,
                status: 1,
                progress: 100,
                base: path.basename(path.dirname(file.FILE_PATH)),
                columns: file?.columns
              }))

              setFileState(prevState => ({ ...prevState, fileList: list, successCount: list.length }))
            } else {
              setTableState(prevState => ({ ...prevState, tableInfo: tableInfo.table, tableData: tableData.DATA ? tableData.DATA : [] }))
            }

            if (dataInfo.DATA_TYPE === "T") {
              const columnList = await TabDataSetApi._getfeatures({ DATASET_CD: dataInfo.DATASET_CD })
              const filterList = columnList.map(el => {
                if (el.DEFAULT_VALUE) {
                  return el
                } else {
                  return { ...el, DEFAULT_VALUE: "null" }
                }
              })

              const target = columnList.find(el => el.IS_CLASS === 1)?.COLUMN_NM
              setValue("TARGET", target)

              if (dataInfo.UPLOAD_TYPE === "FILE") {
                setFileState(prevState => ({ ...prevState, colList: filterList }))
              } else {
                const columns = columnList.map(el => ({
                  label: el.COLUMN_NM,
                  dataKey: el.COLUMN_NM,
                  className: "text-center",
                  disableSort: true,
                  width: 120
                }))
                const indexColumn = {
                  label: "#",
                  dataKey: "-",
                  className: "text-center",
                  disableSort: true,
                  width: 80,
                  cellRenderer: ({ rowIndex }) => rowIndex + 1
                }
                setTableState(prevState => ({ ...prevState, colList: filterList, columns: [indexColumn, ...columns] }))
              }
            }
            break

          case "DUPLICATION":
            setPageState(prevState => ({ ...prevState, title: "Duplication Dataset" }))
            Object.keys(dataInfo).map(ok => {
              if (ok === "TITLE") setValue("TITLE", dataInfo[ok] + "_" + moment().format("YYYYMMDDHHmmss"))
              else setValue(ok, dataInfo[ok])
            })
            setPageState(prevState => ({ ...prevState, title: "Duplication Dataset", subTitle: "Copy Your Dataset" }))
            setTypeState(prevState => ({ ...prevState, objectType: dataInfo.OBJECT_TYPE, dataType: dataInfo.DATA_TYPE }))
            let f
            if (dataInfo.DATA_TYPE === "T") {
              setUploadType(dataInfo.UPLOAD_TYPE)
              if (dataInfo.UPLOAD_TYPE === "FILE") {
                f = await TabDataSetApi._getFileList({ DATASET_CD: dataInfo.DATASET_CD })
              } else {
                tableInfo = await TabDataSetApi._getDBInfo({ DATASET_CD: dataInfo.DATASET_CD })
                tableData = await TabDataSetApi._getDbConnectionInfo(tableInfo.table)
              }
            } else {
              f = await DataSetApi._getFileList({ DATASET_CD: dataInfo.DATASET_CD })
            }

            if (dataInfo?.UPLOAD_TYPE !== "DB") {
              const l = f.map(file => ({
                name: file.FILE_NAME + file.FILE_EXT,
                size: file.FILE_SIZE,
                type: file.FILE_EXT.substring(1),
                path: file.FILE_PATH,
                status: 1,
                progress: 100,
                base: path.basename(path.dirname(file.FILE_PATH)),
                columns: file?.columns
              }))

              setFileState(prevState => ({ ...prevState, fileList: l, successCount: l.length }))
            } else {
              setTableState(prevState => ({ ...prevState, tableInfo: tableInfo.table, tableData: tableData.DATA ? tableData.DATA : [] }))
            }

            if (dataInfo.DATA_TYPE === "T") {
              const columnList = await TabDataSetApi._getfeatures({ DATASET_CD: dataInfo.DATASET_CD })
              const filterList = columnList.map(el => {
                if (el.DEFAULT_VALUE) {
                  return el
                } else {
                  return { ...el, DEFAULT_VALUE: "null" }
                }
              })
              const target = columnList.find(el => el.IS_CLASS === 1)?.COLUMN_NM
              setValue("TARGET", target)

              if (dataInfo.UPLOAD_TYPE === "FILE") {
                setFileState(prevState => ({ ...prevState, colList: filterList }))
              } else {
                const columns = columnList.map(el => ({
                  label: el.COLUMN_NM,
                  dataKey: el.COLUMN_NM,
                  className: "text-center",
                  disableSort: true,
                  width: 120
                }))
                const indexColumn = {
                  label: "#",
                  dataKey: "-",
                  className: "text-center",
                  disableSort: true,
                  width: 80,
                  cellRenderer: ({ rowIndex }) => rowIndex + 1
                }
                setTableState(prevState => ({ ...prevState, colList: filterList, columns: [indexColumn, ...columns] }))
              }
            }
            break
          default:
            break
        }
      } catch (e) {
        console.log(e)
      }
    }
    init()
    ReactTooltip.rebuild()

    return () => {
      source.cancel("UNMOUNT")
    }
  }, [editData])

  useEffect(() => {
    if (typeState.objectType === "C") {
      const uniq = fileState.fileList.map(file => file.base).filter((v, i, s) => s.indexOf(v) === i)
      const tags = uniq.map(ele => ({ label: ele, value: ele, color: stc(ele) }))
      setOptionState(prevState => ({ ...prevState, tagOption: tags }))
    }
    setOptionState(prevState => ({
      ...prevState,
      filteredModelOption: filterModel(autoModelList, typeState.objectType)
    }))
    ReactTooltip.rebuild()
    setValue("AUTO_TYPE", "N")
  }, [typeState.objectType])

  useEffect(() => {
    if (fileState.fileList.length !== 0) {
      setValue("files", fileState.fileList)
      const selected = fileState.fileList.map(() => false)
      setFileState(prevState => ({ ...prevState, selectedFiles: selected }))
      if (typeState.objectType === "C") {
        const uniq = fileState.fileList.map(file => file.base).filter((v, i, s) => s.indexOf(v) === i)
        const tags = uniq.map(ele => ({ label: ele, value: ele, color: stc(ele) }))
        setOptionState(prevState => ({ ...prevState, tagOption: tags }))
      }
      if (fileState.totalCount < fileState.fileList.length) upload()
    }
    setFileState(prevState => ({ ...prevState, totalCount: fileState.fileList.length }))
  }, [fileState.fileList, typeState.objectType])

  useEffect(() => {
    if (fileState.colList !== 0) {
      const selected = fileState.colList.map(() => false)
      setFileState(prevState => ({ ...prevState, selectedColumns: selected }))
    }
  }, [fileState.colList])

  useEffect(() => {
    if (fileState.fileList.length !== 0) {
      const param = {
        uuid: uuid,
        fileList: fileState.fileList
      }
      DataSetApi._removeTempFiles(param)
        .then(() => {
          setFileState(prevState => ({ prevState, fileList: [], colList: [], selectedFiles: [], selectedColumns: [] }))
          setOptionState(prevState => ({ ...prevState, tagOption: [] }))
        })
        .catch(e => {
          setFileState(prevState => ({ prevState, fileList: [], colList: [], selectedFiles: [], selectedColumns: [] }))
          setOptionState(prevState => ({ ...prevState, tagOption: [] }))
          console.log(e)
        })
    }

    const flag = purposeList.some(el => el.type === typeState.objectType && !el.viewType.includes(typeState.dataType))
    if (flag) {
      if (typeState.dataType === "V") setValue("OBJECT_TYPE", "D")
      else setValue("OBJECT_TYPE", "C")
    }
    setValue("AUTO_TYPE", "N")
    setUploadType("FILE")
    setTableState({
      tableData: [],
      tableInfo: {},
      colList: [],
      columns: []
    })
    ReactTooltip.rebuild()
  }, [typeState.dataType])

  const onSubmit = data => {
    if (pageState.isSave) return
    data.uuid = uuid
    data.DATA_TYPE = typeState.dataType
    data.files = fileState.fileList
    data.USER_ID = JSON.parse(window.sessionStorage.getItem("userInfo"))?.USER_ID
    if (data.AUTO_TYPE === "Y") {
      let autoModel = autoModelList.filter(el => el.AI_CD === data.AUTO_MODEL)
      data.EPOCH = autoModel[0].EPOCH
    }
    setPageState(prevState => ({ ...prevState, isSave: true }))
    switch (editData.pageMode) {
      case "NEW":
        let files = data.files.map(ele => ({ name: ele.name, size: ele.size, path: ele.path, base: ele.base }))
        data.files = files
        data.AUTO_ACC = data.AUTO_ACC ? data.AUTO_ACC / 100 : 0
        data["tags"] = optionState.tagOption
        DataSetApi._setNewDataSets(data)
          .then(result => {
            if (result.status === 1) {
              toast.info(<CommonToast Icon={MdSave} text={"DataSet Create Success"} />)
              setPageState(prevState => ({ ...prevState, isSave: false }))
              panelToggle()
              initDataSet()
            } else {
              throw { err: "status 0", MSG: result.err.MSG }
            }
          })
          .catch(err => {
            console.log(err)
            setPageState(prevState => ({ ...prevState, isSave: false }))
            toast.error(<CommonToast Icon={MdError} text={`DataSet Create Fail \n ${err.MSG}`} />)
          })
        break
      case "EDIT":
        let param = {}
        // remove files
        param = data
        if (fileState.removeFiles.length !== 0) {
          param.remove = fileState.removeFiles
          setFileState(prevState => ({ ...prevState, removeFiles: [] }))
        }
        // add files
        param.DATASET_CD = editData.dataInfo.DATASET_CD
        param.fileList = []
        param.FILE_TYPE = editData.dataInfo.DATA_TYPE
        data.AUTO_ACC = data.AUTO_ACC ? data.AUTO_ACC / 100 : null
        data["tags"] = optionState.tagOption
        data.files.forEach((ele, i) => {
          if (ele.isNew !== undefined && ele.isNew) {
            param.fileList.push({ name: ele.name, path: ele.path, FILE_NUMBER: i, size: ele.size, base: ele.base })
          }
        })
        if (data.AUTO_TYPE === "Y") {
          confirmAlert({
            customUI: ({ onClose }) => {
              return (
                <CustomConfirmAlert
                  title={data.TITLE}
                  onClose={onClose}
                  data={data}
                  param={param}
                  setPageState={setPageState}
                  initDataSet={initDataSet}
                />
              )
            }
          })
        } else {
          DataSetApi._setUpdateDataset(param)
            .then(result => {
              if (result.status === 1) {
                toast.info(<CommonToast Icon={MdSave} text={"DataSet Update Success"} />)
                setPageState(prevState => ({ ...prevState, isSave: false }))
                panelToggle()
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
        }
        break
      case "DUPLICATION":
        data.ORG_DATASET_CD = editData.dataInfo.DATASET_CD
        DataSetApi._setDupDataset(data)
          .then(result => {
            if (result.status === 1) {
              toast.info(<CommonToast Icon={MdSave} text={"DataSet Duplication Success"} />)
              setPageState(prevState => ({ ...prevState, isSave: false }))
              panelToggle()
              initDataSet()
            } else {
              throw { err: "status 0" }
            }
          })
          .catch(err => {
            console.log(err)
            setPageState(prevState => ({ ...prevState, isSave: false }))
            toast.error(<CommonToast Icon={MdError} text={"DataSet Duplication Fail"} />)
          })
        break
    }
  }

  const onTextSubmit = data => {
    if (pageState.isSave) return
    data.uuid = uuid
    data.DATA_TYPE = typeState.dataType
    data.COLUMNS = data.UPLOAD_TYPE === "FILE" ? fileState.colList : tableState.colList
    data.files = fileState.fileList
    data.table = tableState.tableInfo
    data.USER_ID = JSON.parse(window.sessionStorage.getItem("userInfo"))?.USER_ID
    setPageState(prevState => ({ ...prevState, isSave: true }))
    switch (pageState.pageMode) {
      case "NEW":
        let files = data.files.map(ele => ({ name: ele.name, size: ele.size, path: ele.path }))
        data.files = files
        TabDataSetApi._createDataset(data)
          .then(result => {
            if (result.status === 1) {
              toast.info(<CommonToast Icon={MdSave} text={"DataSet Create Success"} />)
              setPageState(prevState => ({ ...prevState, isSave: false }))
              panelToggle()
              initDataSet()
            } else {
              throw { err: "status 0", MSG: result.err.MSG }
            }
          })
          .catch(err => {
            console.log(err)
            setPageState(prevState => ({ ...prevState, isSave: false }))
            toast.error(<CommonToast Icon={MdError} text={`DataSet Create Fail \n ${err.MSG}`} />)
          })
        break
      case "EDIT":
        let param = {}
        // remove files
        param = data
        if (fileState.removeFiles.length !== 0) {
          param.remove = fileState.removeFiles
          setFileState(prevState => ({ ...prevState, removeFiles: [] }))
        }
        // add files
        param.DATASET_CD = editData.dataInfo.DATASET_CD
        param.fileList = []
        param.FILE_TYPE = editData.dataInfo.DATA_TYPE
        data.files.forEach((ele, i) => {
          if (ele.isNew !== undefined && ele.isNew) {
            param.fileList.push({ name: ele.name, path: ele.path, FILE_NUMBER: i, size: ele.size })
          }
        })
        TabDataSetApi._setUpdateDataset(param)
          .then(result => {
            if (result.status === 1) {
              toast.info(<CommonToast Icon={MdSave} text={"DataSet Update Success"} />)
              setPageState(prevState => ({ ...prevState, isSave: false }))
              panelToggle()
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
        break
      case "DUPLICATION":
        data.ORG_DATASET_CD = editData.dataInfo.DATASET_CD
        TabDataSetApi._setDupDataset(data)
          .then(result => {
            if (result.status === 1) {
              toast.info(<CommonToast Icon={MdSave} text={"DataSet Duplication Success"} />)
              setPageState(prevState => ({ ...prevState, isSave: false }))
              panelToggle()
              initDataSet()
            } else {
              throw { err: "status 0" }
            }
          })
          .catch(err => {
            console.log(err)
            setPageState(prevState => ({ ...prevState, isSave: false }))
            toast.error(<CommonToast Icon={MdError} text={"DataSet Duplication Fail"} />)
          })
        break
    }
  }

  const _uploadSeperate = async files => {
    let formData = new FormData()
    let arr = []
    try {
      files.forEach(ele => {
        if (ele.status === 0 || ele.status === -1) {
          let pathName = path.join(uuid, ele.path)
          pathName = pathName.replace(/\//g, "_@_")
          pathName = pathName.replace(/\//g, "_@_")
          formData.append("file", ele, pathName)
          arr.push(ele.path)
        }
      })
      formData.append("uuid", uuid)
      formData.append("dir", JSON.stringify(arr))

      if (arr.length === 0) {
        console.log("Array is empty")
        return false
      }
      const config = {
        headers: {
          "Content-Type": "multipart/form-data"
        },
        timeout: 1000 * 200,
        cancelToken: source.token
      }

      const response = await axios.post("/api/dataset/upload", formData, config)
      const result = response.data

      result?.forEach(ele => {
        let index = fileState.fileList.findIndex(file => file.path === ele.path)
        fileState.fileList[index].status = ele.status
        fileState.fileList[index].isNew = true
        setFileState(prevState => ({
          ...prevState,
          successCount: fileState.fileList.filter(el => el.status === 1).length,
          failCount: fileState.fileList.filter(el => el.status === -1).length
        }))
      })
      setFileState(prevState => ({ ...prevState, fileList: fileState.fileList }))

      return true
    } catch (e) {
      console.log(e)
      if (e.message === "UNMOUNT") {
        return false
      } else {
        arr.forEach(ele => {
          let index = fileState.fileList.findIndex(file => file.path === ele)
          fileState.fileList[index].status = -1
          setFileState(prevState => ({ ...prevState, failCount: prevState.failCount + 1 }))
        })
        setFileState(prevState => ({ ...prevState, fileList: fileState.fileList }))
        toast.error(<CommonToast Icon={MdError} text={"Upload Fail"} />)
        return false
      }
    }
  }

  const difference = (object, base, col) => {
    function changes(object, base) {
      return transform(object, function (result, value, key) {
        if (!isEqual(value[col], base[key][col])) {
          result[key] = isObject(value[col]) && isObject(base[key][col]) ? changes(value[col], base[key][col]) : value[col]
        }
      })
    }
    return changes(object, base)
  }

  const _requestTextUpload = files => {
    return new Promise((resolve, reject) => {
      let formData = new FormData()
      let arr = []
      files.forEach(ele => {
        if (ele.status === 0 || ele.status === -1) {
          let pathName = path.join(uuid, ele.path)
          pathName = pathName.replace(/\//g, "_@_")
          pathName = pathName.replace(/\//g, "_@_")
          formData.append("file", ele, pathName)
          arr.push(ele.path)
        }
      })
      formData.append("uuid", uuid)
      formData.append("dir", JSON.stringify(arr))

      if (arr.length !== 0) {
        const config = {
          headers: {
            "Content-Type": "multipart/form-data"
          },
          cancelToken: source.token
        }
        axios
          .post("/api/tab/dataset/upload", formData, config)
          .then(response => {
            let colList = new Set()
            response.data.forEach(ele => {
              let index = fileState.fileList.findIndex(file => file.path === ele.path)
              fileState.fileList[index].status = ele.status
              fileState.fileList[index].isNew = true
              fileState.fileList[index].columns = ele.columns
              ele.columns.forEach(col => colList.add(col))
              setFileState(prevState => ({
                ...prevState,
                successCount: fileState.fileList.filter(el => el.status === 1).length,
                failCount: fileState.fileList.filter(el => el.status === -1).length
              }))
            })
            setFileState(prevState => ({ ...prevState, fileList: fileState.fileList }))
            resolve(Array.from(colList))
          })
          .catch(e => {
            console.log(e)
            if (e.message === "UNMOUNT") {
              reject(e)
            } else {
              arr.forEach(ele => {
                let index = fileState.fileList.findIndex(file => file.path === ele)
                fileState.fileList[index].status = -1
                setFileState(prevState => ({ ...prevState, failCount: prevState.failCount + 1 }))
              })
              setFileState(prevState => ({ ...prevState, fileList: fileState.fileList }))
              toast.error(<CommonToast Icon={MdError} text={"Upload Fail"} />)
              reject(e)
            }
          })
      }
    })
  }

  const _uploadSeperateText = async () => {
    const files = fileState.fileList.filter(ele => ele.status !== 1)
    const uploadFiles = cloneDeep(files)
    let split = files.length < 50 ? 1 : 50
    const promiseList = []
    while (split <= files.length) {
      let splited = files.splice(0, split)
      const promises = await _requestTextUpload(splited)
      promiseList.push(promises)
    }

    if (files.length !== 0) {
      const p = await _requestTextUpload(files)
      promiseList.push(p)
    }

    Promise.all(promiseList)
      .then(result => {
        const currentCol = new Set()
        const totalCol = new Set()
        fileState.fileList.forEach(el => {
          el.columns?.forEach(col => {
            totalCol.add(col)
          })
        })

        result.forEach(ele => {
          ele.forEach(col => currentCol.add(col))
        })

        const totalColList = Array.from(totalCol).map(el => ({ COLUMN_NM: el, DEFAULT_VALUE: "null" }))
        const curColList = Array.from(currentCol).map(el => ({ COLUMN_NM: el, DEFAULT_VALUE: "null" }))
        totalColList.forEach(el => {
          const idx = fileState.colList.findIndex(col => col.COLUMN_NM === el.COLUMN_NM)
          if (idx !== -1) el.DEFAULT_VALUE = fileState.colList[idx].DEFAULT_VALUE
        })
        if (totalCol.size !== 0 && currentCol.size !== totalCol.size) {
          const diff = difference(curColList, totalColList, "COLUMN_NM").filter(cols => cols !== null)
          if (diff.length !== 0) {
            confirmAlert({
              customUI: ({ onClose }) => {
                return (
                  <CustomColumnAlert
                    onClose={onClose}
                    onCancel={() => {
                      const param = {
                        uuid: uuid,
                        fileList: uploadFiles
                      }
                      DataSetApi._removeTempFiles(param)
                        .then(() => {
                          const filter = fileState.fileList.filter(el => {
                            return !uploadFiles.some(f => f.path === el.path)
                          })
                          setFileState(prevState => ({ ...prevState, fileList: filter }))
                        })
                        .catch(e => {
                          console.log(e)
                        })
                    }}
                    colList={totalColList}
                    setFileState={setFileState}
                    diffCols={JSON.stringify(diff)}
                  />
                )
              }
            })
          } else {
            setFileState(prevState => ({ ...prevState, colList: totalColList }))
          }
        } else {
          setFileState(prevState => ({ ...prevState, colList: totalColList }))
        }
        return true
      })
      .catch(e => {
        console.log(e)
        return false
      })
  }

  const upload = async () => {
    setPageState(prevState => ({ ...prevState, isUpload: true }))
    const files = fileState.fileList.filter(ele => ele.status !== 1)
    const fileLength = files.length
    let split = files.length < 1000 ? 1 : 100

    if (typeState.dataType === "T") {
      await _uploadSeperateText()
    } else {
      while (split <= files.length) {
        let retryCnt = 10
        let splited = files.splice(0, split)
        // console.log(`${files.length}/${fileLength} left`)
        let f = await _uploadSeperate(splited)
        while (f === false && retryCnt > 0) {
          retryCnt -= 1
          console.log(`Upload failed. Remain retry count: ${retryCnt}`)
          f = await _uploadSeperate(splited)
        }
      }
      if (files.length !== 0) {
        // console.log(`${files.length}/${fileLength} left (last)`)
        let f = await _uploadSeperate(files)
        // console.log(`result: ${f}`)
      }
    }
    setPageState(prevState => ({ ...prevState, isUpload: false }))
  }

  const changeLabel = newTagName => {
    if (fileState.selectedFiles.length === 0) return
    let cFileList = cloneDeep(fileState.fileList)

    fileState.selectedFiles.forEach((selected, i) => {
      if (selected) {
        cFileList[i].base = newTagName
      }
    })
    setFileState(prevState => ({ ...prevState, fileList: cFileList, selectedFiles: [], checkAll: false }))
    toggle()
  }

  const removeList = uploadType => () => {
    if (uploadType === "FILE") {
      if (fileState.selectedFiles.length === 0) return
      const cFileList = []
      const newRemoveList = []
      fileState.selectedFiles.forEach((selected, i) => {
        if (selected) {
          // add to remove list
          if (!fileState.fileList[i].isNew) {
            fileState.removeFiles.push(fileState.fileList[i].name)
          } else {
            newRemoveList.push(fileState.fileList[i])
          }

          if (fileState.fileList[i].status === 1) {
            setFileState(prevState => ({ ...prevState, successCount: prevState.successCount - 1 }))
          } else if (fileState.fileList[i].status === -1) {
            setFileState(prevState => ({ ...prevState, failCount: prevState.failCount - 1 }))
          }
        } else cFileList.push(fileState.fileList[i])
      })

      // "text" case : change column list
      if (typeState.dataType === "T") {
        const set = new Set()
        cFileList.forEach(el => {
          el.columns.forEach(col => {
            set.add(col)
          })
        })
        const colList = Array.from(set).map(el => ({ COLUMN_NM: el, DEFAULT_VALUE: "null" }))
        colList.forEach(el => {
          const idx = fileState.colList.findIndex(col => col.COLUMN_NM === el.COLUMN_NM)
          if (idx !== -1) el.DEFAULT_VALUE = fileState.colList[idx].DEFAULT_VALUE
        })
        setFileState(prevState => ({ ...prevState, colList: colList }))
      }

      // "new upload" case : api call for temp file remove
      if (newRemoveList.length !== 0) {
        const param = {
          uuid: uuid,
          fileList: newRemoveList
        }
        DataSetApi._removeTempFiles(param)
          .then(() => {})
          .catch(e => {
            console.log(e)
          })
      }

      setFileState(prevState => ({
        ...prevState,
        fileList: cFileList,
        removeFiles: fileState.removeFiles,
        selectedFiles: [],
        totalCount: cFileList.length,
        checkAll: false
      }))
    } else {
      setTableState({
        tableData: [],
        tableInfo: {},
        colList: [],
        columns: []
      })
    }
  }

  const _onRowMouseOver =
    col =>
    ({ index }) => {
      setPageState(prevState => ({ ...prevState, [col]: index }))
    }

  const _onRowMouseOut = col => () => {
    setPageState(prevState => ({ ...prevState, [col]: null }))
  }

  const _onRowClick =
    selected =>
    ({ index }) => {
      let temp = cloneDeep(fileState[selected])
      temp[index] = !temp[index]
      setFileState(prevState => ({ ...prevState, [selected]: temp }))
    }

  const _rowStyle =
    (hover, selected) =>
    ({ index }) => {
      if (index < 0) return
      if (pageState[hover] === index) {
        return { backgroundColor: "#2f2f2f" }
      }
      if (fileState[selected][index]) {
        return { backgroundColor: "#015aa7af", color: "white" }
      }
      return
    }

  const handleCheckAllFiles = () => {
    let flag = cloneDeep(fileState.checkAllFiles)
    let temp = cloneDeep(fileState.selectedFiles)
    if (flag) {
      temp.fill(false)
      setFileState(prevState => ({ ...prevState, selectedFiles: temp, checkAllFiles: false }))
    } else {
      temp.fill(true)
      setFileState(prevState => ({ ...prevState, selectedFiles: temp, checkAllFiles: true }))
    }
  }

  const columns = [
    {
      label: "-",
      dataKey: "",
      className: "text-center",
      disableSort: true,
      width: 60,
      hide: pageState.pageMode === "DUPLICATION" ? true : false,
      cellRenderer: ({ rowIndex }) => (
        <div>
          <input type="checkbox" name="select" checked={!!fileState.selectedFiles[rowIndex]} readOnly />
        </div>
      ),
      headerRenderer: () => (
        <div>
          <input type="checkbox" name="select-all" checked={!!fileState.checkAllFiles} onClick={handleCheckAllFiles} readOnly />
        </div>
      )
    },
    {
      label: "#",
      width: 60,
      className: "text-center",
      disableSort: true,
      dataKey: "index",
      cellRenderer: ({ rowIndex }) => rowIndex + 1
    },
    {
      label: "Label",
      dataKey: "base",
      className: "text-left",
      disableSort: true,
      width: 120
    },
    {
      label: "File Name",
      dataKey: "name",
      className: "text-left",
      disableSort: true,
      width: 400
    },
    {
      label: "Size",
      dataKey: "size",
      className: "text-left",
      disableSort: true,
      width: 150,
      cellRenderer: ({ rowData }) => bytesToSize(rowData.size)
    },
    {
      label: "Type",
      dataKey: "type",
      className: "text-center",
      disableSort: true,
      width: 200
    },
    {
      label: "Status",
      dataKey: "status",
      className: "text-center",
      disableSort: true,
      width: 80,
      cellRenderer: ({ cellData }) => {
        switch (cellData) {
          case 0:
            return <MdPauseCircleOutline className="material-icons" />
          case 1:
            return <MdCheckCircle className="material-icons font-green" />
          case -1:
            return <MdError className="material-icons font-red" />
        }
      }
    }
  ]

  const handleClassClick = e => {
    const className = e.target.textContent
    const cSelectedFiles = cloneDeep(fileState.selectedFiles)
    fileState.fileList.forEach((file, i) => {
      if (file.base === className) {
        cSelectedFiles[i] = !cSelectedFiles[i]
      }
    })
    setFileState(prevState => ({ ...prevState, selectedFiles: cSelectedFiles }))
  }

  const uploadToggle = () => {
    setPageState(prevState => ({ ...prevState, uploadModal: !prevState.uploadModal }))
  }

  const columnToggle = () => {
    setPageState(prevState => ({ ...prevState, columnModal: !prevState.columnModal }))
  }

  const dbToggle = () => {
    setPageState(prevState => ({ ...prevState, dbModal: !prevState.dbModal }))
  }

  const dbColumnToggle = () => {
    setPageState(prevState => ({ ...prevState, dbColumnModal: !prevState.dbColumnModal }))
  }

  const top = (
    <div className="form pr-2">
      <FormText
        title="Title"
        titleClassName={"mr-4 mt-2"}
        name="TITLE"
        register={register({
          required: true,
          validate: {
            validateTrim: value => String(value).trim().length !== 0
          }
        })}
        errors={errors}
      />

      <FormIcon
        title="Data Type"
        titleClassName={"mr-4 mt-2"}
        name="DATA_TYPE"
        iconList={dataTypeList}
        register={register}
        setValue={setValue}
        watch={watch}
        typeState={typeState}
        setTypeState={setTypeState}
        typeName={"dataType"}
        defaultValue={"I"}
      />
      <FormIcon
        title="Purpose"
        titleClassName={"mr-4 mt-2"}
        name="OBJECT_TYPE"
        iconList={purposeList}
        register={register}
        setValue={setValue}
        watch={watch}
        typeState={typeState}
        setTypeState={setTypeState}
        typeName={"objectType"}
        defaultValue={"C"}
      />
      <FormTextArea title="Details" titleClassName={"mr-4 mt-2"} name="DESC_TXT" register={register} />
      {typeState.objectType !== "C" && (
        <FormSelect
          title="Import Type"
          titleClassName={"mr-4 mt-2"}
          name="IMPORT_TYPE"
          control={control}
          options={importOptions}
          onChange={([selected]) => {
            setTypeState(oldState => ({ ...oldState, importType: selected }))
            return selected
          }}
          isDefault={true}
          defaultValue={"N"}
          disabled={pageState.pageMode !== "NEW"}
        />
      )}

      {typeState.dataType === "T" && (
        <>
          <FormSelect
            inputRef={resultRef}
            title="Result Target"
            titleClassName={"mr-4 mt-2"}
            name="TARGET"
            control={control}
            options={
              uploadType === "FILE"
                ? fileState.colList.map(el => ({ value: el.COLUMN_NM, label: el.COLUMN_NM }))
                : tableState.colList.map(el => ({ value: el.COLUMN_NM, label: el.COLUMN_NM }))
            }
            rules={{ required: true }}
            onFocus={() => {
              if (errors.TARGET?.type === "required") resultRef.current.disabled = false
              return resultRef.current.focus()
            }}
          />
          {errors.TARGET?.type === "required" && <div className="form__form-group-label form-error mt-1">Result Target is Required</div>}
        </>
      )}

      {typeState.dataType !== "T" && typeState.objectType !== "C" && typeState.importType === "N" && (
        <>
          <FormSelect
            title="Auto Labeling"
            titleClassName={"mr-4 mt-2"}
            name="AUTO_TYPE"
            control={control}
            onChange={([selected]) => {
              if (selected === "N") setOptionState(prevState => ({ ...prevState, modelEpochs: [] }))
              return selected
            }}
            isDefault={true}
            disabled={pageState.pageMode == "DUPLICATION" ? true : false}
          />
          {watchAutoType === "Y" && (
            <>
              <FormSelect
                title="Base Model"
                titleClassName={"mr-4 mt-2"}
                name="AUTO_MODEL"
                control={control}
                options={optionState.filteredModelOption}
                isDefault={true}
                onChange={([selected]) => {
                  return selected
                }}
              />
              {typeState.objectType !== "S" && (
                <>
                  <FormNumber
                    title="Accuracy"
                    titleClassName={"mr-4 mt-2"}
                    name="AUTO_ACC"
                    register={register({ required: watchAutoType === "Y", min: 0, max: 100 })}
                    placeholder={"0~100"}
                  />
                  {(errors.AUTO_ACC?.type === "max" || errors.AUTO_ACC?.type === "min") && (
                    <div className="form__form-group-label form-error mt-1">Please enter only numbers between 0 and 100</div>
                  )}
                  {errors.AUTO_ACC?.type === "required" && (
                    <div className="form__form-group-label form-error mt-1">Accuracy is Required</div>
                  )}
                </>
              )}
            </>
          )}
        </>
      )}
    </div>
  )

  const bottom = (
    <StyledLoader active={!isPanelRender} classNamePrefix="MyLoader_" spinner={<BeatLoader color="#4277ff" size={10} />}>
      {isPanelRender && (
        <>
          {typeState.dataType === "T" && (
            <div className="form pr-2">
              <FormSelect
                title="Model Upload"
                titleClassName={"mr-4 mt-2"}
                name="UPLOAD_TYPE"
                control={control}
                onChange={([selected]) => {
                  setUploadType(selected)
                  return selected
                }}
                isDefault={true}
              />
            </div>
          )}
          <Row noGutters>
            <Col xl={8}>{/* <h5 className="mt-1 ml-1 float-left">Data File Upload</h5> */}</Col>
            <Col xl={4}>
              <div className="float-right">
                {uploadType === "FILE" && (
                  <div className="mt-2 font-white" style={{ fontSize: "10px" }}>
                    &nbsp;Total : <span className="mr-1">{fileState.totalCount} /</span>
                    &nbsp;Success : <span className="mr-1 font-green">{fileState.successCount} </span>/ &nbsp;Fail :&nbsp;
                    <span className="font-red">{fileState.failCount}</span>
                  </div>
                )}
              </div>
            </Col>
          </Row>
          <Row noGutters>
            <Col xl={9}>
              {typeState.dataType !== "T" && typeState.objectType === "C" && optionState.tagOption.length !== 0 && (
                <div className="dataset-tag-list">
                  {optionState.tagOption.map((tag, i) => (
                    <div key={i} className={"dataset-tag"}>
                      <div className={"dataset-tag-icon"}>
                        <RiPriceTag3Fill style={{ fontSize: "14px", fill: tag.color }} className="icon-pointer" />
                      </div>
                      <div className={"dataset-tag-text"} onClick={handleClassClick}>
                        {tag.label}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </Col>
            <Col xl={3}>
              <div className="float-right mt-1">
                <TableIcon
                  ButtonIcon={RiPriceTag3Fill}
                  tooltip={"Label"}
                  onClick={toggle}
                  disabled={
                    pageState.pageMode === "DUPLICATION" ||
                    typeState.objectType !== "C" ||
                    typeState.dataType === "T" ||
                    fileState.selectedFiles.filter(el => el).length === 0
                  }
                />
                {uploadType === "FILE" ? (
                  <TableIcon ButtonIcon={FaUpload} tooltip={"Upload"} onClick={uploadToggle} />
                ) : (
                  <TableIcon ButtonIcon={FaDatabase} tooltip={"Database"} onClick={dbToggle} />
                )}
                <TableIcon
                  ButtonIcon={FaColumns}
                  tooltip={"Column"}
                  onClick={uploadType === "FILE" ? columnToggle : dbColumnToggle}
                  disabled={typeState.dataType !== "T"}
                />
                <TableIcon ButtonIcon={FaTrashAlt} tooltip={"Remove"} onClick={removeList(uploadType)} />
              </div>
            </Col>
          </Row>
          <LoadingOverlay
            active={pageState.isUpload}
            spinner={
              <div style={{ width: "80px", height: "80px" }}>
                <CircularProgressbar
                  value={Math.floor((fileState.successCount / fileState.totalCount) * 100)}
                  minValue={0}
                  maxValue={100}
                  text={`${Math.floor((fileState.successCount / fileState.totalCount) * 100)}%`}
                  strokeWidth={5}
                  styles={buildStyles({
                    pathColor: "#fff",
                    textColor: "#fff",
                    trailColor: "#565c61",
                    strokeLinecap: "butt"
                  })}
                />
              </div>
            }
          >
            <VirtualTable
              className="vt-table text-break-word"
              rowClassName="vt-header"
              height={`${panelPos.bottom - 144}px`}
              width={uploadType === "FILE" ? undefined : tableState?.columns?.length * 120}
              headerHeight={40}
              rowHeight={50}
              columns={uploadType === "FILE" ? columns : tableState.columns}
              data={uploadType === "FILE" ? fileState.fileList : tableState.tableData}
              scrollIndex={fileState.successCount}
              onRowMouseOver={_onRowMouseOver("fileHoverIndex")}
              onRowMouseOut={_onRowMouseOut("fileHoverIndex")}
              onRowClick={_onRowClick("selectedFiles")}
              rowStyle={_rowStyle("fileHoverIndex", "selectedFiles")}
              style={{ overflowX: "scroll", overflowY: "hidden" }}
            />
          </LoadingOverlay>
        </>
      )}
    </StyledLoader>
  )

  const tail = (
    <>
      <div className="line-separator mx-2 mt-2" />
      <CommonButton
        ButtonIcon={RiSendPlaneFill}
        className="bg-green float-right"
        text="Apply"
        onClick={typeState.dataType === "T" ? handleSubmit(onTextSubmit) : handleSubmit(onSubmit)}
        disabled={pageState.isUpload}
      />
    </>
  )

  return (
    <>
      {pageState.modal && <LabelModal modal={pageState.modal} toggle={toggle} changeLabel={changeLabel} />}
      {pageState.uploadModal && (
        <UploadModal
          modal={pageState.uploadModal}
          toggle={uploadToggle}
          control={control}
          pageState={pageState}
          setPageState={setPageState}
          typeState={typeState}
          fileState={fileState}
          optionState={optionState}
          setFileState={setFileState}
          errors={errors}
          maxFiles={dataSetUpload[typeState.dataType].COUNT}
        />
      )}
      {pageState.columnModal && (
        <ColumnModal modal={pageState.columnModal} toggle={columnToggle} fileState={fileState} setFileState={setFileState} />
      )}

      {pageState.dbModal && <DbModal modal={pageState.dbModal} toggle={dbToggle} setTableState={setTableState} tableState={tableState} />}

      {pageState.dbColumnModal && (
        <ColumnModal modal={pageState.dbColumnModal} toggle={dbColumnToggle} tableState={tableState} setTableState={setTableState} />
      )}

      <CommonPanel
        title={pageState.title}
        isSave={pageState.isSave}
        loadingText="Create DataSet ..."
        panelToggle={panelToggle}
        springProps={springProps}
        panelPos={panelPos}
        setPanelPos={setPanelPos}
        top={top}
        bottom={bottom}
        tail={tail}
      />
    </>
  )
}

NewDataSetPanel.propTypes = {}

export default NewDataSetPanel
