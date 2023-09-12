// react-libraries
import React, { useState, useEffect, useCallback, useRef } from "react"
import PropTypes from "prop-types"
import { Container } from "reactstrap"
import { confirmAlert } from "react-confirm-alert"
import { toast } from "react-toastify"
import { Transition } from "react-spring/renderprops"

// react-icons
import { FiCopy, FiEdit, FiImage, FiRepeat, FiTrash2, FiCheck, FiPieChart, FiExternalLink } from "react-icons/fi"
import { MdDeleteForever, MdError } from "react-icons/md"
import { BiError } from "react-icons/bi"
import { FaPlusCircle } from "react-icons/fa"
import { MedicalOutline, AlertCircleOutline } from "react-ionicons"

// apis
import * as DataSetApi from "../../../Config/Services/DataSetApi"

// custom components
import Header2 from "../../../Components/Common/Header2"
import CommonToast from "../../../Components/Common/CommonToast"
import CommonButton from "../../../Components/Common/CommonButton"
import useEnterpriseDivision from "../../../Components/Utils/useEnterpriseDivision"
import VirtualList from "../../../Components/Common/VirtualList"
import NewDataSetPanel from "./NewDataSetPanel"
import Filter from "../../../Components/Filter/Filter"
import useListHeight from "../../../Components/Utils/useListHeight"

const statusList = {
  CT: [
    {
      items: [
        {
          title: "STATUS",
          key: "DATASET_STS",
          mapper: {
            DONE: "Ready",
            CREATE: "Creating",
            CREATING: "Creating",
            CRN_FAIL: "Create Fail",
            AUTO: "labeling",
            AUTO_FAIL: "Labeling Fail"
          },
          icons: {
            DONE: { icon: <FiCheck />, color: "yellowgreen" },
            CREATE: { icon: <MedicalOutline rotate color={"#00bfff"} width="14px" height="14px" />, color: "#00bfff" },
            CRN_FAIL: { icon: <AlertCircleOutline beat color={"red"} width="14px" height="14px" />, color: "red" },
            AUTO: { icon: <MedicalOutline rotate color={"#00bfff"} width="14px" height="14px" />, color: "#00bfff" },
            AUTO_FAIL: { icon: <AlertCircleOutline beat color={"red"} width="14px" height="14px" />, color: "red" }
          }
        }
      ]
    },
    {
      items: [{ title: "TARGET", key: "ROW_TARGET" }]
    },
    {
      items: [{ title: "FILES", key: "FILE_COUNT" }]
    },
    {
      items: [{ title: "SIZE", key: "DATA_SIZE", prettyBytes: true }]
    },
    {
      items: [{ title: "CLASS", key: "T_CLASS_COUNT" }]
    },
    {
      items: [{ title: "COLUMNS", key: "COL_CNT" }]
    },
    {
      items: [{ title: "RECORDS", key: "ROW_CNT" }]
    }
  ],
  C: [
    {
      items: [
        {
          title: "STATUS",
          key: "DATASET_STS",
          mapper: {
            DONE: "Ready",
            CREATE: "Creating",
            CREATING: "Creating",
            CRN_FAIL: "Create Fail",
            AUTO: "labeling",
            AUTO_FAIL: "Labeling Fail"
          },
          icons: {
            DONE: { icon: <FiCheck />, color: "yellowgreen" },
            CREATE: { icon: <MedicalOutline rotate color={"#00bfff"} width="14px" height="14px" />, color: "#00bfff" },
            CRN_FAIL: { icon: <AlertCircleOutline beat color={"red"} width="14px" height="14px" />, color: "red" },
            AUTO: { icon: <MedicalOutline rotate color={"#00bfff"} width="14px" height="14px" />, color: "#00bfff" },
            AUTO_FAIL: { icon: <AlertCircleOutline beat color={"red"} width="14px" height="14px" />, color: "red" }
          }
        }
      ]
    },
    {
      items: [{ title: "FILES", key: "FILE_COUNT" }]
    },
    {
      items: [{ title: "SIZE", key: "DATA_SIZE", prettyBytes: true }]
    },
    {
      items: [{ title: "CLASS", key: "CLASS_COUNT" }]
    }
  ],
  D: [
    {
      items: [
        {
          title: "STATUS",
          key: "DATASET_STS",
          mapper: {
            DONE: "Ready",
            CREATE: "Creating",
            CREATING: "Creating",
            CRN_FAIL: "Create Fail",
            AUTO: "labeling",
            AUTO_FAIL: "Labeling Fail"
          },
          icons: {
            DONE: { icon: <FiCheck />, color: "yellowgreen" },
            CREATE: { icon: <MedicalOutline rotate color={"#00bfff"} width="14px" height="14px" />, color: "#00bfff" },
            CRN_FAIL: { icon: <AlertCircleOutline beat color={"red"} width="14px" height="14px" />, color: "red" },
            AUTO: { icon: <MedicalOutline rotate color={"#00bfff"} width="14px" height="14px" />, color: "#00bfff" },
            AUTO_FAIL: { icon: <AlertCircleOutline beat color={"red"} width="14px" height="14px" />, color: "red" }
          }
        }
      ]
    },
    {
      items: [{ title: "FILES", key: "FILE_COUNT" }]
    },
    {
      items: [{ title: "SIZE", key: "DATA_SIZE", prettyBytes: true }]
    },
    {
      items: [{ title: "CLASS", key: "CLASS_COUNT" }]
    },
    {
      items: [{ title: "TAGS", key: "TAGS" }]
    },
    {
      items: [{ title: "LABELED", key: "PERCENT", progress: true }]
    }
  ],
  S: [
    {
      items: [
        {
          title: "STATUS",
          key: "DATASET_STS",
          mapper: {
            DONE: "Ready",
            CREATE: "Creating",
            CREATING: "Creating",
            CRN_FAIL: "Create Fail",
            AUTO: "labeling",
            AUTO_FAIL: "Labeling Fail"
          },
          icons: {
            DONE: { icon: <FiCheck />, color: "yellowgreen" },
            CREATE: { icon: <MedicalOutline rotate color={"#00bfff"} width="14px" height="14px" />, color: "#00bfff" },
            CRN_FAIL: { icon: <AlertCircleOutline beat color={"red"} width="14px" height="14px" />, color: "red" },
            AUTO: { icon: <MedicalOutline rotate color={"#00bfff"} width="14px" height="14px" />, color: "#00bfff" },
            AUTO_FAIL: { icon: <AlertCircleOutline beat color={"red"} width="14px" height="14px" />, color: "red" }
          }
        }
      ]
    },
    {
      items: [{ title: "FILES", key: "FILE_COUNT" }]
    },
    {
      items: [{ title: "SIZE", key: "DATA_SIZE", prettyBytes: true }]
    },
    {
      items: [{ title: "CLASS", key: "CLASS_COUNT" }]
    },
    {
      items: [{ title: "TAGS", key: "TAGS" }]
    },
    {
      items: [{ title: "LABELED", key: "PERCENT", progress: true }]
    }
  ],
  R: [
    {
      items: [
        {
          title: "STATUS",
          key: "DATASET_STS",
          mapper: {
            DONE: "Ready",
            CREATE: "Creating",
            CREATING: "Creating",
            CRN_FAIL: "Create Fail",
            AUTO: "labeling",
            AUTO_FAIL: "Labeling Fail"
          },
          icons: {
            DONE: { icon: <FiCheck />, color: "yellowgreen" },
            CREATE: { icon: <MedicalOutline rotate color={"#00bfff"} width="14px" height="14px" />, color: "#00bfff" },
            CRN_FAIL: { icon: <AlertCircleOutline beat color={"red"} width="14px" height="14px" />, color: "red" },
            AUTO: { icon: <MedicalOutline rotate color={"#00bfff"} width="14px" height="14px" />, color: "#00bfff" },
            AUTO_FAIL: { icon: <AlertCircleOutline beat color={"red"} width="14px" height="14px" />, color: "red" }
          }
        }
      ]
    },
    {
      items: [{ title: "TARGET", key: "ROW_TARGET" }]
    },
    {
      items: [{ title: "FILES", key: "FILE_COUNT" }]
    },
    {
      items: [{ title: "SIZE", key: "DATA_SIZE", prettyBytes: true }]
    },
    {
      items: [{ title: "CLASS", key: "T_CLASS_COUNT" }]
    },
    {
      items: [{ title: "COLUMNS", key: "COL_CNT" }]
    },
    {
      items: [{ title: "RECORDS", key: "ROW_CNT" }]
    }
  ]
}

const sortList = {
  createdSortByNewest: { key: "CRN_DTM", direction: "DESC" },
  createdSortByOldest: { key: "CRN_DTM", direction: "ASC" },
  updatedSortByNewest: { key: "UPT_DTM", direction: "DESC" },
  updatedSortByOldest: { key: "UPT_DTM", direction: "ASC" },
  sortFileCountDESC: { key: "FILE_COUNT", direction: "DESC" },
  sortFileCountASC: { key: "FILE_COUNT", direction: "ASC" },
  sortFileSizeDESC: { key: "DATA_SIZE", direction: "DESC" },
  sortFileSizeASC: { key: "DATA_SIZE", direction: "ASC" },
  sortAZ: { key: "TITLE", direction: "ASC" },
  sortZA: { key: "TITLE", direction: "DESC" }
}

const DataSets = props => {
  const { history } = props
  const [dataSetList, setDataSetList] = useState([])
  const [filterList, setFilterList] = useState([])
  const [filterTempList, setFilterTempList] = useState(null)
  const [isPanel, setIsPanel] = useState(false)
  const [searchKey, setSearchKey] = useState(null)
  const [editData, setEditData] = useState({})
  const [isLoad, setIsLoad] = useState(false)
  const [toggleFilter, setToggleFilter] = useState(null)

  const [isPanelRender, setIsPanelRender] = useState(false)
  const confirmTextRef = useRef("")
  const dataSplitRef = useRef([70, 20, 10])
  const createDataSet = useEnterpriseDivision(process.env.BUILD, "dataSet", "createDataSet")
  const dataSetDuplication = useEnterpriseDivision(process.env.BUILD, "dataSet", "dataSetDuplication")
  const listHeight = useListHeight()

  useEffect(() => {
    initDataSet()

    const timer = setInterval(
      () =>
        DataSetApi._getDataSetList(null).then(data => {
          setDataSetList(data)
          setToggleFilter(true)
        }),
      15000
    )

    return () => {
      clearInterval(timer)
    }
  }, [])

  useEffect(() => {
    if (searchKey !== null) setToggleFilter(true)
  }, [searchKey])

  useEffect(() => {
    if (filterTempList !== null) {
      const newList = filterTempList.filter(
        dataset => dataset.TITLE.indexOf(searchKey ? searchKey : "") !== -1 || dataset.DATASET_CD.indexOf(searchKey ? searchKey : "") !== -1
      )
      setFilterList(newList)
    }
  }, [filterTempList])

  useEffect(() => {
    if (!isPanel) {
      setEditData({})
    }
  }, [isPanel])

  const initDataSet = () => {
    setIsLoad(true)
    DataSetApi._getDataSetList(null)
      .then(data => {
        setDataSetList(data)
        setFilterList(data)
        setIsLoad(false)
      })
      .catch(e => {
        console.log(e)
        setIsLoad(false)
      })
  }

  const _handleRemove = useCallback(
    data => () => {
      if (data.DATASET_STS === "DONE" || data.DATASET_STS.includes("_FAIL")) {
        confirmAlert({
          customUI: ({ onClose }) => {
            return (
              <div className="react-confirm-alert-custom">
                <h1>
                  <BiError />
                  Delete Dataset
                </h1>
                <div className="custom-modal-body">
                  <div className="text-warning">Warning. This action is irreversible.</div>
                  <div className="explain">
                    All resources of <strong>{data.TITLE}</strong> dataset will be removed.
                  </div>
                  <div>
                    Please type <strong>[ {data.TITLE} ]</strong> to avoid unexpected action.
                  </div>
                  <input
                    type="text"
                    className="react-confirm-alert-input"
                    onChange={e => {
                      confirmTextRef.current = e.target.value
                    }}
                  />
                </div>
                <div className="custom-buttons">
                  <CommonButton
                    className="bg-green"
                    text="Apply"
                    onClick={() => {
                      if (confirmTextRef.current.trim() === data.TITLE.trim()) {
                        document.getElementById("wrapper").scrollIntoView()
                        onClose()
                        DataSetApi._removeDataSet({ DATASET_CD: data.DATASET_CD, TITLE: data.TITLE })
                          .then(result => {
                            if (result.status === 1) {
                              const filter = dataSetList.filter(ele => ele.DATASET_CD !== data.DATASET_CD)
                              setDataSetList(filter)
                              const filter2 = filterList.filter(ele => ele.DATASET_CD !== data.DATASET_CD)
                              setFilterList(filter2)
                              toast.info(<CommonToast Icon={MdDeleteForever} text={"DataSet Delete Success"} />)
                            } else {
                              throw { err: "status 0" }
                            }
                          })
                          .catch(err => {
                            toast.error(<CommonToast Icon={MdError} text={"DataSet Delete Fail"} />)
                            console.log(err)
                          })
                      } else alert("Not matched.")
                    }}
                  />
                  <CommonButton className="bg-red" text="Cancel" onClick={onClose} />
                </div>
              </div>
            )
          }
        })
      } else if (data.DATASET_STS === "AUTO") {
        return alert("Auto labeling...")
      } else {
        return alert(`DataSet ${data.DATASET_STS}...`)
      }
    },
    [dataSetList, filterList]
  )

  const _handleDuplication = useCallback(
    data => () => {
      if (data.DATASET_STS === "DONE" || data.DATASET_STS.includes("_FAIL")) {
        setIsPanel(isPanel => !isPanel)
        setEditData({ dataInfo: data, pageMode: "DUPLICATION" })
      } else if (data.DATASET_STS === "AUTO") {
        return alert("Auto labeling...")
      } else {
        return alert(`DataSet ${data.DATASET_STS}...`)
      }
    },
    [history]
  )

  const _checkDirExist = async datasetName => {
    let result = await DataSetApi._checkDirExist({ datasetName })
    return result.result
  }

  const _deployDataSet = (type, split, datasetName, datasetCd) => {
    DataSetApi._deployDataSet({ type, split, datasetName, datasetCd })
      .then(result => {
        return result.result
      })
      .catch(err => {
        toast.error(<CommonToast Icon={MdError} text={"Deploy dataset failed"} />)
        console.log(err.result)
      })
  }

  const _removeDeployedDataSet = datasetName => {
    console.log(`remove ${datasetName}`)
    DataSetApi._removeDeployedDataSet({ datasetName })
      .then(result => {
        return result.result
      })
      .catch(err => {
        toast.error(<CommonToast Icon={MdError} text={"Remove deployed dataset failed"} />)
        console.log(err.result)
      })
  }

  const _handleDeployment = useCallback(
    data => async () => {
      if (data.DATASET_STS === "DONE" || data.DATASET_STS.includes("_FAIL")) {
        let datasetName = data.TITLE.replace(/[^\w\d\uAC00-\uD7AF]/g, "_")
        let datasetPath = "/shared/datasets/" + datasetName
        let checkDirExist = await _checkDirExist(datasetName)
        confirmAlert({
          closeOnClickOutside: false,
          customUI: ({ onClose }) => {
            return (
              <div className="react-confirm-alert-custom">
                <h1>
                  <FiExternalLink />
                  {checkDirExist ? "Remove deployed DataSet" : data.OBJECT_TYPE === "D" ? "Deploy DataSet (YoloV7)" : "Deploy DataSet"}
                </h1>

                <div className="custom-modal-body">
                  {checkDirExist && <div className="text-warning">Warning. This action is irreversible.</div>}
                  <div className="text-info">Path: {datasetPath}</div>
                  {data.OBJECT_TYPE === "D" && !checkDirExist && (
                    <div>
                      <div className="react-confirm-alert-input-split">
                        <div className="text">Train</div>
                        <input
                          type="text"
                          className="value"
                          onChange={e => {
                            dataSplitRef.current[0] = e.target.value
                          }}
                          defaultValue={dataSplitRef.current[0]}
                        />
                      </div>
                      <div className="react-confirm-alert-input-split">
                        <div className="text">Validation</div>
                        <input
                          type="text"
                          className="value"
                          onChange={e => {
                            dataSplitRef.current[1] = e.target.value
                          }}
                          defaultValue={dataSplitRef.current[1]}
                        />
                      </div>
                      <div className="react-confirm-alert-input-split">
                        <div className="text">Test</div>
                        <input
                          type="text"
                          className="value"
                          onChange={e => {
                            dataSplitRef.current[2] = e.target.value
                          }}
                          defaultValue={dataSplitRef.current[2]}
                        />
                      </div>
                    </div>
                  )}
                  <div style={{ marginTop: "10px" }}>
                    Please type <strong>[ {data.TITLE} ]</strong> to avoid unexpected action.
                  </div>
                  <input
                    type="text"
                    className="react-confirm-alert-input"
                    onChange={e => {
                      confirmTextRef.current = e.target.value
                    }}
                  />
                </div>
                <div className="custom-buttons">
                  {checkDirExist ? (
                    <CommonButton
                      className="bg-green"
                      text="Remove"
                      onClick={() => {
                        if (confirmTextRef.current.trim() === data.TITLE.trim()) {
                          _removeDeployedDataSet(datasetName)
                          onClose()
                        } else alert("Not matched.")
                      }}
                    />
                  ) : (
                    <CommonButton
                      className="bg-green"
                      text="Deploy"
                      onClick={() => {
                        if (confirmTextRef.current.trim() === data.TITLE.trim()) {
                          if (
                            parseInt(dataSplitRef.current[0]) + parseInt(dataSplitRef.current[1]) + parseInt(dataSplitRef.current[2]) ===
                            100
                          ) {
                            _deployDataSet(data.OBJECT_TYPE, dataSplitRef.current, datasetName, data.DATASET_CD)
                            onClose()
                          } else {
                            alert("The sum of train + validation + test should be 100.")
                          }
                        } else alert("Not matched.")
                      }}
                    />
                  )}
                  <CommonButton className="bg-red" text="Cancel" onClick={onClose} />
                </div>
              </div>
            )
          }
        })
      } else if (data.DATASET_STS === "AUTO") {
        return alert("Auto labeling...")
      } else {
        return alert(`DataSet ${data.DATASET_STS}...`)
      }
    },
    [history]
  )

  const _handleEdit = useCallback(
    data => () => {
      if (data.DATASET_STS === "DONE" || data.DATASET_STS.includes("_FAIL")) {
        setIsPanel(isPanel => !isPanel)
        setEditData({ dataInfo: data, pageMode: "EDIT" })
      } else if (data.DATASET_STS === "AUTO") {
        return alert("Auto labeling...")
      } else {
        return alert(`DataSet ${data.DATASET_STS}...`)
      }
    },
    [history]
  )

  const _handleLabel = useCallback(
    data => () => {
      if (data.DATASET_STS === "DONE" || data.DATASET_STS.includes("_FAIL")) {
        return history.push({
          pathname: "/label",
          state: { dataInfo: data }
        })
      } else if (data.DATASET_STS === "AUTO") {
        return alert("Auto labeling...")
      } else {
        return alert(`DataSet ${data.DATASET_STS}...`)
      }
    },
    [history]
  )

  const _handleAnalysis = useCallback(
    data => () => {
      return history.push({
        pathname: "/datasetAnalytics",
        state: { dataInfo: data }
      })
    },
    [history]
  )
  const _handleRetry = useCallback(
    data => () => {
      confirmAlert({
        customUI: ({ onClose }) => {
          return (
            <div className="react-confirm-alert-custom">
              <h1>
                <BiError />
                Auto Labeling
              </h1>
              <div className="custom-modal-body">
                <div className="text-warning">Warning. This action is irreversible.</div>
                <div className="explain">Auto Labeling option is Yes</div>
                <div className="explain">Saved labeling information is deleted</div>
                <div className="explain">
                  Please type <strong>[ {data.TITLE} ]</strong> to avoid unexpected action.
                </div>
                <input
                  type="text"
                  className="react-confirm-alert-input"
                  onChange={e => {
                    confirmTextRef.current = e.target.value
                  }}
                />
              </div>
              <div className="custom-buttons">
                <CommonButton
                  className="bg-green"
                  text="Apply"
                  onClick={() => {
                    if (confirmTextRef.current.trim() === data.TITLE.trim()) {
                      document.getElementById("wrapper").scrollIntoView()
                      onClose()
                      DataSetApi._autoLabeling({ DATASET_CD: data.DATASET_CD })
                        .then(() => {
                          DataSetApi._getDataSetList(null).then(data => {
                            setDataSetList(data)
                            setToggleFilter(true)
                          })
                        })
                        .catch(e => console.log(e))
                    } else alert("Not matched.")
                  }}
                />
                <CommonButton className="bg-red" text="Cancel" onClick={onClose} />
              </div>
            </div>
          )
        }
      })
    },
    []
  )

  const getCardFunc = data => {
    let arr = []
    if (data.DATASET_STS === "AUTO_FAIL") {
      arr.push({
        func: _handleRetry(data),
        label: "Auto Labeling",
        icon: <FiRepeat />
      })
    }
    if (data.OBJECT_TYPE === "D" || data.OBJECT_TYPE === "S") {
      arr.push({
        func: _handleLabel(data),
        label: "Label",
        icon: <FiImage />
      })
    }
    if (data.DATA_TYPE === "T") {
      arr.push({
        func: _handleAnalysis(data),
        label: "Analytics",
        icon: <FiPieChart />
      })
    }
    arr.push({
      func: _handleEdit(data),
      label: "Edit",
      icon: <FiEdit />
    })
    if (dataSetDuplication) {
      arr.push({
        func: _handleDuplication(data),
        label: "Duplication",
        icon: <FiCopy />
      })
    }
    arr.push({
      func: _handleDeployment(data),
      label: "Deployment",
      icon: <FiExternalLink />
    })
    arr.push({
      func: _handleRemove(data),
      label: "Delete",
      icon: <FiTrash2 />
    })

    return arr
  }

  const panelToggle = () => {
    if (dataSetList.length < createDataSet) {
      setIsPanel(isPanel => !isPanel)
    } else {
      toast.warn(<CommonToast Icon={MdError} text={`The maximum number of dataSet generated in the CE version is ${createDataSet}`} />)
    }
  }

  return (
    <Container>
      <Transition
        unique
        reset
        items={isPanel}
        from={{ transform: "translate3d(100%,0,0)" }}
        enter={{ transform: "translate3d(0%,0,0)" }}
        leave={{ transform: "translate3d(100%,0,0)" }}
        onDestroyed={isEnd => {
          if (!isEnd) setIsPanelRender(true)
          else setIsPanelRender(false)
        }}
      >
        {isPanel =>
          isPanel &&
          (springProps => (
            <NewDataSetPanel
              springProps={springProps}
              isPanelRender={isPanelRender}
              modal={isPanel}
              panelToggle={panelToggle}
              editData={editData}
              initDataSet={initDataSet}
            />
          ))
        }
      </Transition>
      <Header2
        title="DataSet"
        buttons={[
          {
            btnTitle: "Add New Dataset",
            btnIcon: FaPlusCircle,
            color: "rgb(98, 181, 86)",
            onClick: () => {
              setEditData({ dataInfo: {}, pageMode: "NEW" })
              panelToggle()
            }
          }
        ]}
        onSearch={setSearchKey}
      />

      <Filter
        filterType="DATASET"
        list={dataSetList}
        filterList={filterTempList}
        setList={setFilterTempList}
        sortList={sortList}
        setToggleFilter={setToggleFilter}
        toggleFilter={toggleFilter}
        listCount={filterList.length}
      />

      <VirtualList
        data={filterList}
        height={listHeight}
        funcList={getCardFunc}
        searchKey={searchKey}
        type="DATASET"
        status={statusList}
        isLoad={isLoad}
      />
    </Container>
  )
}

DataSets.propTypes = {
  history: PropTypes.object.isRequired
}

export default DataSets
