import React, { useEffect, useState } from "react"
import { Col, Container, Row } from "reactstrap"
import moment from "moment"

import ActiveCount from "Components/DashBoard/ActiveCount"
import GpuRate from "Components/DashBoard/GpuRate"
import SystemResources from "Components/DashBoard/SystemResources"
import DBNumber from "Components/DashBoard/DBNumber"
import DBTime from "Components/DashBoard/DBTime"
import ActiveTime from "Components/DashBoard/ActiveTime"

import * as DashBoardApi from "Config/Services/DashBoardApi"
import TreeMap from "../../../Components/DashBoard/TreeMap"

const ZeroPush = (arr, key, value) => {
  if (value !== 0 && value !== null) {
    arr.push({ name: key, value: value })
  }
  return arr
}

const Dashboard = () => {
  const [aiInfo, setAiInfo] = useState([])
  const [modelTimeInfo, setModelTimeInfo] = useState([])
  const [modelActiveTimeInfo, setModelActiveTimeInfo] = useState([])
  const [resourceInfo, setResource] = useState({})
  const [datasetInfo, setDatasetInfo] = useState([])
  const [modelInfo, setModelInfo] = useState([])
  const [sourceInfo, setSourceInfo] = useState([])
  const [model, setModel] = useState([])
  const [source, setSource] = useState([])
  const [gpuInfo, setgpuInfo] = useState([])
  const [curGpuInfo, setCurGpuInfo] = useState([])
  const [sourceTreeMap, setSourceTreeMap] = useState([])

  const barColor = ["#0061D1", "#3580D7", "#5E8DC3", "#87ABD5"]

  useEffect(() => {
    getDashBoardInfo()
    const timer = setInterval(() => {
      getDashBoardInfo()
    }, 60000)

    return () => {
      clearInterval(timer)
    }
  }, [])

  const getDashBoardInfo = () => {
    DashBoardApi._getAiInfo(null)
      .then(result => {
        const runAi = result.AI_LIST.filter(v => v.AI_STS === "LEARN")
        const endAi = result.AI_LIST.filter(v => v.AI_STS === "DONE")
        const creAi = result.AI_LIST.filter(v => v.AI_STS === "NONE")
        setAiInfo(result)
        let info = []
        ZeroPush(info, "Tabular", result.DATASET_LIST.filter(v => v.DATA_TYPE === "T").length)
        ZeroPush(info, "Image", result.DATASET_LIST.filter(v => v.DATA_TYPE === "I").length)
        ZeroPush(info, "Video", result.DATASET_LIST.filter(v => v.DATA_TYPE === "V").length)
        setDatasetInfo(info)

        info = []
        ZeroPush(info, "Regression", result.AI_LIST.filter(v => v.OBJECT_TYPE === "R").length)
        ZeroPush(info, "Classification", result.AI_LIST.filter(v => v.OBJECT_TYPE === "C").length)
        ZeroPush(info, "Detection", result.AI_LIST.filter(v => v.OBJECT_TYPE === "D").length)
        ZeroPush(info, "Segmentation", result.AI_LIST.filter(v => v.OBJECT_TYPE === "S").length)
        setModelInfo(info)
        info = []

        let sortList = result.AI_LIST.sort(function (a, b) {
          return b["TRAIN_TIME"] - a["TRAIN_TIME"]
        })
        sortList.forEach(ai => {
          if (info.length < 8) ZeroPush(info, ai.TITLE, ai.TRAIN_TIME)
        })
        setModelTimeInfo(info)

        info = []
        sortList.forEach(ai => {
          if (info.length <= 8 && ai.AI_STS === "LEARN") info.push({ name: ai.TITLE, value: ai.TRAIN_SRT_DTM })
        })

        const infoSort = info.sort((a, b) => {
          return moment(b.value, "YYYY-MM-DD HH:mm:ss").diff(moment(a.value, "YYYY-MM-DD HH:mm:ss"), "minute") * -1
        })
        setModelActiveTimeInfo(infoSort)

        info = []
        ZeroPush(info, "Real-time", result.IS_LIST.filter(v => v.IS_TYPE === "R").length)
        ZeroPush(info, "Tabular", result.IS_LIST.filter(v => v.IS_TYPE === "T").length)
        ZeroPush(info, "Image", result.IS_LIST.filter(v => v.IS_TYPE === "I").length)
        ZeroPush(info, "Video", result.IS_LIST.filter(v => v.IS_TYPE === "V").length)
        setSourceInfo(info)

        setSource([
          { value: result.IS_LIST.filter(v => v.IS_STS === "ACTIVE").length, label: "Active" },
          { value: result.IS_LIST.filter(v => v.IS_STS === "ACT_FAIL").length, label: "Active Fail" }
        ])

        setModel([
          { value: runAi.length, label: "Run" },
          { value: endAi.length, label: "Trained" },
          { value: creAi.length, label: "Ready" }
        ])
      })
      .catch(err => {
        console.log(err)
      })

    DashBoardApi._getSystemInfo(null)
      .then(result => {
        const resourceData = {
          CPU: [
            { value: Number(result.CPU_USED), fill: "#2465E3" },
            { value: 100 - result.CPU_USED, fill: "#464646" }
          ],
          CPU_USED: result.CPU_USED,
          MEM: [
            { value: Number(result.RAM_USED), fill: "#3289D4" },
            { value: 100 - result.RAM_USED, fill: "#464646" }
          ],
          MEM_USED: result.RAM_USED,
          DISK: [
            { value: Number(result.DISK_USED), fill: "#74A4CE" },
            { value: 100 - result.DISK_USED, fill: "#464646" }
          ],
          DISK_USED: result.DISK_USED
        }
        setResource(resourceData)
      })
      .catch(err => {
        console.log(err)
      })

    DashBoardApi._getGpuInfo(null)
      .then(result => {
        setgpuInfo(result)
      })
      .catch(err => {
        console.log(err)
      })

    DashBoardApi._getCurGpuInfo(null)
      .then(result => {
        setCurGpuInfo(result)
      })
      .catch(err => {
        console.log(err)
      })

    DashBoardApi._getSourceTreeMap(null)
      .then(result => {
        // result.map(ele => {
        //   if (ele.COUNT < 500) {
        //     ele.COUNT = Math.floor(Math.random() * 1300)
        //   }
        // })
        let arr = []
        result.forEach(ele => {
          let idx = arr.findIndex(item => item.isCd === ele.IS_CD)
          if (idx === -1) {
            let obj = {}
            obj.name = ele.IS_TITLE
            obj.isCd = ele.IS_CD
            let filter = result.filter(el => el.IS_CD === ele.IS_CD)
            let children = filter.map(el => ({ name: el.DP_LABEL, size: el.COUNT }))
            obj.children = children
            arr.push(obj)
          }
        })
        setSourceTreeMap(arr)
      })
      .catch(err => {
        console.log(err)
      })
  }

  return (
    <Container>
      {/* <Header title={"Welcome To BLUAI"}  /> */}

      <Row className="dashboard__row" style={{ paddingTop: "4rem" }}>
        <Col xs={12} sm={12} md={6} lg={6} xl={3}>
          <DBNumber type={"D"} title="DataSet" num={1} value={aiInfo.DATASET_LIST?.length} data={datasetInfo} />
        </Col>
        <Col xs={12} sm={12} md={6} lg={6} xl={3}>
          <DBNumber type={"T"} title="Trainer" num={3} value={aiInfo.AI_LIST?.length} data={modelInfo} />
        </Col>
        <Col xs={12} sm={12} md={6} lg={6} xl={3}>
          <DBNumber type={"S"} title="Service" num={5} value={aiInfo.IS_LIST?.length} data={sourceInfo} />
        </Col>
        <Col xs={12} sm={12} md={6} lg={6} xl={3}>
          <DBNumber type={"A"} title="Analytics" value={aiInfo.PROJECT_LIST?.length} disableClick={true} />
        </Col>
      </Row>

      <Row className="dashboard__row">
        <Col xs={12} sm={12} md={6} lg={6} xl={4}>
          <SystemResources title={"CPU / MEM / DISK"} value={resourceInfo} />
        </Col>
        <Col xs={12} sm={12} md={6} lg={6} xl={2}>
          <ActiveCount title="Active Trainer" model={model} />
        </Col>
        <Col xs={12} sm={12} md={6} lg={6} xl={3}>
          <DBTime title="Top Training Time" data={modelTimeInfo} barColor={barColor[0]} />
        </Col>
        <Col xs={12} sm={12} md={6} lg={6} xl={3}>
          <ActiveTime title="Active Trainer Time" data={modelActiveTimeInfo} barColor={barColor[3]} />
        </Col>
      </Row>

      <Row className="dashboard__row">
        <Col xs={12} sm={12} md={6} lg={6} xl={4}>
          <GpuRate gpuData={gpuInfo} curGpuData={curGpuInfo} />
        </Col>
        <Col xs={12} sm={12} md={6} lg={6} xl={2}>
          <ActiveCount title="Active Service" model={source} />
        </Col>
        <Col xs={12} sm={12} md={12} lg={12} xl={6}>
          <TreeMap title="Service Tree Map" data={sourceTreeMap} />
        </Col>
        {/* <Col xs={12} sm={12} md={6} lg={6} xl={3}>
          <DBTime title="source 부분-데이터연결필요" data={modelTimeInfo} barColor={barColor[1]} />
        </Col>
        <Col xs={12} sm={12} md={6} lg={6} xl={3}>
          <DBTime title="source 부분-데이터연결필요" data={modelTimeInfo} barColor={barColor[2]} />
        </Col> */}
      </Row>
    </Container>
  )
}

export default Dashboard
