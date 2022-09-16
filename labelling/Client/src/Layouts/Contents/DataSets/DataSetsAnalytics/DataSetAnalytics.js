import React, { useEffect, useState } from "react"
import { Col, Container, Row } from "reactstrap"
import styled from "styled-components"

import * as TabDataSetApi from "../../../../Config/Services/TabDataSetApi"

import Header2 from "../../../../Components/Common/Header2"
import DataSetAnalyticsInfo from "./DataSetAnalyticsInfo"
import DataSetAnalyticsReport from "../DataSetsAnalytics/DataSetAnalyticsReport"

const ContentWrapper = styled("div")`
  border-radius: 10px;
  width: 100%;
  background-color: black;
`

function DataSetAnalytics(props) {
  const [data, setData] = useState({})
  const [datasetInfo, setDatasetInfo] = useState({})

  useEffect(() => {
    // api call
    const datasetInfo = props.history.location.state.dataInfo
    setDatasetInfo(datasetInfo)
    const datasetCode = props.history.location.state.dataInfo.DATASET_CD
    const param = { DATASET_CD: datasetCode }
    TabDataSetApi._getAnalysis(param)
      .then(result => {
        const json = JSON.parse(result[0]?.SAMPLES.replace(/\bNaN\b/g, "null"))
        setData(json)
      })
      .catch(e => {
        console.log(e)
      })
  }, [])

  return (
    <Container>
      <Row>
        <Col>
          <Header2 title="Dataset Analytics" />
        </Col>
      </Row>
      <Row>
        <Col xl={3} className="pr-0">
          <ContentWrapper style={{ height: "92vh" }}>
            <div className="p-2 h-100">
              <h5 className="m-1 w-100 text-left">Overview</h5>
              <DataSetAnalyticsInfo data={data} datasetInfo={datasetInfo} />
            </div>
          </ContentWrapper>
        </Col>
        <Col xl={9}>
          <ContentWrapper style={{ height: "92vh" }}>
            <div className="p-2 h-100">
              <h5 className="m-1 w-100 text-left">Profiling Report</h5>
              <DataSetAnalyticsReport data={data} />
            </div>
          </ContentWrapper>
        </Col>
      </Row>
    </Container>
  )
}

DataSetAnalytics.propTypes = {}

export default DataSetAnalytics
