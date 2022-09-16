import React, { useEffect, useState } from "react"
import { Row, Col } from "reactstrap"
import { bytesToSize } from "../../../../Components/Utils/Utils"
import moment from "moment"

function DataSetAnalyticsInfo({ data, datasetInfo }) {
  const [statistics, setStatistics] = useState({})

  useEffect(() => {
    if (Object.keys(data).length !== 0 && data.OVER_VIEW) {
      setStatistics(data.OVER_VIEW)
    }
  }, [data])

  function switchObjectType(type) {
    switch (type) {
      case "C":
        return "Classfication"
      case "R":
        return "Regression"
      default:
        return "-"
    }
  }

  return (
    <Row>
      <Col>
        <div className="tableWrapper without-y-scrollbar">
          <table className="table mt-2">
            <thead>
              <tr>
                <td>
                  <h6 className="m-1 w-100 text-left">Dataset Infomation</h6>
                </td>
              </tr>
            </thead>
            <tbody>
              <tr className="tableRow" />
              <tr className="tableRow">
                <td className="dataName">Dataset Title</td>
                <td className="tableData">{datasetInfo.TITLE}</td>
              </tr>
              <tr className="tableRow">
                <td className="dataName">Dataset Code</td>
                <td className="tableData">{datasetInfo.DATASET_CD}</td>
              </tr>
              <tr className="tableRow">
                <td className="dataName">Upload Type</td>
                <td className="tableData">{datasetInfo.UPLOAD_TYPE}</td>
              </tr>
              <tr className="tableRow">
                <td className="dataName">Data Type</td>
                <td className="tableData">{datasetInfo.DATA_TYPE === "T" && "Tabular"}</td>
              </tr>
              <tr className="tableRow">
                <td className="dataName">Object Type</td>
                <td className="tableData">{switchObjectType(datasetInfo.OBJECT_TYPE)}</td>
              </tr>

              <tr className="tableRow">
                <td className="dataName">Target Row</td>
                <td className="tableData">{datasetInfo.ROW_TARGET}</td>
              </tr>
              <tr className="tableRow">
                <td className="dataName">Data Size</td>
                <td className="tableData">{bytesToSize(datasetInfo.DATA_SIZE)}</td>
              </tr>
              <tr className="tableRow">
                <td className="dataName">Create Time</td>
                <td className="tableData">{moment(datasetInfo.CRN_DTM).format("YYYY-MM-DD HH:mm:SS")}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div className="tableWrapper">
          <table className="table mt-3">
            <thead>
              <tr>
                <td>
                  <h6 className="m-1 w-100 text-left">Dataset statistics</h6>
                </td>
              </tr>
            </thead>
            <tbody>
              <tr className="tableRow" />
              {statistics?.DATASET_STATISTICS &&
                Object.entries(statistics?.DATASET_STATISTICS).map(([key, value], i) => {
                  return (
                    <tr className="tableRow" key={i}>
                      <td className="dataName">{key}</td>
                      <td className={key.indexOf("MISSING_CELLS") !== -1 ? "tableData accent" : "tableData"}>
                        {Number.isInteger(value) ? value : value.toFixed(4)}
                      </td>
                    </tr>
                  )
                })}
            </tbody>
          </table>
        </div>
        <div className="tableWrapper">
          <table className="table mt-3">
            <thead>
              <tr>
                <td>
                  <h6 className="m-1 w-100 text-left">Variable types</h6>
                </td>
              </tr>
            </thead>
            <tbody>
              <tr className="tableRow" />
              {statistics.VARIABLE_TYPES &&
                Object.entries(statistics?.VARIABLE_TYPES).map(([key, value], i) => {
                  return (
                    <tr className="tableRow" key={i}>
                      <td className="dataName">{key}</td>
                      <td className={"tableData"}>{value}</td>
                    </tr>
                  )
                })}
            </tbody>
          </table>
        </div>
      </Col>
    </Row>
  )
}

DataSetAnalyticsInfo.propTypes = {}
export default DataSetAnalyticsInfo
