import React, { useState, useEffect } from "react"
import { Row, Col } from "reactstrap"
import { RiDatabase2Line } from "react-icons/ri"
import { cloneDeep } from "lodash-es"

import AnalyticsColumns from "./AnalyticsColumns"
import Correlations from "./Correlations"
import AnalyticsSample from "./AnalyticsSample"

const columnHeader = [
  {
    id: "head",
    type: "head",
    title: "Correlations"
  },
  {
    id: "head",
    type: "head",
    title: "Sample"
  },
  {
    id: "First rows",
    type: "Sample",
    title: "First & Last rows"
  },
  {
    id: "head",
    type: "head",
    title: "Columns"
  }
]

const DataSetAnalyticsReport = ({ data }) => {
  const [menuList, setMenuList] = useState([])
  const [menuIndex, setMenuIndex] = useState(1)

  useEffect(() => {
    if (Object.keys(data).length !== 0 && data.VARIABLES) {
      const columnList = data.VARIABLES.map(ele => {
        ele.type = "column"
        ele.title = ele.VARIABLE_NAME
        return ele
      })

      const correlationsList = data.CORRELATIONS.map(ele => {
        ele.type = "correlations"
        ele.title = ele.GRAPH_DATA[0].GRAPH_NAME
        return ele
      })
      const c = cloneDeep(columnHeader)
      c.splice(1, 0, ...correlationsList)

      setMenuList([...c, ...columnList])
    }
  }, [data])

  const handleMenuClick = index => () => {
    setMenuIndex(index)
  }

  const switchAnalyticsInfo = () => {
    if (menuList.length !== 0) {
      const currentMenu = menuList[menuIndex]
      switch (currentMenu.type) {
        case "column":
          return <AnalyticsColumns data={currentMenu} />
        case "correlations":
          return <Correlations data={currentMenu} />
        case "Sample":
          return <AnalyticsSample data={data.SAMPLES} count={data.OVER_VIEW.DATASET_STATISTICS.COUNT} />
        default:
          break
      }
    }
  }

  return (
    <Row className="h-100">
      <Col className="h-100 pb-5" xl={3}>
        <div className="p-2 h-100" style={{ borderRight: "1px solid #3f3f3f", overflow: "auto" }}>
          <div className="mt-2">
            {menuList.map((ele, i) => (
              <React.Fragment key={i}>
                {ele.type === "head" ? (
                  <div className="model-result-head">{ele.title}</div>
                ) : (
                  <div className={`model-result-title ${menuIndex === i ? "model-result-active" : ""}`} onClick={handleMenuClick(i)}>
                    {ele.title}
                  </div>
                )}
              </React.Fragment>
            ))}
          </div>
        </div>
      </Col>
      <Col className="h-100" xl={9}>
        <div className="without-y-scrollbar" style={{ height: "95%", overflowY: "auto" }}>
          {Object.keys(data).length !== 0 ? (
            switchAnalyticsInfo()
          ) : (
            <div className="flex-center w-100 h-100">
              <span className="mr-1 mb-1">
                <RiDatabase2Line size={19} />
              </span>
              No data
            </div>
          )}
        </div>
      </Col>
    </Row>
  )
}

export default DataSetAnalyticsReport
