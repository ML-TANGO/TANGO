import React, { useEffect, useState } from "react"
import { Row, Col } from "reactstrap"
import VirtualTable from "../../../../Components/Common/VirtualTable"

const AnalyticsSample = ({ data, count }) => {
  const [firstColumnList, setFirstColumnList] = useState([])
  const [lastColumnList, setLastColumnList] = useState([])

  useEffect(() => {
    const columnArr = data[0].COLUMNS.map((ele, i) => ({
      id: i + 1,
      label: ele,
      dataKey: ele,
      className: "text-center test",
      disableSort: true,
      width: 150
    }))
    const firstList = [...columnArr]
    const lastList = [...columnArr]
    firstList.splice(0, 0, {
      id: 0,
      label: "Index",
      dataKey: "Index",
      className: "text-center",
      disableSort: true,
      width: 100,
      cellRenderer: ({ rowIndex }) => {
        return rowIndex + 1
      }
    })

    lastList.splice(0, 0, {
      id: 0,
      label: "Index",
      dataKey: "Index",
      className: "text-center",
      disableSort: true,
      width: 100,
      cellRenderer: ({ rowIndex }) => {
        return count + rowIndex - 9
      }
    })
    setFirstColumnList(firstList)
    setLastColumnList(lastList)
  }, [data])

  return (
    <>
      <Row>
        <Col>
          <div className="line-separator mr-2 ml-2" />
          <h5>First Rows</h5>
          <div className="line-separator mr-2 ml-2" />
          <VirtualTable
            className="vt-table text-break-word"
            rowClassName="vt-header"
            height={420 + "px"}
            headerHeight={50}
            rowHeight={35}
            columns={firstColumnList}
            data={data[0].DATA}
            style={{ overflowX: "scroll", overflowY: "hidden" }}
            width={firstColumnList.length * 120}
          />
        </Col>
      </Row>
      <Row>
        <Col>
          <div className="line-separator mr-2 ml-2 mt-3" />
          <h5>Last Rows</h5>
          <div className="line-separator mr-2 ml-2" />
          <VirtualTable
            className="vt-table text-break-word"
            rowClassName="vt-header"
            height={420 + "px"}
            headerHeight={50}
            rowHeight={35}
            columns={lastColumnList}
            data={data[1].DATA}
            style={{ overflowX: "scroll", overflowY: "hidden" }}
            width={lastColumnList.length * 120}
          />
        </Col>
      </Row>
    </>
  )
}

export default AnalyticsSample
