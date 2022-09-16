import React, { useMemo, useEffect, useState } from "react"
import { Card, CardBody, Col } from "reactstrap"
import PropTypes from "prop-types"
import VirtualTable from "Components/Common/VirtualTable"

const ModelTable = props => {
  const [list, setList] = useState([])

  useEffect(() => {
    if (props.aiList !== undefined && props.aiList.length > 0) setList(props.aiList)
  }, [props.aiList])

  const columns = useMemo(
    () => [
      {
        label: "#",
        width: 200,
        className: "text-center",
        disableSort: true,
        dataKey: "index",
        cellRenderer: ({ rowIndex }) => {
          return rowIndex
        }
      },
      {
        label: "TITLE",
        dataKey: "TITLE",
        className: "text-center",
        disableSort: false,
        width: 300
      },
      {
        label: "AI TYPE",
        dataKey: "OBJECT_TYPE",
        className: "text-center",
        disableSort: true,
        width: 200,
        //eslint-disable-next-line react/display-name, react/prop-types
        cellRenderer: ({ cellData }) => {
          switch (cellData) {
            case "C":
              return <div>Classification</div>
            case "S":
              return <div>Segmentation</div>
            case "D":
              return <div>Detction</div>
            case "R":
              return <div>Regression</div>
          }
        }
      },
      {
        label: "DATA TYPE",
        dataKey: "DATA_TYPE",
        className: "text-center",
        disableSort: false,
        width: 150,
        cellRenderer: ({ cellData }) => {
          switch (cellData) {
            case "I":
              return <div>Image</div>
            case "V":
              return <div>Video</div>
            case "T":
              return <div>Text</div>
          }
        }
      }
    ],
    [list]
  )
  return (
    <Col>
      <Card>
        <CardBody className="dashboard__health-chart-card">
          <div className="card__title">
            <div style={{ float: "left", marginRight: "10px" }}>
              <h4 className="bold-text card__title-center">{"AI model list"}</h4>
            </div>
          </div>
          <VirtualTable
            className="vt-table"
            rowClassName="vt-header"
            height="310px"
            headerHeight={40}
            rowHeight={50}
            columns={columns}
            data={list}
          />
        </CardBody>
      </Card>
    </Col>
  )
}

ModelTable.propTypes = {}

export default ModelTable
