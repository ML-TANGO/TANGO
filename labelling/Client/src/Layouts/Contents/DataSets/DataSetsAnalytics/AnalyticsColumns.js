import React from "react"
import { Row, Col } from "reactstrap"
import { omit } from "lodash-es"
import { BiBarChart } from "react-icons/bi"
import { XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from "recharts"

import { isNumber } from "../../../../Components/Utils/Utils"

const bgColor = { backgroundColor: "#000" }
const labelColor = { color: "#fff" }

const AnalyticsColumns = ({ data }) => {
  const histogramData = () => {
    const result = data.HISTOGRAM.GRAPH_DATA[0]?.GRAPH_POSITION.map(ele => ({
      [data.HISTOGRAM.LEGEND_X]: isNumber(ele.X) ? Number(ele.X) : ele.X,
      [data.HISTOGRAM.LEGEND_Y]: isNumber(ele.Y) ? Number(ele.Y) : ele.Y
    }))
    return result
  }
  return (
    <>
      <Row>
        <Col xl={12}>
          <div className="tableWrapper">
            <table className="table mt-3">
              <thead>
                <tr>
                  <td>
                    <h5 className="m-1 w-100 text-center">{data.VARIABLE_NAME}</h5>
                  </td>
                </tr>
              </thead>
              <tbody>
                <tr className="tableRow" />
                {Object.entries(omit(data, ["HISTOGRAM", "type", "title"])).map(([key, value], i) => {
                  return (
                    <tr className="tableRow split" key={i}>
                      <td className="dataName">{key}</td>
                      <td className="tableData">
                        {value !== null ? (isNaN(value) || Number.isInteger(value) ? value : value.toFixed(4)) : "-"}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </Col>
      </Row>
      <Row>
        <Col xl={12}>
          <div className="line-separator mr-2 ml-2" />
          <h5>
            <BiBarChart className="mr-1" style={{ verticalAlign: "bottom" }} fontSize={16} />
            HISTOGRAM
          </h5>
          <div className="line-separator mr-2 ml-2" />
          <div style={{ height: "300px" }}>
            <ResponsiveContainer className="mt-2">
              <BarChart
                data={histogramData()}
                margin={{
                  top: 20,
                  right: 20,
                  left: 20,
                  bottom: 5
                }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey={data.HISTOGRAM.LEGEND_X}
                  label={{ value: data.HISTOGRAM.LEGEND_X, position: "insideBottomLeft", offset: -5 }}
                />
                <YAxis
                  dataKey={data.HISTOGRAM.LEGEND_Y}
                  label={{ value: data.HISTOGRAM.LEGEND_Y, position: "insideLeft", angle: -90, offset: -15 }}
                />
                <Tooltip labelStyle={labelColor} contentStyle={bgColor} wrapperStyle={bgColor} />
                <Bar barSize={20} dataKey={data.HISTOGRAM.LEGEND_Y} fill="rgb(0, 151, 230)" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Col>
      </Row>
    </>
  )
}

export default AnalyticsColumns
