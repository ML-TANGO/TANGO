import React, { useState, useEffect } from "react"
import { BarChart, Bar, Cell, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"
import PropTypes from "prop-types"
import { Card, CardBody } from "reactstrap"

const colorSet = ["#0061D1", "#0073e3", "#0086f7", "#0094ff", "#2ba4ff", "#58b4ff", "#8bc9ff", "#b9deff", "#e2f2ff", "#e3f3ff"]

const GpuRate = ({ dir, gpuData, curGpuData }) => {
  const [labelList, setLabelList] = useState([])
  const [barData, setBarData] = useState([])
  const [clicked, setClicked] = useState(false)
  const [data, setData] = useState([])

  useEffect(() => {
    let info = []
    curGpuData.GPU_LIST?.map(gpu => {
      info.push({ name: gpu, value: curGpuData.DATA[curGpuData.DATA.length - 1][gpu] })
    })
    setLabelList(gpuData.GPU_LIST)
    setData(gpuData.DATA)
    setBarData(info)
  }, [gpuData])

  return (
    <Card className="gpu-rate-card" onClick={() => setClicked(click => !click)}>
      <CardBody className={clicked ? "card-bg-dark" : "card-bg-light"}>
        <div className="card__title" style={{ paddingTop: "20px" }}>
          <div className="bold-text card__title-center stop-dragging">
            {"GPU Status"}
            {clicked ? " (Hourly)" : " (Current)"}
          </div>
        </div>
        {data?.length === 0 ? (
          <div style={{ height: "80%", display: "flex", justifyContent: "center", alignItems: "center" }}>Not Found Gpu</div>
        ) : (
          <ResponsiveContainer className="dashboard__active-users-chart gpu-rate-chart">
            {clicked ? (
              <LineChart
                data={data}
                className="card-bg-dark"
                margin={{
                  top: 20,
                  right: 30,
                  left: 20,
                  bottom: 5
                }}
              >
                <YAxis
                  tickLine={false}
                  // tickFormatter={tickFormer}
                  interval="preserveStartEnd"
                  width={50}
                  // tick={{ transform: 'translate(-30, 0)', fontSize: 11 }}
                  orientation={dir === "rtl" ? "right" : "left"}
                  unit="%"
                  domain={[0, 100]}
                />
                <XAxis
                  padding={{ left: 30, right: 30 }}
                  tick={{ fontSize: 10 }}
                  reversed={dir === "rtl"}
                  dataKey="DATE"
                  // tickFormatter={tick => moment(tick, "D(HH)").toString()}
                />
                <CartesianGrid vertical={false} />
                <Tooltip
                  labelStyle={{ color: "#fff" }}
                  contentStyle={{ backgroundColor: "#000" }}
                  wrapperStyle={{ backgroundColor: "#000" }}
                />
                {labelList?.map((ele, i) => (
                  <Line key={i} type="linear" dataKey={ele} stroke={colorSet[i % colorSet.length]} strokeWidth={2} />
                ))}
              </LineChart>
            ) : (
              <BarChart
                className="card-bg-light"
                data={barData}
                margin={{
                  top: 20,
                  right: 30,
                  left: 20,
                  bottom: 5
                }}
                barSize={20}
              >
                <XAxis dataKey="name" interval={0} tick={{ fontSize: 10 }} padding={{ left: 10, right: 10 }} />
                <YAxis width={50} unit="%" domain={[0, 100]} />

                <CartesianGrid vertical={false} />
                <Bar dataKey="value" fill="#8884d8" label={{ position: "top" }}>
                  {data?.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={colorSet[index % 10]} />
                  ))}
                </Bar>
                <Tooltip
                  labelStyle={{ color: "#fff" }}
                  contentStyle={{ backgroundColor: "#191e23" }}
                  wrapperStyle={{ backgroundColor: "blue" }}
                  itemStyle={{ color: "#777" }}
                />
              </BarChart>
            )}
          </ResponsiveContainer>
        )}
      </CardBody>
    </Card>
  )
}

GpuRate.propTypes = {
  dir: PropTypes.string,
  themeName: PropTypes.string
}

export default GpuRate
