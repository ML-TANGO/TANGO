import React, { useState, useEffect } from "react"
import { Card, CardBody } from "reactstrap"
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts"
import moment from "moment"

const tooltipLabelStyle = { color: "#fff" }
const tooltipContentStyle = { backgroundColor: "#191e23" }
const tooltipItemStyle = { color: "#777" }

const minToTime = t => {
  let min = parseInt(t, 10)
  let day = Math.floor(min / (24 * 60))
  let hours = Math.floor((min - day * 24 * 60) / 60)
  let minutes = Math.round(t - day * 24 * 60 - hours * 60)

  if (hours < 10) {
    hours = "0" + hours
  }
  if (minutes < 10) {
    minutes = "0" + minutes
  }

  return `${day}d ${hours}:${minutes}`
}

const formatXAxis = tickItem => {
  return moment(tickItem, "YYYYMMDDHHmm").format("YYYY-MM-DD HH:mm")
}

const CustomLabel = props => {
  const x = props.viewBox.x
  const y = props.viewBox.y
  const height = props.viewBox.height

  return (
    <svg width="100%">
      <g transform={`translate(${x},${y})`}>
        {props.type !== "now" ? (
          <text x={1} y={height + 15} textAnchor="middle" fill="#777">
            Start Time
          </text>
        ) : (
          <>
            <text x={-4} y={height + 15} textAnchor="middle" fill="#777">
              Current Time
            </text>
            <text x={-5} y={height + 30} textAnchor="middle" fill="#777">
              {moment().format("YYYY-MM-DD HH:mm")}
            </text>
          </>
        )}
      </g>
    </svg>
  )
}

const CustomYaxisTick = (props, time) => {
  const { x, y, payload } = props
  const date = formatXAxis(time)
  const ET = moment().diff(date, "minute")
  let checkKor = /[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]/

  return (
    <>
      <svg width={128}>
        <g transform={`translate(${15},${y + 5})`}>
          <text x={0} y={0} textAnchor="start" fill="#eee">
            {payload.value.length > 12
              ? checkKor.test(payload.value)
                ? payload.value.slice(0, 8) + "…"
                : payload.value.slice(0, 12) + "…"
              : payload.value}
          </text>
        </g>
      </svg>
      <svg>
        <g transform={`translate(${15},${y + 5})`}>
          <text x={x + 210} y={0} textAnchor="start" fill="#999" className="sliding">
            {minToTime(ET)}
          </text>
        </g>
      </svg>
    </>
  )
}

const CustomizedTooltip = props => {
  const date = formatXAxis(props.label)
  const time = moment().diff(date, "minute")
  return (
    <div style={{ textAlign: "left", border: "1px solid #fff", padding: "15px", background: "#000" }}>
      {String(props.label) !== moment().format("YYYYMMDDHHmm") ? (
        <>
          {props.payload.length !== 0 && (
            <div>
              <span style={tooltipItemStyle}>Train Model : </span>
              {props.payload[0].value}
            </div>
          )}
          <div>
            <span style={tooltipItemStyle}>Start Time : </span>
            {formatXAxis(props.label)}
          </div>
          <div>
            <span style={tooltipItemStyle}>Elapsed Time : </span>
            {minToTime(time)}
          </div>
        </>
      ) : (
        <div>Current Time : {moment().format("YYYY-MM-DD HH:mm")}</div>
      )}
    </div>
  )
}

const CustomLineLabel = props => {
  return <div>{props.viewBox.x}</div>
}

const ActiveTime = props => {
  const [data, setData] = useState([])

  useEffect(() => {
    const graphData = props.data.map(ele => {
      const dataArr = []
      dataArr.push(
        {
          category: Number(moment(ele.value).format("YYYYMMDDHHmm")),
          value: ele.name
        },
        {
          category: Number(moment().format("YYYYMMDDHHmm")),
          value: ele.name
        }
      )
      return { name: ele.name, data: dataArr }
    })
    setData(graphData)
  }, [props])

  return (
    <Card>
      <CardBody className="card-bg-dark">
        <div>
          <div className="card__title mt-2 mb-1">
            <div className="bold-text card__title-center stop-dragging">{props.title}</div>
          </div>
          <div className="dashboard__health-chart">
            {data.length === 0 ? (
              <div style={{ height: "300px", display: "flex", justifyContent: "center", alignItems: "center" }}>Not Found Model</div>
            ) : (
              <ResponsiveContainer height={285} strokeDasharray="3 3">
                <LineChart
                  data={data}
                  margin={{
                    top: 10,
                    right: 70,
                    left: 80,
                    bottom: 0
                  }}
                >
                  <XAxis
                    dataKey="category"
                    type="number"
                    scale="time"
                    allowDuplicatedCategory={false}
                    domain={["dataMin", "dataMax"]}
                    interval={0}
                    minTickGap={0}
                    tickFormatter={formatXAxis}
                    tick={false}
                    axisLine={false}
                  />
                  <YAxis
                    dataKey="value"
                    type="category"
                    padding={{ top: 20, bottom: 30 }}
                    axisLine={false}
                    tick={props => CustomYaxisTick(props, data[props.index].data[0].category)}
                  />
                  <Tooltip
                    content={<CustomizedTooltip />}
                    labelStyle={tooltipLabelStyle}
                    contentStyle={tooltipContentStyle}
                    itemStyle={tooltipItemStyle}
                  />
                  <ReferenceLine x={Number(moment().format("YYYYMMDDHHmm"))} label={<CustomLabel type="now" />} stroke="#555" />
                  <ReferenceLine x={data[0].data[0].category} label={<CustomLabel />} stroke="#555" />
                  {data.map(s => {
                    return (
                      <Line
                        key={s.name}
                        dataKey={"value"}
                        data={s.data}
                        name={s.name}
                        stroke={props.barColor}
                        strokeWidth={7}
                        strokeDasharray="3 3"
                        dot={{ stroke: props.barColor, strokeWidth: 0, r: 0 }}
                        label={<CustomLineLabel />}
                      />
                    )
                  })}
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </CardBody>
    </Card>
  )
}

export default ActiveTime
