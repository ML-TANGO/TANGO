import React, { useState, useEffect, useRef } from "react"
import { Card, CardBody } from "reactstrap"
import { BarChart, Bar, Cell, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts"

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

const CustomYaxisTick = props => {
  const { y, payload } = props
  let checkKor = /[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]/

  return (
    <svg width={130}>
      <g transform={`translate(${0},${y})`}>
        <text x={20} y={4} textAnchor="start" fill="#eee">
          {payload.value.length > 12
            ? checkKor.test(payload.value)
              ? payload.value.slice(0, 8) + "…"
              : payload.value.slice(0, 12) + "…"
            : payload.value}
        </text>
      </g>
    </svg>
  )
}

const CustomLabel = props => {
  const { x, y, value, barWidth } = props
  return (
    <svg x={x}>
      <g transform={`translate(${barWidth + 5},${y + 9})`}>
        <text textAnchor="start" fill="#999" className="sliding">
          {minToTime(value)}
        </text>
      </g>
    </svg>
  )
}

const barMargin = {
  top: 5,
  right: 70,
  left: 50,
  bottom: 5
}

const yPadding = { top: 10, bottom: 10 }
const yMargin = { left: 0, right: 20 }

const barBackground = { fill: "#353A40", radius: 20 }
const tooltipLabelStyle = { color: "#fff" }
const tooltipContentStyle = { backgroundColor: "#191e23" }
const tooltipItemStyle = { color: "#777" }

const DBTime = props => {
  const [data, setData] = useState([])
  const [barWidth, setBarWidth] = useState(0)
  const barRef = useRef(null)

  useEffect(() => {
    setData(props.data)
  }, [props])

  useEffect(() => {
    window.addEventListener("resize", _handleWindowDimensions)
    return () => {
      window.removeEventListener("resize", _handleWindowDimensions)
    }
  }, [])

  useEffect(() => {
    if (barRef.current !== null) {
      setBarWidth(barRef.current?.props?.width)
    }
  }, [barRef.current?.props?.width])

  const _handleWindowDimensions = () => {
    if (barRef.current !== null) {
      setBarWidth(barRef.current?.props?.width)
    }
  }

  return (
    <Card>
      <CardBody className="card-bg-dark">
        <div>
          <div className="card__title mt-2 mb-1">
            <div className="bold-text card__title-center stop-dragging">{props.title}</div>
          </div>
          <div className="dashboard__health-chart">
            {data.length === 0 ? (
              <div style={{ height: "250px", display: "flex", justifyContent: "center", alignItems: "center" }}>Not Found Model</div>
            ) : (
              <ResponsiveContainer height={300} minHeight={300}>
                <BarChart data={data} layout="vertical" margin={barMargin}>
                  <YAxis
                    style={{ textAlign: "left" }}
                    dataKey="name"
                    type="category"
                    tickLine={false}
                    width={90}
                    interval={0}
                    padding={yPadding}
                    margin={yMargin}
                    tick={<CustomYaxisTick />}
                  />
                  <XAxis type="number" hide={true} />
                  <Bar
                    ref={barRef}
                    dataKey="value"
                    fill="#8884d8"
                    label={<CustomLabel barWidth={barWidth} />}
                    barSize={8}
                    radius={20}
                    background={barBackground}
                  >
                    {data.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={props.barColor} />
                    ))}
                  </Bar>
                  <Tooltip
                    formatter={value => [minToTime(value), "Elapsed Time"]}
                    labelStyle={tooltipLabelStyle}
                    contentStyle={tooltipContentStyle}
                    itemStyle={tooltipItemStyle}
                  />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </CardBody>
    </Card>
  )
}

export default DBTime
