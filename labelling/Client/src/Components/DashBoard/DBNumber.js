import React, { useState, useRef, useEffect, useCallback } from "react"
import { Card, CardBody } from "reactstrap"
import { PieChart, Pie, Cell, ResponsiveContainer, Sector } from "recharts"
import { useSpring, animated } from "react-spring"
import useResizeListener from "../Utils/useResizeListener"
import { RiDatabase2Line } from "react-icons/ri"

const dataColor = ["#178fd6", "#d7d8da", "#767f88", "#87ABD5", "#BEAE83"]
const aiColor = ["#BDC971", "#E9C300", "#F2E8B9", "#CDBF82", "#7EB9A0"]
const sourceColor = ["#D9C3C6", "#A513B1", "#CC8FA5", "#BBA6E5", "#B67BCB"]

const renderActiveShape = props => {
  const RADIAN = Math.PI / 180
  const { cx, cy, midAngle, innerRadius, outerRadius, startAngle, endAngle, fill, payload, percent, value, name } = props
  const sin = Math.sin(-RADIAN * midAngle)
  const cos = Math.cos(-RADIAN * midAngle)
  const sx = cx + (outerRadius + 10) * cos
  const sy = cy + (outerRadius + 10) * sin
  const mx = cx + (outerRadius + 15) * cos
  const my = cy + (outerRadius + 15) * sin
  const ex = mx + (cos >= 0 ? 1 : -1) * 10
  const ey = my
  const textAnchor = cos >= 0 ? "start" : "end"
  const p = Number(percent * 100).toFixed(2)
  return (
    <g>
      <text x={cx} y={cy} dy={0} style={{ fontSize: "11px" }} textAnchor="middle" fill={fill}>
        {payload.name}
      </text>
      <text x={cx} y={cy} dy={15} style={{ fontSize: "11px" }} textAnchor="middle" fill={fill}>
        {p}%
      </text>
      <Sector cx={cx} cy={cy} innerRadius={innerRadius} outerRadius={outerRadius} startAngle={startAngle} endAngle={endAngle} fill={fill} />
      <Sector
        cx={cx}
        cy={cy}
        startAngle={startAngle}
        endAngle={endAngle}
        innerRadius={outerRadius + 6}
        outerRadius={outerRadius + 10}
        // fill={"#E9F2F9"}
        fill={fill}
      />
      <path d={`M${sx},${sy}L${mx},${my}L${ex},${ey}`} stroke={fill} fill="none" />
      <circle cx={ex} cy={ey} r={2} fill={fill} stroke="none" />
      <text
        x={ex + (cos >= 0 ? 1 : -1) * 6}
        y={ey + 3}
        textAnchor={textAnchor}
        style={{ fontSize: "11px" }}
        fill="#fff"
      >{`${name} : ${value}`}</text>
      {/* <text x={ex + (cos >= 0 ? 1 : -1) * 12} y={ey} dy={18} textAnchor={textAnchor} fill="#999"></text> */}
    </g>
  )
}

const DBNumber = props => {
  const pieRef = useRef(null)
  const [width, height] = useResizeListener(pieRef)

  const [clicked, setClicked] = useState(false)
  const [mouseOver, setMouseOver] = useState(false)
  const [data, setData] = useState([])
  const [activeIndex, setActiveIndex] = useState(0)

  const { transform, opacity } = useSpring({
    opacity: !props.disableClick && (clicked || mouseOver) ? 1 : 0,
    transform: `perspective(600px) rotateY(${!props.disableClick && (clicked || mouseOver) ? 180 : 0}deg)`,
    config: { mass: 5, tension: 500, friction: 100 }
  })

  useEffect(() => {
    setData(props.data)
  }, [props])

  const onPieEnter = useCallback((data, index) => {
    setActiveIndex(index)
  }, [])

  useEffect(() => {
    const timer = setInterval(() => {
      setClicked(click => !click)
    }, 5000)
    return () => {
      clearInterval(timer)
    }
  }, [])

  const getColor = type => {
    switch (type) {
      case "D":
        return "#1881CD"
      case "T":
        return "#E1C117"
      case "S":
        return "#A219B8"
      case "A":
        return "#19B883"
    }
  }
  return (
    <Card
      onMouseEnter={() => {
        setMouseOver(true)
      }}
      onMouseLeave={() => {
        setMouseOver(false)
      }}
    >
      <CardBody className="db-number-card bg-transparent">
        <div style={{ width: "100%", height: "100%" }} ref={pieRef}>
          <animated.div className="c" style={{ zIndex: "2", opacity, transform: transform.interpolate(t => `${t} rotateY(180deg)`) }}>
            {data.length !== 0 ? (
              <ResponsiveContainer className="card-bg-dark db-number-pie" width={"100%"} height={"100%"}>
                <PieChart>
                  <Pie
                    activeIndex={activeIndex}
                    activeShape={renderActiveShape}
                    nameKey="name"
                    dataKey="value"
                    data={data}
                    cx={width / 2}
                    cy={height / 2}
                    innerRadius={40}
                    outerRadius={50}
                    fill="#11644D"
                    onMouseEnter={onPieEnter}
                  >
                    {data?.map((entry, index) => (
                      <Cell
                        key={index}
                        fill={
                          props.type === "D"
                            ? dataColor[index % dataColor.length]
                            : props.type === "T"
                            ? aiColor[index % aiColor.length]
                            : props.type === "S"
                            ? sourceColor[index % sourceColor.length]
                            : ""
                        }
                      />
                    ))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="card-bg-dark db-number-pie h-100 d-flex" style={{ justifyContent: "center", alignItems: "center" }}>
                <RiDatabase2Line className="mr-1" style={{ fontSize: "16px", verticalAlign: "sub" }} />
                <span style={{ fontSize: "14px" }}>No Data</span>
              </div>
            )}
          </animated.div>
          <animated.div className="c card-bg-light" style={{ zIndex: "1", opacity: opacity.interpolate(o => 1 - o), transform }}>
            <div className="wrap">
              <div className="c-left">
                <div className="initial stop-dragging" style={{ color: getColor(props.type) }}>
                  {props.type}
                </div>
                <div className="circle"></div>
              </div>
              <div className="c-right">
                <div className="title">{props.title}</div>
                <div className="value">{props.value}</div>
              </div>
            </div>
          </animated.div>
        </div>
      </CardBody>
    </Card>
  )
}

DBNumber.defaultProps = {
  data: []
}

export default DBNumber
