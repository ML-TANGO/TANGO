import React from "react"
import { Treemap, ResponsiveContainer, Tooltip } from "recharts"
import { Card, CardBody } from "reactstrap"

// const COLORS = ["#0061D1", "#3580D7", "#5E8DC3", "#87ABD5"]
const COLORS = ["#49708A", "#046D8B", "#36544F", "#1F5F61", "#0B8185"]
// const COLORS = ["#30261C", "#403831", "#36544F", "#1F5F61", "#0B8185"]
// const COLORS = ["#001449", "#012677", "#005BC5", "#5E8DC3", "#0061D1"]
// const COLORS = ["#F0C27B", "#D38157", "#7F2B51", "#4B1248", "#1D0B38"]
// const COLORS = ["#ED6464", "#BF6370", "#87586C", "#574759", "#1A1B1C"]
// const COLORS = ["#D3B8A7", "#AB9588", "#826754", "#30272A", "#2F1F20"]
// const COLORS = ["#8C867A", "#AB9F77", "#8B628A", "#A879AF", "#B4A3CF"]
// const COLORS = ["#107FC9", "#0E4EAD", "#0B108C", "#0C0F66", "#07093D"]
// const COLORS = ["#6DA67A", "#77B885", "#86C28B", "#859987", "#4A4857"]

const CustomizedContent = props => {
  const { depth, x, y, width, height, index, colors, name } = props
  let fontSize
  if (name && depth === 1) {
    fontSize = name.length * 12 < width ? 12 : Math.floor(width / name.length)
  }
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        style={{
          fill: depth < 2 ? colors[index % colors.length] : "rgba(255,255,255,0)",
          stroke: "#191e23",
          strokeWidth: 2 / (depth + 1e-10),
          strokeOpacity: 1 / (depth + 1e-10)
        }}
      />
      {depth === 1 && (
        <text x={x + width / 2} y={y + height / 2} textAnchor="middle" fill="#fff" fontSize={fontSize}>
          {name}
        </text>
      )}
      {/* {depth === 2 && (
        <text x={x + 4} y={y + 10} fill="#fff" fontSize={8}>
          {name} : {size}
        </text>
      )} */}
    </g>
  )
}

const CusteomTooltip = props => {
  const { active, payload } = props
  if (active) {
    return (
      <div className="recharts-tooltip-wrapper">
        <div
          className="recharts-default-tooltip"
          style={{
            margin: "0px",
            padding: "10px",
            backgroundColor: "rgb(25, 30, 35)",
            border: "1px solid rgb(204, 204, 204)",
            whiteSpace: "nowrap"
          }}
        >
          <p className="recharts-tooltip-label">{`${payload[0].payload?.root.name}`}</p>
          <p
            className="recharts-tooltip-item-list"
            style={{ color: "#777" }}
          >{`${payload[0].payload?.name} : ${payload[0].payload?.value}`}</p>
        </div>
      </div>
    )
  }
  return null
}

function TreeMap(props) {
  const { data, title } = props
  return (
    <Card>
      <CardBody className="card-bg-dark">
        <div className="w-100 h-100">
          <div className="card__title mt-2 mb-1">
            <div className="bold-text card__title-center stop-dragging">{title}</div>
          </div>
          {data.length !== 0 && (
            <ResponsiveContainer height={300}>
              <Treemap
                isAnimationActive={true}
                data={data}
                aspectRatio={4 / 3}
                ratio={4 / 3}
                dataKey="size"
                content={<CustomizedContent colors={COLORS} />}
              >
                <Tooltip content={<CusteomTooltip />} />
              </Treemap>
            </ResponsiveContainer>
          )}
        </div>
      </CardBody>
    </Card>
  )
}

TreeMap.propTypes = {}

export default TreeMap
