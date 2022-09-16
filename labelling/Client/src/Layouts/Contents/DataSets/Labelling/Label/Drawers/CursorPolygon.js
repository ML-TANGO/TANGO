import React, { useState } from "react"
import PropTypes from "prop-types"
import { Line, Circle } from "react-konva"

function pointToSerializer(value, mouseX, mouseY) {
  let array = []
  value.map(v => {
    array.push(v.X)
    array.push(v.Y)
  })
  array.push(mouseX)
  array.push(mouseY)
  return array
}

function CursorPolygon(props) {
  const { position, mouseX, mouseY, scale, color, polygonClick } = props
  const [isHover, setIsHover] = useState(false)
  const [hoverIndex, setHoverIndex] = useState(null)

  const _handleMouseEnter = e => {
    const temp = e.target.id().split("_")
    const posIdx = Number(temp[1])
    if (posIdx === 0) {
      setIsHover(true)
      setHoverIndex(posIdx)
    }
  }

  const _handleMouseLeave = e => {
    const temp = e.target.id().split("_")
    const posIdx = Number(temp[1])
    if (posIdx === 0) {
      setIsHover(false)
      setHoverIndex(null)
    }
  }

  if (position.length === 0 || mouseX === null || mouseY === null) {
    return null
  }
  return (
    <>
      <Line
        id={"cursor_linefill"}
        points={pointToSerializer(position, mouseX, mouseY)}
        stroke={color}
        strokeWidth={2 / scale}
        listening={false}
        visible={true}
        fill={`${color}33`}
        closed={true}
      />
      {position.map((point, index) => (
        <Circle
          key={`cursorP_${index}_point`}
          id={`cursorP_${index}_point`}
          x={Number(point.X)}
          y={Number(point.Y)}
          width={hoverIndex === index && isHover ? 12 / props.scale : index === 0 ? 10 / props.scale : 8 / props.scale}
          height={hoverIndex === index && isHover ? 12 / props.scale : index === 0 ? 10 / props.scale : 8 / props.scale}
          stroke={"black"}
          fill={props.color}
          strokeWidth={2 / props.scale}
          onMouseEnter={_handleMouseEnter}
          onMouseLeave={_handleMouseLeave}
          onClick={polygonClick}
          listening={true}
          draggable={false}
          perfectDrawEnabled={false}
        />
      ))}
    </>
  )
}

CursorPolygon.propTypes = {
  mouseX: PropTypes.any.isRequired,
  mouseY: PropTypes.any.isRequired,
  color: PropTypes.string,
  position: PropTypes.array.isRequired,
  scale: PropTypes.number.isRequired,
  needCount: PropTypes.number
}

export default CursorPolygon
