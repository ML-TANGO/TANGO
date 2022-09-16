import React from "react"
import PropTypes from "prop-types"
import { Rect } from "react-konva"

function CursorRect(props) {
  const { position, mouseX, mouseY, scale, color } = props
  if (position.length === 0 || mouseX === null || mouseY === null) {
    return null
  }
  return (
    <>
      <Rect
        x={position[0].X}
        y={position[0].Y}
        width={mouseX - position[0].X}
        height={mouseY - position[0].Y}
        opacity={1}
        visible={true}
        stroke={color}
        strokeWidth={2 / scale}
        listening={false}
      />
      {/* {position.map((point, index) => (
        <Circle
          key={`cursorR_${index}_point`}
          id={`cursorR_${index}_point`}
          x={Number(point.X)}
          y={Number(point.Y)}
          width={10 / props.scale}
          height={10 / props.scale}
          stroke={"black"}
          fill={props.color}
          strokeWidth={2 / props.scale}
          listening={false}
          draggable={false}
          perfectDrawEnabled={false}
        />
      ))} */}
    </>
  )
}

CursorRect.propTypes = {
  mouseX: PropTypes.any.isRequired,
  mouseY: PropTypes.any.isRequired,
  color: PropTypes.string,
  position: PropTypes.array.isRequired,
  scale: PropTypes.number.isRequired,
  needCount: PropTypes.number
}

export default React.memo(CursorRect)
