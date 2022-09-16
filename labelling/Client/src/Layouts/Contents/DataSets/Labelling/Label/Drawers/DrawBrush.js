import React from "react"
import PropTypes from "prop-types"
import { Line } from "react-konva"

function DrawBrush({ brushList, opacity, tagList }) {
  return brushList.map((line, i) => {
    const isShow = tagList.filter(tag => Number(tag.TAG_CD) === Number(line.TAG_CD))[0]?.isShow
    return (
      <React.Fragment key={`${line.TAG_CD}_brushLine_${i}`}>
        {line.MODE === "source-over" && (
          <Line
            name="delete"
            globalCompositeOperation="destination-out"
            points={line.POINTS}
            stroke={line.COLOR}
            strokeWidth={line.LINE_WIDTH}
            lineCap={"round"}
            lineJoin={"round"}
            perfectDrawEnabled={false}
            listening={false}
            shadowForStrokeEnabled={false}
            visible={isShow}
          />
        )}
        <Line
          name="brush"
          globalCompositeOperation={line.MODE}
          points={line.POINTS}
          stroke={line.COLOR}
          strokeWidth={line.LINE_WIDTH}
          lineCap={"round"}
          lineJoin={"round"}
          perfectDrawEnabled={false}
          listening={false}
          shadowForStrokeEnabled={false}
          opacity={line.MODE === "source-over" ? opacity * 0.01 : 1}
          visible={isShow}
        />
      </React.Fragment>
    )
  })
}

DrawBrush.propTypes = {
  brushList: PropTypes.array,
  opacity: PropTypes.number
}

export default React.memo(DrawBrush)
