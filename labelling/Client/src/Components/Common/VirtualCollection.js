import React, { useState, useEffect, useMemo, useRef } from "react"
import PropTypes from "prop-types"
import { Collection, AutoSizer } from "react-virtualized"
import styled from "styled-components"
import Cards from "./../Cards/Cards"
import { animated } from "react-spring"

const Styles = styled.div`
  margin-top: 0.5rem;
  height: ${props => props.theme.height};
`

function VirtualCollection(props) {
  const { wrapperClassName, style, data, height, dataType, body, funcList, overlay, type, animationList, getImage } = props
  const [scrollTop, setScrollTop] = useState(0)
  const columnYMap = useRef([])
  const collectionRef = useRef(null)

  // const height = useMemo(() => Math.ceil(data.length / 5) * 180, [data])
  const heightStyle = useMemo(() => ({ height: height }), [height])

  useEffect(() => {
    columnYMap.current = []
  }, [data])

  useEffect(() => {
    setScrollTop(0)
  }, [dataType])

  const cellRenderer = ({ index, key, style }) => {
    const datum = data[index]
    const animation = animationList[index]
    return (
      <animated.div key={key} style={animation}>
        <Cards
          style={style}
          title={type === "P" ? datum.QP_TITLE : type === "I" ? datum.IS_TITLE : datum.TITLE}
          type={type}
          dataType={type === "I" ? datum.IS_TYPE : datum.DATA_TYPE}
          objectType={datum.OBJECT_TYPE}
          aiStatus={datum.AI_STS ? datum.AI_STS : undefined}
          image={type === "I" ? getImage(datum) : datum.THUM_NAIL}
          body={body(datum)}
          funcList={funcList(datum)}
          overlay={overlay ? overlay(datum) : undefined}
          isCount={type === "P" ? datum.IS_COUNT : undefined}
        />
      </animated.div>
    )
  }

  const cellSizeAndPositionGetter = ({ index }) => {
    const columnPosition = index % 5
    const height = 180
    const width = 310
    const x = columnPosition * width + 5
    const y = columnYMap.current[columnPosition] || 0
    columnYMap.current[columnPosition] = y + height
    return { height, width, x, y }
  }

  return (
    <Styles className={wrapperClassName} theme={heightStyle} style={style}>
      <AutoSizer disableHeight>
        {({ width }) => (
          <Collection
            ref={collectionRef}
            cellCount={data.length}
            cellRenderer={cellRenderer}
            cellSizeAndPositionGetter={cellSizeAndPositionGetter}
            height={height}
            width={width}
            scrollTop={scrollTop}
            horizontalOverscanSize={0}
            scrollToAlignment={"start"}
            verticalOverscanSize={0}
            onScroll={({ scrollTop }) => {
              setScrollTop(scrollTop)
            }}
          />
        )}
      </AutoSizer>
    </Styles>
  )
}

VirtualCollection.propTypes = {}

export default React.memo(VirtualCollection)
