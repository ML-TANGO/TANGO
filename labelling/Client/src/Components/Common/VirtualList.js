import React, { useEffect, useMemo, useRef } from "react"
import { List, AutoSizer } from "react-virtualized"
import styled from "styled-components"
import CardList from "../Cards/CardList"
import { ButtonDropdown, DropdownToggle } from "reactstrap"

import { RiDatabase2Line } from "react-icons/ri"
import { BsThreeDotsVertical } from "react-icons/bs"

const Styles = styled.div`
  margin-top: 0.5rem;
  height: ${props => `${props.theme.height}px`};
`

const StyleNoData = styled.div`
  position: absolute;
  top: 50%;
  right: 50%;
`

const iconStyle = { fontSize: "16px" }

const LoadingSkeleton = () => (
  <div style={{ height: "110px", width: "100%", paddingRight: "10px" }}>
    <div className="card-list">
      <div className="card-list-image"></div>
      <div className="card-list-main">
        <div className="card-list-main-section1 loading-skeleton"></div>
        <div className="card-list-main-section2 ">
          <div className="card-list-main-section2-text"></div>
        </div>
        <div className="card-list-main-section3 loading-skeleton">
          <div className="card-list-main-section3-text"></div>
        </div>
      </div>
      <div className="card-list-status p-1">
        <div className="loading-skeleton w-100 h-100"></div>
      </div>
      <div className="card-list-action">
        <div className="card-list-action-ago"></div>
        <ButtonDropdown className="card-list-action-panel">
          <DropdownToggle>
            <BsThreeDotsVertical />
          </DropdownToggle>
        </ButtonDropdown>
      </div>
    </div>
  </div>
)

function VirtualList(props) {
  const { style, data, height, funcList, status, searchKey, type, isLoad } = props
  const columnYMap = useRef([])
  const collectionRef = useRef(null)

  const heightStyle = useMemo(() => ({ height: height }), [height])

  useEffect(() => {
    columnYMap.current = []
  }, [data])

  const cellRenderer = ({ index, style }) => {
    const datum = data[index]
    return (
      <div key={index} style={{ ...style, paddingRight: "10px" }}>
        <CardList funcList={funcList(datum)} data={datum} status={status} type={type} searchKey={searchKey} />
      </div>
    )
  }

  if (isLoad) {
    const skeleton = new Array(Math.floor(height / 110) + 2).fill(true)
    return (
      <div className="mt-2" style={{ height: `${heightStyle.height}px`, overflow: "auto" }}>
        {skeleton.map((el, i) => (
          <LoadingSkeleton key={`list_${i}`} />
        ))}
      </div>
    )
  }

  if (!isLoad && data.length === 0) {
    return (
      <Styles theme={heightStyle} style={{ ...style, overflow: "visible", position: "relative" }}>
        <StyleNoData>
          <RiDatabase2Line className="mr-1" style={iconStyle} />
          No Data
        </StyleNoData>
      </Styles>
    )
  }

  return (
    <Styles theme={heightStyle} style={{ ...style, overflow: "visible" }}>
      <AutoSizer>
        {({ width }) => (
          <List
            ref={collectionRef}
            rowCount={data.length}
            rowRenderer={cellRenderer}
            height={height}
            width={width}
            overscanRowCount={10}
            rowHeight={110}
          />
        )}
      </AutoSizer>
    </Styles>
  )
}

VirtualList.propTypes = {}

export default React.memo(VirtualList)
