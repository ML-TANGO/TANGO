import React, { useRef, useMemo } from "react"
import PropTypes from "prop-types"
import styled from "styled-components"
import { MultiGrid, AutoSizer } from "react-virtualized"

const Styles = styled.div`
  margin-top: 0.5rem;
  height: ${props => props.theme.height};
`
const STYLE = {
  border: "1px solid #ccccccdd"
}
const STYLE_BOTTOM_LEFT_GRID = {
  borderRight: "1px solid #ccccccdd"
}
const STYLE_TOP_LEFT_GRID = {
  borderBottom: "1px solid #ccccccdd",
  borderRight: "1px solid #ccccccdd",
  fontWeight: "bold"
}
const STYLE_TOP_RIGHT_GRID = {
  borderBottom: "1px solid #ccccccdd",
  fontWeight: "bold"
}

function VirtualMultiGrid(props) {
  const {
    className,
    wrapperClassName,
    wrapperStyle,
    rowHeight,
    fixedRowCount,
    fixedColumnCount,
    data,
    columns,
    styleHeaderGrid,
    styleBottomLeftGrid,
    styleTopLeftGrid,
    styleTopRightGrid,
    styleWrapperGrid
  } = props
  const multiRef = useRef(null)

  const cellRenderer = ({ columnIndex, key, rowIndex, style }) => {
    let content,
      headerStyle = {}
    if (rowIndex === 0) {
      content = columns[columnIndex].label
      headerStyle = styleHeaderGrid
    } else {
      if (typeof columns[columnIndex].cellRenderer === "function" && columns[columnIndex].cellRenderer) {
        content = columns[columnIndex].cellRenderer({
          cellData: data[rowIndex - 1][columns[columnIndex].dataKey],
          rowIndex: rowIndex,
          rowData: data[rowIndex - 1]
        })
      } else {
        content = data[rowIndex - 1][columns[columnIndex].dataKey]
      }
    }

    return (
      <div
        key={key}
        className={`d-flex align-items-center justify-content-center`}
        style={{
          ...style,
          ...headerStyle,
          borderBottom: "1px solid #eeeeee44",
          borderRight: "1px solid #eeeeee44",
          wordBreak: "break-word",
          lineHeight: 1
        }}
      >
        {content}
      </div>
    )
  }

  const wh = useMemo(() => ({ height: props.height }), [props.height])

  return (
    <Styles className={wrapperClassName} theme={wh} style={wrapperStyle}>
      <AutoSizer>
        {({ width, height }) => (
          <MultiGrid
            ref={multiRef}
            className={className}
            cellRenderer={cellRenderer}
            width={width}
            height={height}
            columnWidth={({ index }) => columns[index].width}
            columnCount={columns.length}
            rowHeight={rowHeight}
            rowCount={data.length + 1}
            fixedColumnCount={fixedColumnCount}
            fixedRowCount={fixedRowCount}
            scrollToColumn={0}
            scrollToRow={0}
            style={{ ...STYLE, ...styleWrapperGrid }}
            styleBottomLeftGrid={{ ...STYLE_BOTTOM_LEFT_GRID, ...styleBottomLeftGrid }}
            styleTopLeftGrid={{ ...STYLE_TOP_LEFT_GRID, ...styleTopLeftGrid }}
            styleTopRightGrid={{ ...STYLE_TOP_RIGHT_GRID, ...styleTopRightGrid }}
            enableFixedColumnScroll
            enableFixedRowScroll
            hideTopRightGridScrollbar
            hideBottomLeftGridScrollbar
          />
        )}
      </AutoSizer>
    </Styles>
  )
}

VirtualMultiGrid.propTypes = {
  className: PropTypes.string,
  wrapperClassName: PropTypes.string,
  wrapperStyle: PropTypes.object,
  height: PropTypes.string,
  width: PropTypes.number,
  columnWidth: PropTypes.number,
  rowHeight: PropTypes.number,
  fixedRowCount: PropTypes.number,
  fixedColumnCount: PropTypes.number,
  data: PropTypes.array,
  columns: PropTypes.array
}

VirtualMultiGrid.defaultProps = {
  styleHeaderGrid: { backgroundColor: "#18233E" }
}

export default VirtualMultiGrid
