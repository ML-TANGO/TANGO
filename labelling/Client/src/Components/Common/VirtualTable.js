/* eslint-disable react/no-string-refs */
import React, { useState, useEffect, useRef, useCallback, useMemo } from "react"
import styled from "styled-components"
import { Column, Table, AutoSizer, SortDirection, SortIndicator } from "react-virtualized"
import PropTypes from "prop-types"
import { FaDatabase } from "react-icons/fa"

const Styles = styled.div`
  margin-top: 0.5rem;
  /* background: white; */
  height: ${props => props.theme.height};
`

function VirtualTable(props) {
  const [list, setList] = useState([])
  const tableRef = useRef(null)

  useEffect(() => {
    setList(props.data)
  }, [props.data])

  const _headerRenderer = useCallback(({ label, dataKey, sortBy, sortDirection, disableSort }) => {
    return (
      <div>
        {label}
        {!disableSort && sortBy === dataKey && <SortIndicator sortDirection={sortDirection} />}
      </div>
    )
  }, [])

  const rowGetter = useCallback(({ index }) => list[index], [list])
  const wh = useMemo(() => ({ height: props.height, width: props.width }), [props.height, props.width])
  const rowStyle = useMemo(() => props.rowStyle, [props.rowStyle])
  const noRowsRenderer = useCallback(
    () => (
      <div className="mt-5">
        <FaDatabase /> No Data
      </div>
    ),
    []
  )

  return (
    <Styles className={props.wrapperClassName} theme={wh} style={props.style}>
      <AutoSizer>
        {({ height, width }) => (
          <Table
            ref={tableRef}
            className={props.className}
            disableHeader={props.disableHeader}
            width={props.width ? (props.width > width ? props.width : width) : width}
            height={height}
            headerHeight={props.headerHeight}
            rowHeight={props.rowHeight}
            rowCount={list.length}
            rowGetter={rowGetter}
            rowClassName={props.rowClassName}
            onRowMouseOver={props.onRowMouseOver}
            onRowMouseOut={props.onRowMouseOut}
            onRowClick={props.onRowClick}
            onRowDoubleClick={props.onRowDoubleClick}
            rowStyle={rowStyle}
            scrollToIndex={props.scrollIndex || 0}
            noRowsRenderer={props.isNoRowRender ? noRowsRenderer : undefined}
          >
            {props.columns.map((col, i) => {
              return (
                !col.hide && (
                  <Column
                    key={i}
                    className={col.className}
                    headerClassName={col.className}
                    headerRenderer={col.headerRenderer !== undefined ? col.headerRenderer : _headerRenderer}
                    cellRenderer={col.cellRenderer !== undefined ? col.cellRenderer : ({ cellData }) => cellData}
                    sortDirection={SortDirection.ASC}
                    disableSort={col.disableSort}
                    label={col.label}
                    dataKey={col.dataKey}
                    width={col.width}
                  />
                )
              )
            })}
          </Table>
        )}
      </AutoSizer>
    </Styles>
  )
}

VirtualTable.propTypes = {
  className: PropTypes.string,
  wrapperClassName: PropTypes.string,
  columns: PropTypes.array.isRequired,
  data: PropTypes.array.isRequired,
  headerHeight: PropTypes.number.isRequired,
  rowHeight: PropTypes.number.isRequired,
  height: PropTypes.string,
  rowClassName: PropTypes.string,
  scrollIndex: PropTypes.number,
  disableHeader: PropTypes.bool,
  _onRowMouseOver: PropTypes.func,
  _onRowMouseOut: PropTypes.func,
  _onRowDoubleClick: PropTypes.func,
  _onRowClick: PropTypes.func,
  _rowStyle: PropTypes.func,
  isNoRowRender: PropTypes.bool
}

VirtualTable.defaultProps = {
  rowStyle: {},
  disableHeader: false,
  isNoRowRender: true
}

export default React.memo(VirtualTable)
