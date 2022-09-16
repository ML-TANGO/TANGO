import React from "react"
import PropTypes from "prop-types"
import ReactTooltip from "react-tooltip"
import classNames from "classnames"

function ToolbarIcon({ id, btnSts, tooltip, IconElement, _handleClick }) {
  return (
    <div
      data-tip={tooltip}
      data-for={id}
      className={classNames({
        toolbar_icon_wrap: true,
        toolbar_icon_click: btnSts == id
      })}
      onClick={() => {
        _handleClick(id)
      }}
    >
      <IconElement className="toolbar_icon" />
      <ReactTooltip id={id} place="right" delayShow={300} />
    </div>
  )
}

ToolbarIcon.propTypes = {
  id: PropTypes.string.isRequired,
  tooltip: PropTypes.string,
  IconElement: PropTypes.elementType,
  _handleClick: PropTypes.func
}

export default ToolbarIcon
