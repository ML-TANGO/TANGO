import React from "react"
import PropTypes from "prop-types"

const style = { verticalAlign: "text-top", fontSize: "18px" }
function CommonToast(props) {
  const { Icon, text } = props
  return (
    <div>
      <Icon className="mr-2" style={style} />
      {text.split("\n").map((line, i) => (
        <span key={i}>
          {line}
          <br />
        </span>
      ))}
    </div>
  )
}

CommonToast.propTypes = {
  Icon: PropTypes.any,
  text: PropTypes.string.isRequired
}

CommonToast.defaultPropTypes = {
  Icon: null,
  text: ""
}

export default CommonToast
