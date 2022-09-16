import React from "react"
import PropTypes from "prop-types"

function CommonButton({ ButtonIcon, text, onClick, className, style, disabled, tooltip }) {
  return (
    <>
      <div
        className={`${disabled ? "disable-button" : "common-button"} ${className}`}
        style={style}
        onClick={disabled ? () => {} : onClick}
        data-tip={tooltip ? tooltip : null}
      >
        {ButtonIcon && (
          <div className="common-button-icon">
            {/* <ButtonIcon style={{ fontSize: "14px" }} /> */}
            {typeof ButtonIcon === "object" ? ButtonIcon : <ButtonIcon className="font-12 mb-1" />}
          </div>
        )}
        {text && <div className="common-button-text">{text}</div>}
      </div>
    </>
  )
}

CommonButton.propTypes = {
  ButtonIcon: PropTypes.any,
  text: PropTypes.string,
  onClick: PropTypes.func,
  className: PropTypes.string,
  disabled: PropTypes.bool,
  tooltip: PropTypes.string
}

export default React.memo(CommonButton)
