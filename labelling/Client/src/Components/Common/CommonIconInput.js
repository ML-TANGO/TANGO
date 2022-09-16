import React, { useState } from "react"
import PropTypes from "prop-types"
import { Input } from "reactstrap"
import styled from "styled-components"
import { BsEyeSlashFill, BsEyeFill } from "react-icons/bs"

const IconSpan = styled.span`
  margin: none !important;
  margin-top: 4px !important;
  width: 35px !important;
  height: 35px !important;
  max-height: 35px !important;
  display: flex !important;
  justify-content: center;
  align-items: center;
  background: #3e3e3e;
  border-radius: 2px 0 0 2px;
`

const StyleInput = styled(Input)`
  height: 35px !important;
  border: none !important;
  border-radius: 0px 2px 2px 0 !important;
  &::placeholder {
    font-size: 14px;
    color: #666 !important;
  }
`

function CommonIconInput(props) {
  const { className, style, inputClassName, name, icon, placeholder, register, onChange, onKeyDown, isPassword } = props
  const [showPassword, setShowPassword] = useState(false)

  const _showPassword = e => {
    e.preventDefault()
    setShowPassword(showPassword => !showPassword)
  }

  return (
    <div className={`form__form-group ${className}`} style={{ ...style, ...{ background: "none", flexWrap: "nowrap" } }}>
      <IconSpan className="form__form-group-label">{icon}</IconSpan>
      <div className="form__form-group-field mt-1">
        <StyleInput
          type={isPassword ? (showPassword ? "text" : "password") : "text"}
          name={name}
          autoComplete="off"
          placeholder={placeholder}
          className={`${inputClassName} login-border-radius`}
          innerRef={register}
          onChange={onChange}
          onKeyDown={onKeyDown}
        />
        {isPassword && (
          <button type="button" className={`form__form-group-button account__password__icon`} onClick={_showPassword}>
            {showPassword ? <BsEyeFill /> : <BsEyeSlashFill />}
          </button>
        )}
      </div>
    </div>
  )
}

CommonIconInput.propTypes = {
  className: PropTypes.string,
  style: PropTypes.object,
  inputClassName: PropTypes.string,
  name: PropTypes.string,
  placeholder: PropTypes.string,
  register: PropTypes.any,
  icon: PropTypes.element,
  onChange: PropTypes.func,
  onKeyDown: PropTypes.func,
  isPassword: PropTypes.bool
}

CommonIconInput.defaultProps = {
  isPassword: false
}

export default CommonIconInput
