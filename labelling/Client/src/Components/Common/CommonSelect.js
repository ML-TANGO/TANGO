import React from "react"
import Select, { components } from "react-select"
import PropTypes from "prop-types"

CommonSelect.defaultProps = {
  isMulti: false,
  className: "",
  placeholder: "",
  menuPortalTarget: true,
  selected: "",
  style: {},
  prefix: "react-select"
}

CommonSelect.propTypes = {
  className: PropTypes.string,
  style: PropTypes.object,
  options: PropTypes.array.isRequired,
  selected: PropTypes.any,
  onChange: PropTypes.func,
  isMulti: PropTypes.bool,
  placeholder: PropTypes.string,
  isDefault: PropTypes.bool,
  disabled: PropTypes.bool,
  isClearable: PropTypes.bool,
  defaultValue: PropTypes.any
}

const defaultStyle = {
  menuPortal: provided => ({ ...provided, zIndex: 999 })
}
const groupStyles = {
  // border: `2px dotted red`,
  // borderRadius: "5px",
  // background: "#f2fcff",
  color: "#888",
  textAlign: "left"
}

const Group = props => (
  <div style={groupStyles}>
    <components.Group {...props} />
  </div>
)

function CommonSelect(props) {
  const {
    inputRef,
    className,
    style,
    options,
    selected,
    onChange,
    isMulti,
    placeholder,
    isDefault,
    disabled,
    isClearable,
    menuPortalTarget,
    prefix,
    group
  } = props
  let transformedValue = transformValue(selected, options, isMulti, group)

  return (
    <div className={`form__form-group-input-wrap ${className}`} style={style}>
      <Select
        ref={inputRef ? inputRef : null}
        styles={defaultStyle}
        valueKey="value"
        value={transformedValue}
        isMulti={isMulti}
        options={options.filter(ele => !ele?.isHide)}
        isDisabled={disabled}
        isClearable={isClearable}
        onChange={isMulti ? multiChangeHandler(onChange) : singleChangeHandler(onChange)}
        defaultValue={isDefault ? (group ? options[0]["options"][0] : options[0]) : ""}
        // defaultValue={{ value: "alldDa", label: "test" }}
        placeholder={placeholder}
        maxMenuHeight={200}
        className="react-select"
        classNamePrefix={prefix}
        menuPortalTarget={menuPortalTarget ? document.querySelector("body") : null}
        menuPlacement="auto"
        components={{ Group }}
      />
    </div>
  )
}

/**
 * onChange from Redux Form Field has to be called explicity.
 */
function singleChangeHandler(func) {
  return function handleSingleChange(value) {
    func(value ? value.value : "")
  }
}

/**
 * onBlur from Redux Form Field has to be called explicity.
 */
function multiChangeHandler(func) {
  return function handleMultiHandler(values) {
    func(values ? values.map(value => value.value) : "")
  }
}

/**
 * For single select, Redux Form keeps the value as a string, while React Select
 * wants the value in the form { value: "grape", label: "Grape" }
 *
 * * For multi select, Redux Form keeps the value as array of strings, while React Select
 * wants the array of values in the form [{ value: "grape", label: "Grape" }]
 */
function transformValue(value, options, multi, group) {
  let newOption = []
  if (group) {
    options.map(option => {
      option.options.map(ele => {
        newOption.push(ele)
      })
    })
  } else newOption = options

  if (multi && typeof value === "string") return []
  const filteredOptions = newOption.filter(option => {
    return multi ? value.indexOf(option.value) !== -1 : option.value === value
  })
  return multi ? filteredOptions : filteredOptions[0] === undefined ? "" : filteredOptions[0]
}

export default React.memo(CommonSelect)
