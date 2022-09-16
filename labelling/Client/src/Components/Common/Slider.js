/* eslint-disable prefer-destructuring */
import React, { PureComponent } from "react"
import Tooltip from "rc-tooltip"
import Slider from "rc-slider"
import PropTypes from "prop-types"

const Handle = Slider.Handle

const handle = props => {
  const { value, dragging, index, ...rest } = props
  return (
    <Tooltip prefixCls="rc-slider-tooltip" overlay={value} placement="bottom" key={index} visible={dragging} destroyTooltipOnHide={true}>
      <Handle key={index} value={value} {...rest} />
    </Tooltip>
  )
}

handle.propTypes = {
  value: PropTypes.number.isRequired,
  index: PropTypes.number.isRequired
}

export default class SliderTheme extends PureComponent {
  constructor(props) {
    super(props)
  }
  static propTypes = {
    marks: PropTypes.shape(),
    value: PropTypes.number,
    min: PropTypes.number.isRequired,
    max: PropTypes.number.isRequired,
    tipFormatter: PropTypes.func,
    _handleSlider: PropTypes.func
  }

  static defaultProps = {
    marks: {},
    value: 0,
    tipFormatter: value => value
  }

  render() {
    const { marks, value, min, max, tipFormatter, step } = this.props

    return (
      <div className={`${this.props.className} rc-slider`} style={this.props.style}>
        <Slider
          min={min}
          max={max}
          step={step}
          value={value}
          handle={handle}
          marks={marks}
          tipFormatter={tipFormatter}
          onChange={this.props._handleSlider}
        />
      </div>
    )
  }
}
