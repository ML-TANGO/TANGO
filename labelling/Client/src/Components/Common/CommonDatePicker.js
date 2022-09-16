import React from "react"
import PropTypes from "prop-types"
import DatePicker, { registerLocale } from "react-datepicker"
import moment from "moment"
import en from "date-fns/locale/en-US"
registerLocale("en", en)

function CommonDatePicker(props) {
  const handleChange = date => {
    const d = moment(date).format("YYYY-MM-DD HH:mm")
    props.onChange(d)
  }

  return (
    <div className={`date-picker ${props.className}`} style={props.style}>
      <DatePicker
        locale="en"
        className="form__form-group-datepicker"
        selected={Date.parse(props.selected)}
        placeholderText={props.placeholderText}
        onChange={handleChange}
        preventOpenOnFocus={true}
        dateFormat={props.dateFormat}
        showTimeSelect={props.showTimeSelect}
        showTimeSelectOnly={props.showTimeSelectOnly}
        timeIntervals={props.timeIntervals}
        timeCaption={props.timeCaption}
        disabled={props.isDisabled}
        isClearable={props.isClearable}
        selectsStart={props.selectsStart}
        selectsEnd={props.selectsEnd}
        startDate={props.startDate}
        endDate={props.endDate}
        minDate={props.minDate}
        showWeekNumbers={props.showWeekNumbers}
        highlightDates={props.highlightDates}
        dayClassName={props.dayClassName}
      />
    </div>
  )
}

CommonDatePicker.propTypes = {
  className: PropTypes.string,
  style: PropTypes.object,
  selected: PropTypes.string,
  onChange: PropTypes.func,
  placeholderText: PropTypes.string,
  showTimeSelectOnly: PropTypes.bool,
  timeIntervals: PropTypes.number,
  timeCaption: PropTypes.string,
  isDisabled: PropTypes.bool,
  isClearable: PropTypes.bool,
  selectsStart: PropTypes.bool,
  selectsEnd: PropTypes.bool,
  startDate: PropTypes.string,
  endDate: PropTypes.string,
  minDate: PropTypes.string,
  showWeekNumbers: PropTypes.bool,
  highlightDates: PropTypes.array
}

CommonDatePicker.defaultProps = {
  className: "",
  style: {},
  showTimeSelect: false,
  showTimeSelectOnly: false,
  dateFormat: "YYYY-MM-DD"
}

export default React.memo(CommonDatePicker)
