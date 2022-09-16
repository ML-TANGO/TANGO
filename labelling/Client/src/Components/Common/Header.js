import React, { useState, useCallback } from "react"
import PropTypes from "prop-types"
import { Row, Col } from "reactstrap"
import { NavLink } from "react-router-dom"
import CommonButton from "./CommonButton"
import { convertString } from "Components/Utils/Utils"

const Header = props => {
  const [clickTag, setClickTag] = useState(null)

  const tagClick = useCallback(
    (key, index) => () => {
      if (index === clickTag) {
        props._handleTag("A_A")
        setClickTag(null)
      } else {
        props._handleTag(key)
        setClickTag(index)
      }
    },
    [clickTag, props]
  )

  return (
    <div className="header">
      <Row style={{ marginBottm: "3px" }}>
        <Col xs={12} md={9} lg={9} xl={9}>
          <div className="header-title stop-dragging">{props.title}</div>
          <div className="header-subtitle stop-dragging">
            <span
              onDoubleClick={() => {
                if (props.setEgg) props.setEgg(egg => !egg)
              }}
            >
              {props.subtitle}
            </span>
            {props.dataCount !== null && (
              <div>
                <ul className="tags header-tag">
                  {Object.entries(props.dataCount).map(([key, value], index) => (
                    <li key={key}>
                      <a className={index === clickTag ? "border-selected" : ""} onClick={tagClick(key, index)}>
                        #{convertString(key)} <span>{value}</span>
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </Col>

        {props.custom !== undefined && (
          <Col xs={12} md={3} lg={3} xl={3} style={{ textAlign: "right" }}>
            <div style={{ position: "absolute", right: "20px", bottom: "0px" }}>
              {props.custom?.map((ele, i) =>
                ele.onClick ? (
                  <CommonButton
                    key={i}
                    className={ele.btnClass}
                    ButtonIcon={ele.btnIcon}
                    text={ele.btnTitle}
                    disabled={ele.disabled}
                    onClick={ele.onClick}
                    tooltip={ele.tooltip ? ele.tooltip : null}
                  />
                ) : (
                  <NavLink key={i} to={ele.btnLink}>
                    <CommonButton className={ele.btnClass} ButtonIcon={ele.btnIcon} text={ele.btnTitle} disabled={ele.disabled} />
                  </NavLink>
                )
              )}
            </div>
          </Col>
        )}
      </Row>
    </div>
  )
}

Header.defaultProps = {
  dataCount: {}
}

Header.propTypes = {
  title: PropTypes.string,
  subtitle: PropTypes.string,
  custom: PropTypes.array,
  dataCount: PropTypes.object,
  _handleTag: PropTypes.func,
  setEgg: PropTypes.func
}

export default React.memo(Header)
