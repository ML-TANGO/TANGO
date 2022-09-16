import React, { useState, useEffect } from "react"
import PropTypes from "prop-types"
import styled from "styled-components"
import { Card, CardLink, CardImg, CardBody } from "reactstrap"
import { FiType, FiClock } from "react-icons/fi"
import { FaImages, FaVideo } from "react-icons/fa"

const Wrapper = styled.div`
  /* display: block; */

  /* width: 300px; */
  padding: 0.5rem;
  /* margin: 0.5rem; */
  /* float: left; */
`

const getFuncComp = funcList => {
  return funcList.map((ele, i) => (
    <CardLink key={`func_${i}`} onClick={ele.func}>
      <span className="card-buttons-title">{ele.label}</span>
      {ele.icon}
    </CardLink>
  ))
}

const getIconColor = (type, dataType, objectType) => {
  switch (type) {
    case "D":
    case "A":
      return objectType === "C" ? "#4277ff" : objectType === "S" ? "rgb(10, 208, 0)" : objectType === "D" ? "rgb(255, 164, 70)" : "#000e2e"
    case "I":
      return dataType === "I" ? "#4277ff" : dataType === "R" ? "rgb(10, 208, 0)" : dataType === "V" ? "rgb(255, 164, 70)" : "#000e2e"
    default:
      return null
  }
}

const Cards = props => {
  const { title, type, dataType, objectType, aiStatus, image, body, funcList, overlay, style } = props
  const [BodyContents, setBodyContents] = useState()
  const [isHover, setIsHover] = useState(false)

  useEffect(() => {
    setBodyContents(body)
  }, [body])

  return (
    <Wrapper style={style}>
      <Card inverse className="tab-card">
        <CardBody>
          <div className="card-image" onMouseEnter={() => setIsHover(true)} onMouseLeave={() => setIsHover(false)}>
            <div className="card-overlay">
              {BodyContents}
              <div className="diagonal" />
              <div className="card-buttons">{getFuncComp(funcList)}</div>
            </div>
            <div
              data-tip={title}
              style={{
                position: "absolute",
                width: "30px",
                height: "30px",
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                top: "5px",
                left: "15px",
                borderRadius: "50%",
                marginLeft: "-8px",
                fontSize: "14px",
                zIndex: 3,
                animation: aiStatus === "LEARN" || aiStatus === "READY" ? "blink-animation 1s steps(5, start) infinite" : "none",
                background: props.type !== "P" ? "#000e2ebd" : "transparent",
                color: getIconColor(type, dataType, objectType)
              }}
            >
              {dataType === "V" ? (
                <FaVideo />
              ) : dataType === "I" ? (
                <FaImages />
              ) : dataType === "T" ? (
                <FiType />
              ) : type === "I" && dataType === "R" ? (
                <FiClock />
              ) : null}
            </div>
            {overlay}
            {props.type === "P" ? (
              <div className="db-number-card bg-transparent" style={{ zIndex: "1" }}>
                <div className="c card-bg-light" style={{ zIndex: "1" }}>
                  <div className="wrap">
                    <div className="c-left">
                      <div className="initial stop-dragging" style={{ color: "#878787" }}>
                        {String(props.title).substr(0, 1)}
                      </div>
                      <div className="circle" style={{ width: "12.5rem" }}></div>
                    </div>
                    <div className="c-right">
                      <div className="title">
                        {/* {props.title} */}
                        Source
                      </div>
                      <div className="value">{props.isCount}</div>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <CardImg
                className={isHover ? "hover-image" : "no-hover-image"}
                width="100%"
                height="150px"
                src={image}
                alt="Card image cap"
              />
            )}
          </div>
        </CardBody>
      </Card>
    </Wrapper>
  )
}

Cards.propTypes = {
  title: PropTypes.string.isRequired,
  type: PropTypes.string,
  dataType: PropTypes.string,
  objectType: PropTypes.string,
  image: PropTypes.string,
  body: PropTypes.any.isRequired,
  aiStatus: PropTypes.string,
  funcList: PropTypes.array.isRequired,
  overlay: PropTypes.any
}

export default React.memo(Cards)
