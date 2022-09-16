// react-libraries
import React, { useEffect, useState } from "react"
import moment from "moment"
import { Progress } from "reactstrap"
import TimeAgo from "react-timeago"
import prettyBytes from "pretty-bytes"
import Highlighter from "react-highlight-words"
import ReactTooltip from "react-tooltip"

// react-icons
import { FaUnlink, FaDatabase } from "react-icons/fa"
import { MdError } from "react-icons/md"
import { BsThreeDotsVertical } from "react-icons/bs"
import { RiFileCopy2Line, RiUser3Line } from "react-icons/ri"
import { ButtonDropdown, DropdownItem, DropdownMenu, DropdownToggle } from "reactstrap"

const getStatus = (info, status, type) => {
  if (type === "IS") {
    if (info.IS_TYPE === "I") return status.I
    else if (info.IS_TYPE === "V") return status.V
    else if (info.IS_TYPE === "R") return status.R
    else if (info.IS_TYPE === "T") return status.T
    else return status
  } else {
    if (info.OBJECT_TYPE === "C" && info.DATA_TYPE === "T") return status.CT
    else if (info.OBJECT_TYPE === "C") return status.C
    else if (info.OBJECT_TYPE === "D") return status.D
    else if (info.OBJECT_TYPE === "S") return status.S
    else if (info.OBJECT_TYPE === "R") return status.R
    else if (info.OBJECT_TYPE === "F") return status.F
    else return status
  }
}

const getValue = (m, v) => {
  if (m && v) {
    if (typeof m === "function") {
      return m(v)
    } else {
      return m
    }
  } else {
    return v ? v : null
  }
}

const CardList = props => {
  const info = props.data
  const message = info.LAST_MSG ? info.LAST_MSG : info.STATUS_MSG ? info.STATUS_MSG : null
  // const messageId = info.DATASET_CD ? info.DATASET_CD : info.IS_CD ? "IS_" + info.IS_CD : info.AI_CD ? info.AI_CD : null
  const status = getStatus(info, props.status, props.type)
  const [isOpen, setIsOpen] = useState(false)
  const [isLoadError, setIsLoadError] = useState(false)

  const toggle = () => {
    setIsOpen(isOpen => !isOpen)
  }

  useEffect(() => {
    ReactTooltip.rebuild()
  }, [info])

  const getCardImage = () => {
    if (props.type === "QP" || props.type === "MS") {
      return (
        <div className="list-image-card bg-transparent" style={{ zIndex: "1" }}>
          <div className="c card-bg-light" style={{ zIndex: "1" }}>
            <div className="wrap">
              <div className="c-left">
                <div className="initial stop-dragging" style={{ color: "#878787" }}>
                  {props.type === "QP" ? String(info.QP_TITLE).substr(0, 1) : String(info.HW_TYPE).substr(0, 1)}
                </div>
                <div className="circle"></div>
              </div>
              <div className="c-right"></div>
            </div>
          </div>
        </div>
      )
    } else if (props.type === "M") {
      return <RiUser3Line />
    } else if (info.DATA_TYPE === "T") {
      if (info.UPLOAD_TYPE === "DB") {
        return <FaDatabase style={{ color: "rgb(141 195 134)", fontSize: "35px" }} />
      } else {
        return <RiFileCopy2Line style={{ color: "rgb(141 195 134)" }} />
      }
    } else if (!info.THUM_NAIL) {
      return <FaUnlink />
    } else if (isLoadError) {
      return (
        <div className="w-100 h-100">
          <MdError className="mr-1" style={{ fontSize: "14px", color: "red" }} />
          Not Found
        </div>
      )
    } else {
      return <img src={info.THUM_NAIL} onError={() => setIsLoadError(true)} alt="Card image cap" />
    }
  }

  const getTitle = () => {
    if (props.type === "IS" || props.type === "QP") {
      return info[props.type + "_TITLE"]
    } else if (props.type === "MS") {
      return info.HW_TITLE
    } else if (props.type === "M") {
      return info.USER_ID
    } else {
      return info.TITLE
    }
  }

  const getSubTitle = () => {
    if (props.type === "MS") {
      return ""
    } else if (props.type === "M") {
      return info.USER_NM
    } else if (info.DESC_TXT && info.DESC_TXT !== "") {
      return info.DESC_TXT
    }
  }

  const getCodeTitle = () => {
    if (props.type === "MS") {
      return info.IS_CD !== null ? String(info.IS_CD) : ""
    } else if (props.type === "M") {
      return info.USER_ID
    } else {
      return String(info[props.type + "_CD"])
    }
  }

  return (
    <div className="card-list">
      <div className="card-list-image">{getCardImage()}</div>
      <div
        data-tip={message}
        className={props.type === "QP" || props.type === "M" ? "card-list-main card-list-main-large" : "card-list-main"}
      >
        <div className="card-list-main-section1">
          <div className="card-list-main-section1-title">
            <Highlighter
              highlightClassName="highlight-class"
              searchWords={[props.searchKey]}
              autoEscape={true}
              textToHighlight={getTitle()}
              onClick={() => {
                props.funcList[0]?.func()
              }}
            />
          </div>
          {(info.DATA_TYPE === "I" || info.IS_TYPE === "I") && <div className="card-list-main-section1-tags skyblue-outline">IMAGE</div>}
          {(info.DATA_TYPE === "V" || info.IS_TYPE === "V") && <div className="card-list-main-section1-tags red-outline">VIDEO</div>}
          {(info.DATA_TYPE === "T" || info.IS_TYPE === "T") && <div className="card-list-main-section1-tags white-outline">TABULAR</div>}
          {info.IS_TYPE === "R" && <div className="card-list-main-section1-tags purple-outline">REALTIME</div>}
          {info.OBJECT_TYPE === "C" && <div className="card-list-main-section1-tags orange-outline">CLASSIFICATION</div>}
          {info.OBJECT_TYPE === "R" && <div className="card-list-main-section1-tags purple-outline">REGRESSION</div>}
          {info.OBJECT_TYPE === "S" && <div className="card-list-main-section1-tags plum-outline">SEGMANTATION</div>}
          {info.OBJECT_TYPE === "D" && <div className="card-list-main-section1-tags green-outline">DETECTION</div>}
          {info.OBJECT_TYPE === "F" && <div className="card-list-main-section1-tags green-outline">FEATURE ENGINEERING</div>}
          {info.AUTO_MODEL && <div className="card-list-main-section1-tags yellow-outline">AUTO</div>}
        </div>
        <div className={"card-list-main-section2"}>
          <div className="card-list-main-section2-text">{getSubTitle()}</div>
        </div>

        <div className="card-list-main-section3">
          <div className={props.type !== "MS" ? "card-list-main-section3-id" : info.IS_CD !== null && "card-list-main-section3-id"}>
            <Highlighter
              highlightClassName="highlight-class"
              searchWords={[props.searchKey]}
              autoEscape={true}
              textToHighlight={getCodeTitle()}
            />
          </div>
          {props.type !== "MS" ? (
            <div className="card-list-main-section3-text">
              created by <p> {info.CRN_USR} </p> at {moment(info.UPT_DTM ? info.UPT_DTM : info.CRN_DTM).format("YYYY-MM-DD HH:mm")}
            </div>
          ) : (
            <div className="card-list-main-section3-text">{info.IS_TITLE}</div>
          )}
        </div>
      </div>
      <div
        className={props.type === "QP" || props.type === "M" ? "card-list-status card-list-status-small" : "card-list-status"}
        style={{ justifyContent: props.type === "DATASET" ? "flex-start" : "space-between" }}
      >
        {status.length !== 0 &&
          status?.map((ele, key) => {
            return ele.items.length === 1 ? (
              ele.items.map((item, key) => {
                // let v = info[item.key] ? info[item.key] : null
                const value = getValue(item.render, info[item.key])
                const label = item.mapper && item.mapper[value] ? item.mapper[value] : value
                const isNull = item.isNull && label === null ? true : false
                let tooltip = getValue(item.tooltip, info[item.key])
                tooltip = tooltip ? (item.prettyBytes ? prettyBytes(tooltip) : item.isFixed ? Number(tooltip).toFixed(4) : tooltip) : "-"
                // tooltip = item.title === "STATUS" ? message : tooltip
                return (
                  <React.Fragment key={key}>
                    {!isNull && (
                      <div className="card-list-status-item" style={{ visibility: item.hide ? "hidden" : "visible" }}>
                        <div className="card-list-status-item-title" style={{ color: item.titleColor }}>
                          <div>{item.title}</div>
                        </div>
                        <div className="card-list-status-item-value">
                          {item.icons && (
                            <div className="card-list-status-item-value-icon" style={{ color: item?.icons[value]?.color }}>
                              {item?.icons[value]?.icon}
                            </div>
                          )}
                          {item.progress ? (
                            <div className="card-list-status-item-value-text">
                              <Progress value={value ? value : 0} style={{ minWidth: "30px", height: "22px" }}>
                                <div
                                  style={{
                                    color: value === 100 ? "white" : "black",
                                    padding: "3px 5px 3px 5px",
                                    textAlign: "right",
                                    width: "100%"
                                  }}
                                >
                                  {value ? value : 0}%
                                </div>
                              </Progress>
                            </div>
                          ) : (
                            <div data-tip={tooltip} className="card-list-status-item-value-text">
                              {value
                                ? item.prettyBytes
                                  ? prettyBytes(value)
                                  : item.isFixed && !Number.isInteger(value)
                                  ? Number(value).toFixed(4)
                                  : label
                                : "-"}
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </React.Fragment>
                )
              })
            ) : (
              <div className="card-list-status-item" key={key}>
                {ele.items.map((subitem, key) => {
                  const value = getValue(subitem.render, info[subitem.key])
                  const label = subitem.mapper && subitem.mapper[value] ? subitem.mapper[value] : value
                  const isNull = subitem.isNull && label === null ? true : false
                  let subTooltip = getValue(subitem.tooltip, info[subitem.key])
                  subTooltip = subTooltip
                    ? subitem.prettyBytes
                      ? prettyBytes(subTooltip)
                      : subitem.isFixed
                      ? Number(subTooltip).toFixed(4)
                      : subTooltip
                    : "-"
                  if (info[subitem.key] === undefined || info[subitem.key] === null) info[subitem.key] = 0
                  return (
                    <React.Fragment key={`sub_${key}`}>
                      {!isNull && (
                        <>
                          <div className="card-list-status-item-vitem-title">{subitem.title}</div>
                          <div className="card-list-status-item-vitem-value">
                            {subitem.icons && <div className="card-list-status-item-vitem-value-icon">{subitem?.icons[value]?.icon}</div>}
                            {subitem.progress ? (
                              <div className="card-list-status-item-vitem-value-text">
                                <Progress value={value ? value : 0} style={{ minWidth: "30px", height: "22px" }}>
                                  <div
                                    style={{
                                      color: value === 100 ? "white" : "black",
                                      padding: "3px 5px 3px 5px",
                                      textAlign: "right",
                                      width: "100%"
                                    }}
                                  >
                                    {value ? value : 0}%
                                  </div>
                                </Progress>
                              </div>
                            ) : (
                              <div data-tip={subTooltip} className="card-list-status-item-vitem-value-text">
                                {value
                                  ? subitem.prettyBytes
                                    ? prettyBytes(value)
                                    : subitem.isFixed && !Number.isInteger(value)
                                    ? Number(value).toFixed(4)
                                    : label
                                  : "-"}
                              </div>
                            )}
                          </div>
                        </>
                      )}
                    </React.Fragment>
                  )
                })}
              </div>
            )
          })}
      </div>
      <div className="card-list-action">
        <div className="card-list-action-ago">
          {(info.UPT_DTM || info.CRN_DTM) && (
            <TimeAgo
              date={moment(info.UPT_DTM ? info.UPT_DTM : info.CRN_DTM ? info.CRN_DTM : undefined)}
              formatter={(value, unit, suffix) => `${value} ${unit === "minute" ? "min" : unit} ${suffix}`}
            />
          )}
        </div>

        <ButtonDropdown isOpen={isOpen} toggle={toggle} className="card-list-action-panel">
          <DropdownToggle>
            <BsThreeDotsVertical />
          </DropdownToggle>
          <DropdownMenu>
            {props.funcList.map((func, key) => (
              <DropdownItem key={key} onClick={func.func}>
                <div className="card-list-action-icon">{func.icon}</div>
                <div className="card-list-action-label">{func.label}</div>
              </DropdownItem>
            ))}
          </DropdownMenu>
        </ButtonDropdown>
      </div>
    </div>
  )
}

export default CardList
