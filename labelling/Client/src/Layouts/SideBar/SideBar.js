import React, { useEffect, useState } from "react"
import { NavLink, Link, useHistory } from "react-router-dom"
import ReactTooltip from "react-tooltip"
import axios from "axios"

import { BsFolderFill } from "react-icons/bs"
import { FiLogOut, FiDownload, FiUsers, FiChevronsRight, FiChevronsLeft } from "react-icons/fi"
import { FaUserCircle } from "react-icons/fa"

import * as CommonApi from "./../../Config/Services/CommonApi"
import { cloneDeep } from "lodash-es"

const SideBar = props => {
  const history = useHistory()
  const [selected, setSelected] = useState("")
  const [userName, setUserName] = useState("")
  const [groupLinks, setGroupLinks] = useState({
    TRAINING: {
      DataSet: {
        icon: <BsFolderFill />,
        sub: "",
        link: "/dataset"
      }
    }
  })

  const onClick = (group, link) => () => {
    setSelected(link)
  }

  useEffect(() => {
    try {
      const userInfo = JSON.parse(window.sessionStorage.getItem("userInfo"))
      // if (userInfo !== undefined) {
      //   setUserName(userInfo.USER_NM)
      // }
    } catch (e) {
      console.log(e)
    }

    CommonApi._getSidebarInfo({})
      .then(result => {
        let groupLink = cloneDeep(groupLinks)
        groupLink.TRAINING.DataSet.sub = `${result[0].DS_CNT} Datasets are ready`
        // groupLink.TRAINING.Trainer.sub = `${result[0].AI_CNT} Trainer are ready`
        // groupLink.SERVICE.Service.sub = `${result[0].IS_CNT} Service are ready`
        // groupLink.SERVICE.Analytics.sub = `${result[0].PRJ_CNT} Analytics are ready`
        setGroupLinks(groupLink)
      })
      .catch(e => {
        console.log(e)
      })
  }, [])

  useEffect(() => {
    ReactTooltip.rebuild()
  }, [groupLinks, props.collaps])

  const handleLogout = () => {
    delete axios.defaults.headers.common["Authorization"]
    window.sessionStorage.clear()
    history.push({
      pathname: "/"
    })
  }

  return (
    <div className={props.collapse ? "bluai-sidebar bluai-sidebar-collapse" : "bluai-sidebar"}>
      <Link to="/dashboard" onClick={onClick(null, "")}>
        <div className="bluai-logo">
          <div className="bluai-logo-img"></div>
          {!props.collapse && (
            <div className="d-flex">
              <div className="bluai-logo-text">
                <div className="bluai-logo-text-title">BluAI</div>
                <div className="bluai-logo-text-subtitle">ML tool by weda</div>
              </div>
              {/* <div className="bluai-logo-down">
                <NavLink to="/downloadPretrain">
                  <FiDownload fontSize={20} data-tip="DownLoad PreTrain Model" />
                </NavLink>
              </div> */}
            </div>
          )}
        </div>
      </Link>
      <div className="bluai-sidebar-user">
        <div className="user-icon">
          <FaUserCircle />
        </div>
        <div className={!props.collapse ? "user-text" : "display-none"}>
          <div className="user-text-title">{userName}</div>
        </div>
        <div className={!props.collapse ? "user-logout" : "display-none"}>
          <div className="user-down">
            <Link to={{ pathname: "/downloadPretrain", state: { redirectUrl: "/dashboard" } }}>
              <FiDownload fontSize="15" data-tip="DownLoad PreTrain Model" />
            </Link>
          </div>
          {/* <FiLogOut data-tip="logout" onClick={handleLogout} /> */}
        </div>
      </div>
      {Object.keys(groupLinks).map((gl, key) => {
        return (
          <div key={key}>
            <div className={!props.collapse ? "bluai-group" : "bluai-group display-none"}>{gl}</div>
            {Object.keys(groupLinks[gl]).map((linknm, i) => {
              if (linknm === "Camera" && process.env.BUILD !== "EE") {
                return null
              } else {
                return (
                  <NavLink to={groupLinks[gl][linknm].link} key={`${key}_${i}`}>
                    <div className={selected === linknm ? "bluai-link bluai-link-selected" : "bluai-link"} onClick={onClick(gl, linknm)}>
                      <div className="bluai-link-icon" data-tip={props.collapse ? linknm : ""}>
                        {groupLinks[gl][linknm].icon}
                      </div>
                      {!props.collapse && (
                        <div className="bluai-link-text">
                          <div className="bluai-link-text-title">{linknm}</div>
                          <div className="bluai-link-text-subtitle">{groupLinks[gl][linknm].sub}</div>
                        </div>
                      )}
                    </div>
                  </NavLink>
                )
              }
            })}
          </div>
        )
      })}

      <div
        className="bluai-sidebar-footer"
        onClick={() => {
          props.onCollapse()
        }}
      >
        {!props.collapse ? <FiChevronsLeft /> : <FiChevronsRight />}
      </div>
    </div>
  )
}

export default SideBar
