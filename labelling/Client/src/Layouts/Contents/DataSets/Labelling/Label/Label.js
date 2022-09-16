import React, { useEffect, useState, useMemo } from "react"
import DrawArea from "./DrawArea"
import LabelTopBar from "./LabelTopBar"
import DrawVideo from "./DrawVideo"
import { useSelector } from "react-redux"

import { RiShapeLine, RiFocus3Line, RiDragMove2Line, RiEraserLine, RiSearchEyeLine, RiFilterOffFill, RiPriceTag3Line } from "react-icons/ri"
import { FaDrawPolygon, FaMagic } from "react-icons/fa"
import { MdGesture } from "react-icons/md"

function Label(props) {
  const [curMode, setCurMode] = useState("isMove")
  const [modeIcon, setModeIcon] = useState()
  const [modeWord, setModeWord] = useState()
  const [isChanged, setIsChanged] = useState(false)

  const imageLabel = useSelector(
    state => state.imageLabel,
    (prevItem, item) => {
      if (item === prevItem) return true
      else return false
    }
  )

  const videoLabel = useSelector(
    state => state.videoLabel,
    (prevItem, item) => {
      if (item === prevItem) return true
      else return false
    }
  )

  const label = useMemo(() => (props.dataType === "I" ? imageLabel : videoLabel), [props.dataType, imageLabel, videoLabel])

  useEffect(() => {
    if (label.curTag?.TAG_CD) {
      setIsChanged(true)
      let ele = document.getElementById("modechanger")
      if (!ele) return
      ele.classList.remove(["animated-label"])
      ele.style.borderColor = label.curTag.COLOR
      ele.style.color = label.curTag.COLOR
      void ele.offsetWidth
      setModeWord(label.curTag.NAME)
      setModeIcon(<RiPriceTag3Line style={{ color: label.curTag.COLOR }} className="icon" />)
      ele.classList.add(["animated-label"])
    }
  }, [label.curTag])

  useEffect(() => {
    if (curMode !== props.keymode) {
      setCurMode(props.keymode)
      setIsChanged(true)
      let ele = document.getElementById("modechanger")
      if (!ele) return
      ele.classList.remove(["animated-label"])
      ele.style.borderColor = "rgb(253, 88, 101)"
      ele.style.color = "rgb(253, 88, 101)"
      void ele.offsetWidth
      setModeWord(props.keymode === "none" ? "None" : props.keymode.slice(2))
      switch (props.keymode) {
        case "isEdit":
          setModeIcon(<RiFocus3Line className="icon" />)
          break
        case "isMove":
          setModeIcon(<RiDragMove2Line className="icon" />)
          break
        case "isRect":
          setModeIcon(<RiShapeLine className="icon" />)
          break
        case "isPolygon":
          setModeIcon(<FaDrawPolygon className="icon" />)
          break
        case "isMagic":
          setModeIcon(<FaMagic className="icon" />)
          break
        case "isBrush":
          setModeIcon(<MdGesture className="icon" />)
          break
        case "isEraser":
          setModeIcon(<RiEraserLine className="icon" />)
          break
        case "isTracker":
          setModeIcon(<RiSearchEyeLine className="icon" />)
          break
        default:
          setModeIcon(<RiFilterOffFill className="icon" />)
          break
      }
      ele.classList.add(["animated-label"])
    }
  }, [props.keymode])

  return (
    <div className="label_wrap h-100">
      <LabelTopBar dataType={props.dataSet.DATA_TYPE} />

      <div id="modechanger" className={isChanged ? "animated-label" : "animated-label-disable"}>
        <div className="icon-area">{modeIcon}</div>
        <div>{modeWord}</div>
      </div>

      <div className="label_main_wrap">
        {props.dataType === "I" && <DrawArea dataSet={props.dataSet} />}
        {props.dataType === "V" && <DrawVideo dataSet={props.dataSet} />}
      </div>
    </div>
  )
}

export default React.memo(Label)
