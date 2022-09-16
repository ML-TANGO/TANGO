import React, { useState, useEffect, useMemo, useCallback } from "react"
import PropTypes from "prop-types"
import { RiShapeLine, RiEraserLine, RiSearchEyeLine } from "react-icons/ri"
import { FaDrawPolygon, FaMagic } from "react-icons/fa"
import { useDispatch, useSelector } from "react-redux"
import { MdGesture } from "react-icons/md"
import ToolbarIcon from "./ToolbarIcon"

import * as ImageLabelActions from "Redux/Actions/ImageLabelActions"
import * as VideoLabelActions from "Redux/Actions/VideoLabelActions"
import useEnterpriseDivision from "../../../../../Components/Utils/useEnterpriseDivision"

function Toolbar(props) {
  const [btnSts, setBtnSts] = useState("none")
  const labelTracker = useEnterpriseDivision(process.env.BUILD, "dataSet", "labelTracker")

  const dispatch = useDispatch()
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
  const action = useMemo(() => (props.dataType === "I" ? ImageLabelActions : VideoLabelActions), [props.dataType])

  useEffect(() => {
    setBtnSts("none")
    dispatch(action._setBtnSts("none"))
  }, [])

  useEffect(() => {
    setBtnSts(label.btnSts)
  }, [label.btnSts])

  const _handleBtnClick = useCallback(
    id => {
      if (btnSts === id) {
        setBtnSts("none")
        dispatch(action._setBtnSts("none"))
      } else {
        setBtnSts(id)
        dispatch(action._setBtnSts(id))
      }
    },
    [btnSts, dispatch, action]
  )

  return (
    <div className="toolbar_abs_wrap">
      <div className="toolbar_main">
        {props.objectType === "D" && (
          <>
            <ToolbarIcon id={"isRect"} btnSts={btnSts} IconElement={RiShapeLine} tooltip={"Draw Rect [1]"} _handleClick={_handleBtnClick} />
            {props.dataType === "V" && labelTracker && (
              <ToolbarIcon
                id={"isTracker"}
                btnSts={btnSts}
                IconElement={RiSearchEyeLine}
                tooltip={"Tracker [2]"}
                _handleClick={_handleBtnClick}
              />
            )}
          </>
        )}
        {props.objectType === "S" && (
          <>
            <ToolbarIcon
              id={"isPolygon"}
              btnSts={btnSts}
              IconElement={FaDrawPolygon}
              tooltip={"Draw Polygon [1]"}
              _handleClick={_handleBtnClick}
            />
            <ToolbarIcon id={"isMagic"} btnSts={btnSts} IconElement={FaMagic} tooltip={"Magic Wand  [2]"} _handleClick={_handleBtnClick} />
            {props.dataType === "I" && (
              <>
                <ToolbarIcon
                  id={"isBrush"}
                  btnSts={btnSts}
                  IconElement={MdGesture}
                  tooltip={"Draw Brush [3]"}
                  _handleClick={_handleBtnClick}
                />
                <ToolbarIcon
                  id={"isEraser"}
                  btnSts={btnSts}
                  IconElement={RiEraserLine}
                  tooltip={"Eraser [4]"}
                  _handleClick={_handleBtnClick}
                />
              </>
            )}
          </>
        )}
      </div>
    </div>
  )
}

Toolbar.propTypes = {
  dataType: PropTypes.string,
  objectType: PropTypes.string
}

export default React.memo(Toolbar)
