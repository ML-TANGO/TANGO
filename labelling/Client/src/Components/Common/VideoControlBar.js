import React, { useState, useRef, useEffect, useCallback } from "react"
import PropTypes from "prop-types"
import ReactTooltip from "react-tooltip"
import InputNumber from "rc-input-number"
import { VideoSeekSlider } from "react-video-seek-slider"

import {
  BsPlayFill,
  BsPauseFill,
  BsFillSkipBackwardFill,
  BsFillSkipEndFill,
  BsFillSkipForwardFill,
  BsFillSkipStartFill,
  BsArrowClockwise
} from "react-icons/bs"
import { RiSearchEyeLine } from "react-icons/ri"
import CommonSelect from "./CommonSelect"

const getTime = duration => {
  let hour = 0
  let minutes = Math.floor(duration / 60)
  let seconds = Math.floor(duration - minutes * 60)
  if (minutes >= 60) {
    hour = Math.floor(minutes / 60)
    minutes = Math.floor(minutes - hour * 60)
  }

  const h = hour < 10 ? "0" + hour : hour
  const m = minutes < 10 ? "0" + minutes : minutes
  const s = seconds < 10 ? "0" + seconds : seconds

  return hour === 0 ? `${m}:${s}` : `${h}:${m}:${s}`
}

const speedOption = [
  { value: 0.25, label: "0.25x" },
  { value: 0.5, label: "0.5x" },
  { value: 1, label: "1x" },
  { value: 2, label: "2x" },
  { value: 4, label: "4x" },
  { value: 8, label: "8x" }
]

const inputNumberStyle = { width: "70px", height: "25px", lineHeight: "24px" }

function VideoControlBar(props) {
  const sliderRef = useRef(null)
  const { duration, videoInfo, replay, paused, setReplay, setPaused, setControl, className, style, tooltip, setFrameBound } = props

  const [flag, setFlag] = useState(true)
  const [speed, setSpeed] = useState(1)

  const onSpeedChange = useCallback(
    e => {
      setSpeed(e)
      setControl({ k: "playback", v: e })
    },
    [setSpeed, setControl]
  )

  const handleInputNumberKeyDown = useCallback(e => {
    e.stopPropagation()
  }, [])

  const handleInputNumberChange = useCallback(
    value => {
      setFrameBound(value)
    },
    [setFrameBound]
  )

  const seekChange = useCallback(
    time => {
      setControl({ k: "seek", v: time })
    },
    [setControl]
  )

  useEffect(() => {
    if (sliderRef.current) {
      setFlag(flag => !flag)
    }
  }, [sliderRef.current?.offsetWidth])

  useEffect(() => {
    ReactTooltip.rebuild()
  }, [paused, replay])

  const handleMove = useCallback(
    move => () => {
      props.isPredict && setControl({ k: "move", v: move })
    },
    [props.isPredict, setControl]
  )

  const handlePlay = useCallback(
    flag => () => {
      setPaused(flag)
      setControl({ k: "paused", v: flag })
    },
    [setPaused, setControl]
  )

  const handleReplay = useCallback(() => {
    setPaused(false)
    setReplay(false)
    setControl({ k: "replay", v: true })
  }, [setPaused, setReplay, setControl])

  return (
    <div className={`video-control ${className}`} style={style}>
      <div className="video-slider w-100 pl-3 pr-3 pt-1">
        <div className="control-slider" ref={sliderRef}>
          {flag && (
            <VideoSeekSlider
              max={duration}
              currentTime={videoInfo.currentTime}
              progress={videoInfo.bufferedTime}
              onChange={seekChange}
              offset={0}
              secondsPrefix="00:00:"
              minutesPrefix="00:"
              limitTimeTooltipBySides={true}
            />
          )}
          {!flag && (
            <VideoSeekSlider
              max={duration}
              currentTime={videoInfo.currentTime}
              progress={videoInfo.bufferedTime}
              onChange={seekChange}
              offset={0}
              secondsPrefix="00:00:"
              minutesPrefix="00:"
              limitTimeTooltipBySides={true}
            />
          )}
        </div>
      </div>
      <div className="video-controlbar">
        <div className="control-button-icon">
          {paused ? (
            <BsPlayFill data-tip={tooltip ? "Start [k or Space]" : ""} onClick={handlePlay(false)} />
          ) : replay ? (
            <BsArrowClockwise data-tip={tooltip ? "Replay [k or Space]" : ""} onClick={handleReplay} />
          ) : (
            <BsPauseFill data-tip={tooltip ? "Stop [k or Space]" : ""} onClick={handlePlay(true)} />
          )}
        </div>

        <div className={props.isPredict ? "control-button-icon" : "control-button-icon-disable"}>
          <BsFillSkipBackwardFill data-tip={tooltip ? "Prev 10 Frame [Shift + b or <-]" : ""} onClick={handleMove(-10)} />
        </div>
        <div className={props.isPredict ? "control-button-icon" : "control-button-icon-disable"}>
          <BsFillSkipStartFill data-tip={tooltip ? "Prev Frame [b or <-]" : ""} onClick={handleMove(-1)} />
        </div>
        <div className={props.isPredict ? "control-button-icon" : "control-button-icon-disable"}>
          <BsFillSkipEndFill data-tip={tooltip ? "Next Frame [n or ->]" : ""} onClick={handleMove(1)} />
        </div>
        <div className={props.isPredict ? "control-button-icon" : "control-button-icon-disable"}>
          <BsFillSkipForwardFill data-tip={tooltip ? "Next 10 Frame [Shift + n or ->]" : ""} onClick={handleMove(10)} />
        </div>
        <div className="control-range-text ml-2" style={{ width: "120px", textAlign: "left" }}>
          <span className="stop-dragging" style={{ fontSize: "12px" }}>
            {getTime(videoInfo.currentTime)} / {getTime(duration)}
          </span>
        </div>
        <div className={`${props.isPredict ? "control-button-icon" : "control-button-icon-disable"}`}>
          {props.showFrame && (
            <InputNumber
              min={1}
              step={1}
              value={Number(props.frameBound)}
              onChange={handleInputNumberChange}
              onKeyDown={handleInputNumberKeyDown}
              data-tip={tooltip ? "Select Frame Count" : ""}
              style={inputNumberStyle}
              className="video-control-number-button"
            />
          )}
          {props.tracker && <RiSearchEyeLine className="ml-2" data-tip={tooltip ? "Tracker" : ""} onClick={props._handleTracker} />}
        </div>

        <div className="control-input">
          <CommonSelect options={speedOption} onChange={onSpeedChange} selected={speed} isDefault={true} defaultValue={1} isMulti={false} />
        </div>
      </div>
    </div>
  )
}

VideoControlBar.propTypes = {
  isPredict: PropTypes.bool,
  duration: PropTypes.number.isRequired,
  videoInfo: PropTypes.object.isRequired,
  replay: PropTypes.bool.isRequired,
  paused: PropTypes.bool.isRequired,
  setPaused: PropTypes.func.isRequired,
  setControl: PropTypes.func.isRequired,
  setReplay: PropTypes.func.isRequired,
  showFrame: PropTypes.bool,
  tracker: PropTypes.bool,
  frameBound: PropTypes.number,
  setFrameBound: PropTypes.func,
  _handleTracker: PropTypes.func,
  tooltip: PropTypes.bool
}

VideoControlBar.defaultProps = {
  showFrame: false,
  tracker: false,
  isPredict: false,
  tooltip: true
}

export default React.memo(VideoControlBar)
