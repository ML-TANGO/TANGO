import React, { useEffect, useRef } from "react"
import videojs from "video.js"

function VideoPlayer(props) {
  const requestRef = useRef(null)
  const fpsRef = useRef(null)
  const videoPlayerRef = useRef(null) // Instead of ID
  const curFrameRef = useRef(0)

  const videoJSOptions = {
    // autoplay: "muted",
    muted: true,
    controls: false,
    userActions: { hotkeys: true },
    playbackRates: [0.25, 0.5, 1, 1.5, 2],
    loadingSpinner: false
  }

  const startFrame = () => {
    if (curFrameRef.current !== Math.floor(videoPlayerRef.current.currentTime * fpsRef.current)) {
      props.setVideoInfo({
        currentTime: videoPlayerRef.current.currentTime,
        bufferedTime: videoPlayerRef.current.buffered.end(videoPlayerRef.current.buffered.length - 1),
        curFrame: Math.floor(videoPlayerRef.current.currentTime * fpsRef.current)
      })
      curFrameRef.current = Math.floor(videoPlayerRef.current.currentTime * fpsRef.current)
    }
    requestRef.current = videoPlayerRef.current.requestVideoFrameCallback(startFrame)
  }

  useEffect(() => {
    let player
    if (videoPlayerRef) {
      player = videojs(videoPlayerRef.current, videoJSOptions, () => {
        player.fluid = false
        // play 끝난 상태
        player.on("ended", () => {
          // if (props.setIsPlay) props.setIsPlay(false)
          // videoPlayerRef.current.cancelAnimationFrame(requestRef.current)
          videoPlayerRef.current.cancelVideoFrameCallback(requestRef.current)
          props.setReplay(true)
        })

        player.on("play", () => {
          startFrame()
        })

        player.on("pause", () => {
          // videoPlayerRef.current.cancelAnimationFrame(requestRef.current)
          videoPlayerRef.current.cancelVideoFrameCallback(requestRef.current)
          props.setVideoInfo({
            currentTime: videoPlayerRef.current.currentTime,
            bufferedTime: videoPlayerRef.current.buffered.end(videoPlayerRef.current.buffered.length - 1),
            curFrame: Math.floor(videoPlayerRef.current.currentTime * fpsRef.current)
          })
          props.dispatchFrame(Math.floor(videoPlayerRef.current.currentTime * fpsRef.current))
        })

        player.on("loadedmetadata", () => {
          window.cancelAnimationFrame(requestRef.current)
          props.setIsError(false)
          props.setErrorMessage(null)
          props.setDuration(player.duration())
          props.setDefaultVideoSize({ w: videoPlayerRef.current.videoWidth, h: videoPlayerRef.current.videoHeight })
          if (props.setIsPlay) props.setIsPlay(true)
        })

        player.on("error", e => {
          console.log(e)
          props.setErrorMessage(e.target.player.error_.message)
          props.setIsError(true)
        })
      })
    }

    return () => {
      window.cancelAnimationFrame(requestRef.current)
      player.dispose()
    }
  }, [])

  useEffect(() => {
    if (props.src) {
      videoPlayerRef.current.src = props.src
      videoPlayerRef.current.width = props.stageSize.w
      videoPlayerRef.current.height = props.stageSize.h
      props.setDefaultVideoSize({ w: videoPlayerRef.current.videoWidth, h: videoPlayerRef.current.videoHeight })
    }
  }, [props.src])

  useEffect(() => {
    fpsRef.current = props.fps
  }, [props.fps])

  useEffect(() => {
    switch (props.control.k) {
      case "paused":
        if (props.control.v === true) pause()
        else play()
        break
      case "seek":
        seek(props.control.v)
        break
      case "playback":
        playBack(props.control.v)
        break
      case "move":
        move(props.control.v)
        break
      case "replay":
        seek(0)
        play()
        break
      default:
        break
    }
  }, [props.control])

  const play = () => {
    videoPlayerRef.current.play()
  }

  const pause = () => {
    videoPlayerRef.current.pause()
  }

  const seek = pos => {
    videoPlayerRef.current.currentTime = pos
    props.setVideoInfo({
      currentTime: videoPlayerRef.current.currentTime,
      bufferedTime: videoPlayerRef.current.buffered.end(videoPlayerRef.current.buffered.length - 1),
      curFrame: Math.floor(videoPlayerRef.current.currentTime * fpsRef.current)
    })
  }

  const playBack = value => {
    videoPlayerRef.current.playbackRate = value
    props.setVideoInfo({
      currentTime: videoPlayerRef.current.currentTime,
      bufferedTime: videoPlayerRef.current.buffered.end(videoPlayerRef.current.buffered.length - 1),
      curFrame: Math.floor(videoPlayerRef.current.currentTime * fpsRef.current)
    })
  }

  const move = f => {
    videoPlayerRef.current.currentTime += (1 / props.fps) * f
    props.setVideoInfo({
      currentTime: videoPlayerRef.current.currentTime,
      bufferedTime: videoPlayerRef.current.buffered.end(videoPlayerRef.current.buffered.length - 1),
      curFrame: Math.floor(videoPlayerRef.current.currentTime * fpsRef.current)
    })
  }

  return (
    <div>
      <video width={props.stageSize.w} height={props.stageSize.h} ref={videoPlayerRef} />
    </div>
  )
}

VideoPlayer.propTypes = {}

VideoPlayer.defaultProps = {
  fps: 0
}

export default React.memo(VideoPlayer)
