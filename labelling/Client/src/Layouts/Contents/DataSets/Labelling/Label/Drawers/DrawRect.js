import React, { useState, useEffect, useCallback, useMemo, useRef } from "react"
import { Rect, Circle, Text } from "react-konva"
import PropTypes from "prop-types"

const getRectPosition = pos => {
  const hx = pos[0].X >= pos[1].X ? pos[0].X : pos[1].X
  const hy = pos[0].Y >= pos[1].Y ? pos[0].Y : pos[1].Y
  const lx = pos[0].X < pos[1].X ? pos[0].X : pos[1].X
  const ly = pos[0].Y < pos[1].Y ? pos[0].Y : pos[1].Y
  const cx = (pos[0].X + pos[1].X) / 2
  const cy = (pos[0].Y + pos[1].Y) / 2
  return [
    { X: lx, Y: ly },
    { X: cx, Y: ly },
    { X: hx, Y: ly },
    { X: lx, Y: cy },
    { X: hx, Y: cy },
    { X: lx, Y: hy },
    { X: cx, Y: hy },
    { X: hx, Y: hy }
  ]
}

const getRectPositionKonva = (pos1, pos2) => {
  const hx = pos1.x >= pos2.x ? pos1.x : pos2.x
  const hy = pos1.y >= pos2.y ? pos1.y : pos2.y
  const lx = pos1.x < pos2.x ? pos1.x : pos2.x
  const ly = pos1.y < pos2.y ? pos1.y : pos2.y
  const cx = (pos1.x + pos2.x) / 2
  const cy = (pos1.y + pos2.y) / 2
  return [
    { x: lx, y: ly },
    { x: cx, y: ly },
    { x: hx, y: ly },
    { x: lx, y: cy },
    { x: hx, y: cy },
    { x: lx, y: hy },
    { x: cx, y: hy },
    { x: hx, y: hy }
  ]
}

const getRectPoint = pos => {
  const hx = pos[0].X >= pos[1].X ? pos[0].X : pos[1].X
  const hy = pos[0].Y >= pos[1].Y ? pos[0].Y : pos[1].Y
  const lx = pos[0].X < pos[1].X ? pos[0].X : pos[1].X
  const ly = pos[0].Y < pos[1].Y ? pos[0].Y : pos[1].Y
  return [
    { X: lx, Y: ly },
    { X: hx, Y: hy }
  ]
}

const getTextColorByBackgroundColor = (hexColor, isClick, isCurTag) => {
  const c = hexColor.substring(1) // 색상 앞의 # 제거
  const rgb = parseInt(c, 16) // rrggbb를 10진수로 변환
  const r = (rgb >> 16) & 0xff // red 추출
  const g = (rgb >> 8) & 0xff // green 추출
  const b = (rgb >> 0) & 0xff // blue 추출
  const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b // per ITU-R BT.709
  // 색상 선택
  const color = luma < 127.5 ? "#FFFFFF" : "#000000" // 글자색
  return isClick ? `${color}` : isCurTag ? `${color}80` : `${color}55`
}

function DrawRect(props) {
  const {
    id,
    color,
    position,
    scale,
    isMove,
    imageWidth,
    imageHeight,
    _changePos,
    _setClickObject,
    tagName,
    isDash,
    tagCd,
    curTagCd,
    isShow,
    acc,
    setIsCursor,
    isShowPoint
  } = props
  const [isHover, setIsHover] = useState(false)
  const [hoverIndex, setHoverIndex] = useState(null)
  const [isClick, setIsClick] = useState(false)
  const [textWidth, setTextWdith] = useState(0)

  const prevPos = useRef(null)

  const fontSize = useMemo(() => (imageWidth > 1920 ? 40 : imageWidth > 1280 ? 28 : 14), [imageWidth])

  const _handleKeyDown = useCallback(
    e => {
      if (isClick && e.keyCode === 46) {
        setIsClick(false)
      }
    },
    [isClick]
  )

  const _handleClick = useCallback(
    e => {
      if (isMove) {
        e.cancelBubble = true
        if (e.target.id() == id) {
          setIsClick(isClick => !isClick)
          _setClickObject(id)
        }
      }
    },
    [id, isMove, _setClickObject]
  )

  useEffect(() => {
    window.addEventListener("keydown", _handleKeyDown)
    return () => {
      window.removeEventListener("keydown", _handleKeyDown)
    }
  }, [_handleKeyDown])

  const _handleMouseEnter = useCallback(
    e => {
      if (isMove) {
        if (e.target.id() !== id) {
          const temp = e.target.id().split("_")
          const posIdx = Number(temp[1])
          setHoverIndex(posIdx)
          e.target.getStage().container().style.cursor = "pointer"
        } else {
          e.target.getStage().container().style.cursor = "move"
        }
        setIsCursor(false)
        setIsHover(true)
      }
    },
    [isMove, id, setIsCursor]
  )

  const _handleMouseLeave = useCallback(
    e => {
      if (isMove) {
        e.target.getStage().container().style.cursor = "none"
        setHoverIndex(null)
        setIsHover(false)
        setIsCursor(true)
      }
    },
    [isMove, setIsCursor]
  )

  const _handleDragStart = useCallback(
    e => {
      setIsCursor(false)
      if (e.target.attrs.id === id) {
        const pos1 = {
          x: e.target.getX(),
          y: e.target.getY()
        }
        const pos2 = {
          x: e.target.getX() + e.target.attrs.width,
          y: e.target.getY() + e.target.attrs.height
        }
        const positionList = getRectPositionKonva(pos1, pos2)
        let obj = {}
        positionList.forEach((el, i) => {
          obj[`${id}_${i}`] = el
        })
        prevPos.current = obj
      } else {
        const rect = e.target.getLayer().getChildren(node => node.getId() === Number(id))[0]
        const pos1 = {
          x: rect.getX(),
          y: rect.getY()
        }
        const pos2 = {
          x: rect.getX() + rect.attrs.width,
          y: rect.getY() + rect.attrs.height
        }
        const positionList = getRectPositionKonva(pos1, pos2)

        let obj = {}
        positionList.forEach((el, i) => {
          obj[`${id}_${i}`] = el
        })
        prevPos.current = obj
      }
    },
    [id, setIsCursor]
  )

  const _handleDragMove = useCallback(
    e => {
      if (e.target.attrs.id === id) {
        //rect
        const textRect = e.target.getLayer().getChildren(node => node.getId() === `${id}_text_rect`)[0]
        const text = e.target.getLayer().getChildren(node => node.getId() === `${id}_text`)[0]

        textRect.setAttrs({
          x: e.target.getX() + 2,
          y: e.target.getY() - fontSize - 7
        })
        text.setAttrs({
          x: e.target.getX() + 5,
          y: e.target.getY() - fontSize - 2
        })

        const pos1 = {
          x: e.target.getX(),
          y: e.target.getY()
        }
        const pos2 = {
          x: e.target.getX() + e.target.attrs.width,
          y: e.target.getY() + e.target.attrs.height
        }

        const positionList = getRectPositionKonva(pos1, pos2)
        positionList.forEach((el, i) => {
          let point = e.target.getLayer().getChildren(node => node.getId() === `${id}_${i}`)[0]
          point.setAttrs(el)
        })
      } else {
        //point
        const pos = getUpdateRectPoint(e.target)
        const serializePoint = getRectPoint(pos)
        const serializePointList = getRectPosition(serializePoint)
        serializePointList.forEach((el, i) => {
          let point = e.target.getLayer().getChildren(node => node.getId() === `${id}_${i}`)[0]
          point.setAttrs({ x: el.X, y: el.Y })
        })

        const rect = e.target.getLayer().getChildren(node => node.getId() === Number(id))[0]
        const textRect = e.target.getLayer().getChildren(node => node.getId() === `${id}_text_rect`)[0]
        const text = e.target.getLayer().getChildren(node => node.getId() === `${id}_text`)[0]

        textRect.setAttrs({
          x: serializePoint[0].X + 2,
          y: serializePoint[0].Y - fontSize - 7
        })
        text.setAttrs({
          x: serializePoint[0].X + 5,
          y: serializePoint[0].Y - fontSize - 2
        })
        rect.setAttrs({
          x: serializePoint[0].X,
          y: serializePoint[0].Y,
          width: serializePoint[1].X - serializePoint[0].X,
          height: serializePoint[1].Y - serializePoint[0].Y
        })
      }
    },
    [id, fontSize, getUpdateRectPoint]
  )

  const _handleDragEnd = useCallback(
    e => {
      if (e.target.attrs.id === id) {
        // Rect case
        const textRect = e.target.getLayer().getChildren(node => node.getId() === `${id}_text_rect`)[0]
        const text = e.target.getLayer().getChildren(node => node.getId() === `${id}_text`)[0]

        let x = e.target.getX()
        let width = e.target.getX() + e.target.attrs.width
        let y = e.target.getY()
        let height = e.target.getY() + e.target.attrs.height

        if (x < 0 || y < 0 || width > imageWidth || height > imageHeight) {
          // 이미지 범위를 벗어난 경우
          textRect.setAttrs({
            x: prevPos.current[`${id}_0`].x + 2,
            y: prevPos.current[`${id}_0`].y - fontSize - 7
          })
          text.setAttrs({
            x: prevPos.current[`${id}_0`].x + 5,
            y: prevPos.current[`${id}_0`].y - fontSize - 2
          })
          e.target.setAttrs(prevPos.current[`${id}_0`])
          Object.keys(prevPos.current).forEach(key => {
            let point = e.target.getLayer().getChildren(node => node.getId() === key)[0]
            point.setAttrs(prevPos.current[key])
          })
          e.target.getLayer().batchDraw()
        } else {
          let pos1 = {
            x: e.target.getX(),
            y: e.target.getY()
          }
          let pos2 = {
            x: e.target.getX() + e.target.attrs.width,
            y: e.target.getY() + e.target.attrs.height
          }
          const positionList = getRectPositionKonva(pos1, pos2)
          _changePos(
            `${id}`,
            [
              {
                X: positionList[0].x,
                Y: positionList[0].y
              },
              {
                X: positionList[7].x,
                Y: positionList[7].y
              }
            ],
            "all"
          )
        }
      } else {
        // Circle case
        let x = e.target.getX()
        let y = e.target.getY()
        if (x < 0 || x > imageWidth || y < 0 || y > imageHeight) {
          _changePos(
            `${id}`,
            [
              { X: prevPos.current[`${id}_0`].x, Y: prevPos.current[`${id}_0`].y },
              { X: prevPos.current[`${id}_7`].x, Y: prevPos.current[`${id}_7`].y }
            ],
            "all"
          )
        } else {
          const pos = getUpdateRectPoint(e.target)
          const serializePoint = getRectPoint(pos)
          _changePos(`${id}`, serializePoint, "all")
        }
      }
      prevPos.current = null
    },
    [id, prevPos, imageWidth, imageHeight, _changePos, fontSize, getUpdateRectPoint]
  )

  const getUpdateRectPoint = useCallback(pointInfo => {
    // point 이동 시 좌표 계산
    const temp = pointInfo.attrs.id.split("_")
    const id = temp[0]
    let idx = temp[1]
    // 현재 포인트 위치 중에 0,7 point 찾기
    let pos1 = {}
    let pos2 = {}
    const pointList = pointInfo.getLayer().getChildren(node => node.getClassName() === "Circle" && node.getId().split("_")[0] === `${id}`)
    pointList.forEach(el => {
      if (pos1.X) {
        if (pos1.X > el.getX()) pos1.X = el.getX()
      } else {
        pos1.X = el.getX()
      }

      if (pos1.Y) {
        if (pos1.Y > el.getY()) pos1.Y = el.getY()
      } else {
        pos1.Y = el.getY()
      }

      if (pos2.X) {
        if (pos2.X < el.getX()) pos2.X = el.getX()
      } else {
        pos2.X = el.getX()
      }

      if (pos2.Y) {
        if (pos2.Y < el.getY()) pos2.Y = el.getY()
      } else {
        pos2.Y = el.getY()
      }
    })

    // mouse move 시작한 기준점
    let stdPos1 = {}
    let stdPos2 = {}
    stdPos1.X = prevPos.current[`${id}_0`].x
    stdPos1.Y = prevPos.current[`${id}_0`].y
    stdPos2.X = prevPos.current[`${id}_7`].x
    stdPos2.Y = prevPos.current[`${id}_7`].y

    // 포인트 위치별 계산
    const x = pointInfo.getX()
    const y = pointInfo.getY()
    switch (idx) {
      case "0":
        if (x < stdPos2.X) pos1.X = x
        else pos2.X = x
        if (y < stdPos2.Y) pos1.Y = y
        else pos2.Y = y
        break
      case "1":
        pointInfo.setAttrs({ x: prevPos.current[`${id}_1`].x, y: y })
        if (y < stdPos2.Y) pos1.Y = y
        else pos2.Y = y
        break
      case "2":
        if (x > stdPos1.X) pos2.X = x
        else pos1.X = x
        if (y < stdPos2.Y) pos1.Y = y
        else pos2.Y = y
        break
      case "3":
        if (x < stdPos2.X) pos1.X = x
        else pos2.X = x
        pointInfo.setAttrs({ x: x, y: prevPos.current[`${id}_3`].y })
        break
      case "4":
        if (x > stdPos1.X) pos2.X = x
        else pos1.X = x
        pointInfo.setAttrs({ x: x, y: prevPos.current[`${id}_4`].y })
        break
      case "5":
        if (x < stdPos2.X) pos1.X = x
        else pos2.X = x
        if (y > stdPos1.Y) pos2.Y = y
        else pos1.Y = y
        break
      case "6":
        pointInfo.setAttrs({ x: prevPos.current[`${id}_6`].x, y: y })
        if (y > stdPos1.Y) pos2.Y = y
        else pos1.Y = y
        break
      case "7":
        if (x > stdPos1.X) pos2.X = x
        else pos1.X = x
        if (y > stdPos1.Y) pos2.Y = y
        else pos1.Y = y
        break
    }
    return [pos1, pos2]
  }, [])

  const textRef = useRef(null)

  useEffect(() => {
    if (textRef.current !== null) {
      setTextWdith(textRef.current?.textWidth)
    }
  }, [scale])

  const isCurTag = useMemo(() => Number(tagCd) === Number(curTagCd), [tagCd, curTagCd])
  const text = useMemo(() => `${tagName} ${acc ? ` / ${acc.toFixed(4)}` : ""}`, [tagName, acc])

  const pos = getRectPosition(position)
  return (
    <React.Fragment key={`drawRect_${id}`}>
      <Rect
        id={`${id}_text_rect`}
        fill={isClick ? `${color}` : isCurTag ? `${color}80` : `${color}55`}
        x={Number(position[0].X + 2)}
        y={Number(position[0].Y) < 20 ? Number(position[1].Y) + 1 : Number(position[0].Y) - fontSize - 7}
        width={textWidth * 1.15}
        height={fontSize + 8}
        visible={isShow}
        listening={false}
        perfectDrawEnabled={false}
      />
      <Text
        id={`${id}_text`}
        ref={textRef}
        fontSize={fontSize}
        x={Number(position[0].X) + 5}
        y={Number(position[0].Y) < 20 ? Number(position[1].Y) + 5 : Number(position[0].Y) - fontSize - 2}
        fill={getTextColorByBackgroundColor(color, isClick, isCurTag)}
        text={text}
        fontStyle={"bold"}
        visible={isShow}
        listening={false}
        perfectDrawEnabled={false}
      />
      <Rect
        id={id}
        x={Number(position[0].X)}
        y={Number(position[0].Y)}
        width={Number(position[1].X) - Number(position[0].X)}
        height={Number(position[1].Y) - Number(position[0].Y)}
        stroke={color}
        strokeWidth={isCurTag ? 3.5 / scale : 2 / scale}
        fill={isClick ? `${color}BF` : isCurTag ? `${color}80` : `${color}55`}
        dash={isDash ? [5, 5] : null}
        onMouseEnter={_handleMouseEnter}
        onMouseLeave={_handleMouseLeave}
        onDragStart={_handleDragStart}
        onDragMove={_handleDragMove}
        onDragEnd={_handleDragEnd}
        onClick={_handleClick}
        listening={isMove}
        draggable={isMove}
        visible={isShow}
        perfectDrawEnabled={false}
      />

      {isShowPoint &&
        pos.map((point, index) => (
          <Circle
            key={index}
            id={`${id}_${index}`}
            x={Number(point.X)}
            y={Number(point.Y)}
            width={hoverIndex === index && isHover ? 12 / scale : 8 / scale}
            height={hoverIndex === index && isHover ? 12 / scale : 8 / scale}
            stroke={"black"}
            fill={color}
            strokeWidth={2 / scale}
            onMouseEnter={_handleMouseEnter}
            onMouseLeave={_handleMouseLeave}
            onDragStart={_handleDragStart}
            onDragMove={_handleDragMove}
            onDragEnd={_handleDragEnd}
            onClick={_handleClick}
            listening={isMove}
            draggable={isMove}
            visible={isShow}
            perfectDrawEnabled={false}
          />
        ))}
    </React.Fragment>
  )
}

DrawRect.propTypes = {
  id: PropTypes.number,
  cursor: PropTypes.string,
  color: PropTypes.string,
  position: PropTypes.array,
  scale: PropTypes.number,
  isMove: PropTypes.bool,
  imageWidth: PropTypes.number,
  imageHeight: PropTypes.number,
  _changePos: PropTypes.func,
  _deleteObject: PropTypes.func,
  btnSts: PropTypes.string,
  tagName: PropTypes.string,
  isDash: PropTypes.bool,
  tagCd: PropTypes.any,
  curTagCd: PropTypes.any,
  isShow: PropTypes.bool,
  acc: PropTypes.any,
  isShowPoint: PropTypes.bool
}

DrawRect.defaultProps = {
  isShow: true,
  isShowPoint: true
}

export default React.memo(DrawRect)
