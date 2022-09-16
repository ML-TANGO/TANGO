import React, { useState, useEffect, useCallback, useMemo, useRef } from "react"
import PropTypes from "prop-types"
import { Line, Circle, Text } from "react-konva"

function pointToSerializer(value) {
  let array = []
  value.map(v => {
    array.push(v.X)
    array.push(v.Y)
  })
  return array
}

function DrawPolygon(props) {
  const {
    id,
    color,
    position,
    scale,
    isMove,
    imageWidth,
    imageHeight,
    _changePos,
    _deleteObject,
    _setClickObject,
    tagName,
    tagCd,
    curTagCd,
    isShow,
    setIsCursor,
    isShowPoint
  } = props
  const [isHover, setIsHover] = useState(false)
  const [isClick, setIsClick] = useState(false)
  // const [prevPos, setPrevPos] = useState(null)
  const [hoverIndex, setHoverIndex] = useState(null)
  const fontSize = imageWidth > 1920 ? 40 : imageWidth > 1280 ? 28 : 14
  const prevPos = useRef(null)

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
    [isMove, id, _setClickObject]
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
          // point
          const temp = e.target.id().split("_")
          const posIdx = Number(temp[1])
          const type = temp[2]
          if (type === "point") {
            e.target.getStage().container().style.cursor = "pointer"
            setHoverIndex(posIdx)
          }
        } else {
          // ploygon
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
        // polygon drag

        // 레이어에서 현재 polygon의 Circle 리스트 가져옴
        let circleList = e.target.getLayer().getChildren(node => {
          if (node.getClassName() === "Circle") {
            let temp = node.getId().split("_")
            if (String(temp[0]) === String(id) && temp[2] === "point") {
              return node
            }
          }
        })

        // Drag 이전 값 저장
        let obj = {}
        circleList.map((circle, i) => {
          obj[`${id}_${i}`] = {
            x: circle.getX(),
            y: circle.getY()
          }
        })
        prevPos.current = obj
      } else {
        // point drag
        let pos = {
          X: e.target.attrs.x,
          Y: e.target.attrs.y
        }
        let obj = {}
        obj[e.target.attrs.id] = pos
        prevPos.current = obj
      }
    },
    [id, setIsCursor]
  )

  const _handleDragMove = useCallback(
    e => {
      if (e.target.attrs.id === id) {
        // 레이어에서 현재 polygon의 Circle 리스트 가져옴
        let circleList = e.target.getLayer().getChildren(node => {
          if (node.getClassName() === "Circle") {
            let temp = node.getId().split("_")
            if (String(temp[0]) === String(id) && temp[2] === "point") {
              return node
            }
          }
        })
        const text = e.target.getLayer().getChildren(node => node.getId() === `${id}_text`)[0]

        text.setAttrs({
          x: prevPos.current[`${id}_0`].x + e.target.getX() + 5,
          y: prevPos.current[`${id}_0`].y + e.target.getY() - fontSize
        })
        // polygon이 이동에 따라 circle point 값 변경
        circleList.map((circle, i) => {
          circle.setAttrs({
            x: prevPos.current[`${id}_${i}`].x + e.target.getX(),
            y: prevPos.current[`${id}_${i}`].y + e.target.getY()
          })
        })
      } else {
        if (!e.evt.altKey) {
          const polygon = e.target.getLayer().getChildren(node => node.getId() === Number(id))[0]
          const index = e.target.attrs.id.split("_")[1]
          polygon.attrs.points[index * 2] = e.target.attrs.x
          polygon.attrs.points[index * 2 + 1] = e.target.attrs.y

          const text = e.target.getLayer().getChildren(node => node.getId() === `${id}_text`)[0]
          const firstPoint = e.target.getLayer().getChildren(node => node.getId() === `${id}_0_point`)[0]
          text.setAttrs({
            x: firstPoint.getX() + polygon.getX() + 5,
            y: firstPoint.getY() + polygon.getY() - fontSize
          })
          // _changePos(e.target.attrs.id, {
          //   X: e.target.attrs.x,
          //   Y: e.target.attrs.y
          // })
        }
      }
    },
    [prevPos, id, fontSize]
  )

  const _handleDragEnd = useCallback(
    e => {
      if (e.target.attrs.id === id) {
        const text = e.target.getLayer().getChildren(node => node.getId() === `${id}_text`)[0]
        let circleList = e.target.getLayer().getChildren(node => {
          if (node.getClassName() === "Circle") {
            let temp = node.getId().split("_")
            if (String(temp[0]) === String(id) && temp[2] === "point") {
              return node
            }
          }
        })
        // 이미지 범위 벗어나는지 check
        let checkImage = false
        circleList.map(circle => {
          let x = circle.getX()
          let y = circle.getY()
          if (x < 0 || y < 0 || x > imageWidth || y > imageHeight) {
            checkImage = true
          }
        })

        if (checkImage) {
          // 벗어날 경우 초기값으로 다시 그리기
          circleList.map((circle, i) => {
            circle.setAttrs({
              x: prevPos.current[`${id}_${i}`].x,
              y: prevPos.current[`${id}_${i}`].y
            })
          })
          text.setAttrs({
            x: prevPos.current[`${id}_0`].x + 5,
            y: prevPos.current[`${id}_0`].y - fontSize
          })
          e.target.setX(0)
          e.target.setY(0)
        } else {
          let pos = circleList.map(circle => {
            circle.setAttrs({
              x: circle.getX(),
              y: circle.getY()
            })
            return { X: circle.getX(), Y: circle.getY() }
          })
          _changePos(`${id}`, pos, "all")
          e.target.setX(0)
          e.target.setY(0)
        }
      } else {
        // Circle case
        let x = e.target.getX()
        let y = e.target.getY()
        if (x < 0 || x > imageWidth || y < 0 || y > imageHeight) {
          e.target.setAttrs({
            x: prevPos.current[e.target.attrs.id].x,
            y: prevPos.current[e.target.attrs.id].y
          })
          _changePos(e.target.attrs.id, {
            X: prevPos.current[e.target.attrs.id].X,
            Y: prevPos.current[e.target.attrs.id].Y
          })
        } else {
          if (e.evt.altKey) {
            _changePos(e.target.attrs.id, {
              X: prevPos.current[e.target.attrs.id].X,
              Y: prevPos.current[e.target.attrs.id].Y
            }).then(() => {
              _changePos(
                e.target.attrs.id,
                {
                  X: e.target.attrs.x,
                  Y: e.target.attrs.y
                },
                "add"
              )
            })
          } else {
            _changePos(e.target.attrs.id, {
              X: e.target.attrs.x,
              Y: e.target.attrs.y
            })
          }
        }
      }
      prevPos.current = null
    },
    [id, prevPos, imageWidth, imageHeight, _changePos, fontSize, setIsCursor]
  )

  const _handleDbClick = useCallback(
    e => {
      if (isMove) {
        const id = e.target.getId()
        _deleteObject(id, "point")
      }
    },
    [_deleteObject, isMove]
  )

  const isCurTag = useMemo(() => Number(tagCd) === Number(curTagCd), [tagCd, curTagCd])
  if (position.length < 2) return null

  return (
    <React.Fragment key={`drawPolygon_${id}`}>
      <Text
        id={`${id}_text`}
        fontSize={fontSize}
        x={Number(position[0].X) + 5}
        y={Number(position[0].Y) - fontSize}
        fill={color}
        text={tagName}
        fontStyle={"bold"}
        visible={isShow}
        perfectDrawEnabled={false}
      />
      <Line
        id={id}
        name="polygon"
        points={pointToSerializer(position)}
        stroke={color}
        strokeWidth={isCurTag ? 3.5 / scale : 2 / scale}
        fill={isClick ? `${color}BF` : isCurTag ? `${color}99` : `${color}55`}
        closed={true}
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
        position.map((point, index) => (
          <Circle
            key={`${id}_${index}_point`}
            id={`${id}_${index}_point`}
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
            onDblClick={_handleDbClick}
            listening={isMove}
            draggable={isMove}
            visible={isShow}
            perfectDrawEnabled={false}
          />
        ))}
    </React.Fragment>
  )
}

DrawPolygon.propTypes = {
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
  _setClickObject: PropTypes.func,
  btnSts: PropTypes.string,
  tagName: PropTypes.string,
  tagCd: PropTypes.any,
  curTagCd: PropTypes.any,
  isShow: PropTypes.bool,
  acc: PropTypes.string,
  setIsCursor: PropTypes.func,
  isShowPoint: PropTypes.bool
}

DrawPolygon.defaultProps = {
  isShow: true,
  isShowPoint: true
}

export default React.memo(DrawPolygon)
