import { useState, useEffect, useCallback } from "react"
import { useSelector } from "react-redux"

function useResizeListener(target) {
  const [width, setWidth] = useState(0)
  const [height, setHeight] = useState(0)

  const commonRedux = useSelector(
    state => state.common,
    (prevItem, item) => {
      if (item === prevItem) return true
      else return false
    }
  )

  const onResize = useCallback(() => {
    setWidth(target.current?.offsetWidth)
    setHeight(target.current?.offsetHeight)
  }, [target])

  useEffect(() => {
    window.addEventListener("resize", onResize)
    return () => {
      window.removeEventListener("resize", onResize)
    }
  }, [onResize])

  useEffect(() => {
    if (target.current) {
      setWidth(target.current?.offsetWidth)
      setHeight(target.current?.offsetHeight)
    }
  }, [target.current?.offsetWidth, target.current?.offsetHeight, commonRedux.collapse])

  return [width, height]
}

export default useResizeListener
