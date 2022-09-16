import { useState, useEffect, useCallback } from "react"

function useListHeight() {
  const [listHeight, setListHeight] = useState(document.documentElement.clientHeight - 175)

  const onResize = useCallback(() => {
    setListHeight(document.documentElement.clientHeight - 175)
  }, [document.documentElement.clientHeight])

  useEffect(() => {
    window.addEventListener("resize", onResize)
    return () => {
      window.removeEventListener("resize", onResize)
    }
  }, [onResize])

  return listHeight
}

export default useListHeight
