import { SetStateAction, useCallback, useEffect, useRef, useState } from "react"

function useStateCallback(initialState) {
  const [state, _setState] = useState(initialState)

  const callbackRef = useRef()
  const isFirstCallbackCall = useRef(true)

  const setState = useCallback((setStateAction, callback) => {
    callbackRef.current = callback
    _setState(setStateAction)
  }, [])

  useEffect(() => {
    if (isFirstCallbackCall.current) {
      isFirstCallbackCall.current = false
      return
    }
    callbackRef.current?.(state)
  }, [state])

  return [state, setState]
}

export default useStateCallback
