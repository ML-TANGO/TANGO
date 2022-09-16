import { useEffect, useCallback } from "react"

export const useDebouncedEffect = (effect, delay, deps) => {
  const callback = useCallback(effect, deps)

  useEffect(() => {
    const handler = setTimeout(() => {
      callback()
    }, delay)

    return () => {
      clearTimeout(handler)
    }
  }, [callback, delay])
}

export default useDebouncedEffect
