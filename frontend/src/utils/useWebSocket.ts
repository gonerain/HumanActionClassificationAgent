import { useEffect, useRef, useState } from 'react'

export function useWebSocket(url: string | null, onMessage: (data: string) => void) {
  const wsRef = useRef<WebSocket | null>(null)
  const [readyState, setReadyState] = useState<number>(0)
  const timer = useRef<number | null>(null)

  useEffect(() => {
    if (!url) return
    let stopped = false

    function connect() {
      if (stopped) return
      try {
        const ws = new WebSocket(url)
        wsRef.current = ws
        ws.onopen = () => setReadyState(ws.readyState)
        ws.onmessage = (ev) => onMessage(typeof ev.data === 'string' ? ev.data : '')
        ws.onerror = () => setReadyState(ws.readyState)
        ws.onclose = () => {
          setReadyState(ws.readyState)
          // reconnect with backoff
          timer.current = window.setTimeout(connect, 1000)
        }
      } catch {
        timer.current = window.setTimeout(connect, 1500)
      }
    }

    connect()
    return () => {
      stopped = true
      if (timer.current) window.clearTimeout(timer.current)
      try { wsRef.current?.close() } catch {}
      wsRef.current = null
    }
  }, [url])

  return { readyState }
}

