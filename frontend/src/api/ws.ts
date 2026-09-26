// WebSocket client for the live training stream.
export type TrainingEvent = Record<string, unknown> & { type: string }

export function connectTrainingWS(
  jobId: string,
  onEvent: (ev: TrainingEvent) => void,
  onClose?: () => void,
): WebSocket {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const ws = new WebSocket(`${proto}://${window.location.host}/ws/training/${jobId}`)
  ws.onmessage = (m) => {
    try {
      onEvent(JSON.parse(m.data) as TrainingEvent)
    } catch { /* ignore malformed frame */ }
  }
  ws.onclose = () => onClose?.()
  ws.onerror = () => ws.close()
  return ws
}
