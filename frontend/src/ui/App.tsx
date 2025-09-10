import React, { useEffect, useMemo, useState } from 'react'
import { api } from '../utils/api'
import { useWebSocket } from '../utils/useWebSocket'

type Camera = { id: number; name: string; source: string; region: string; running: boolean; health?: any }

export function App() {
  const [cameras, setCameras] = useState<Camera[]>([])
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [snapshot, setSnapshot] = useState<any>(null)
  const [events, setEvents] = useState<any[]>([])
  const [day, setDay] = useState<string>(() => new Date().toISOString().slice(0, 10))
  const [createPayload, setCreatePayload] = useState({ name: '', source: '', region: '' })

  const selected = useMemo(() => cameras.find(c => c.id === selectedId) || null, [cameras, selectedId])

  async function loadCameras() {
    setLoading(true)
    try {
      const data = await api.get('/cameras')
      setCameras(data)
      if (data.length && !selectedId) setSelectedId(data[0].id)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { loadCameras() }, [])

  async function onCreate() {
    const body: any = {}
    if (createPayload.name) body.name = createPayload.name
    if (createPayload.source) body.source = createPayload.source
    if (createPayload.region) {
      try { body.region = JSON.parse(createPayload.region) } catch { alert('Invalid region JSON'); return }
    }
    await api.post('/cameras', body)
    setCreatePayload({ name: '', source: '', region: '' })
    await loadCameras()
  }

  async function onDelete(id: number) {
    await api.del(`/cameras/${id}`)
    await loadCameras()
  }

  async function onSnapshot() {
    if (!selectedId) return
    const data = await api.get(`/cameras/${selectedId}/snapshot`)
    setSnapshot(data)
  }

  async function onLoadEvents() {
    if (!selectedId) return
    const data = await api.get(`/cameras/${selectedId}/dwell_events`)
    setEvents(data)
  }

  return (
    <div className="app">
      <div className="card">
        <h3>相机</h3>
        <div className="row" style={{ marginBottom: 8 }}>
          <button className="btn" onClick={loadCameras} disabled={loading}>刷新</button>
        </div>
        <label>选择相机</label>
        <select value={selectedId ?? ''} onChange={(e) => setSelectedId(Number(e.target.value))}>
          {cameras.map(c => (
            <option key={c.id} value={c.id}>#{c.id} {c.name} ({c.running ? 'running' : 'stopped'})</option>
          ))}
        </select>
        <div className="row" style={{ marginTop: 8 }}>
          {selectedId && <button className="btn danger" onClick={() => onDelete(selectedId)}>删除</button>}
        </div>

        <h4 style={{ marginTop: 16 }}>创建相机</h4>
        <label>名称</label>
        <input value={createPayload.name} onChange={e => setCreatePayload(p => ({ ...p, name: e.target.value }))} />
        <label>源（0 或 rtsp://）</label>
        <input value={createPayload.source} onChange={e => setCreatePayload(p => ({ ...p, source: e.target.value }))} />
        <label>区域 JSON（[[x,y], ...]）</label>
        <input value={createPayload.region} onChange={e => setCreatePayload(p => ({ ...p, region: e.target.value }))} />
        <div className="row" style={{ marginTop: 8 }}>
          <button className="btn primary" onClick={onCreate}>创建</button>
        </div>

        <h4 style={{ marginTop: 16 }}>驻留事件</h4>
        <div className="row" style={{ marginBottom: 6 }}>
          <button className="btn" onClick={onLoadEvents} disabled={!selectedId}>拉取</button>
          <label style={{ margin: 0 }}>日期（24h）</label>
          <input type="date" value={day} onChange={(e) => setDay(e.target.value)} />
        </div>
        <EventTimeline events={events} day={day} />
        <div className="list">
          {events.length === 0 ? <div style={{ color: '#6b7280' }}>暂无</div> : (
            <ul>
              {events.map(e => (
                <li key={e.id}>
                  #{e.id} obj:{e.object_id} {new Date(e.start_ts).toLocaleString()} → {new Date(e.end_ts).toLocaleString()}
                  {e.video_path ? ` · video: ${e.video_path}` : ''}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      <div className="card">
        <LivePanel cameraId={selectedId} />
        <div className="row" style={{ marginTop: 8 }}>
          <button className="btn" onClick={onSnapshot} disabled={!selectedId}>抓拍</button>
        </div>
        {snapshot && (
          <div style={{ marginTop: 8 }}>
            {snapshot.frame && <img className="frame" src={`data:image/jpeg;base64,${snapshot.frame}`} />}
            <pre>{JSON.stringify({ ...snapshot, frame: snapshot.frame ? 'base64' : undefined }, null, 2)}</pre>
          </div>
        )}
      </div>
    </div>
  )
}

function LivePanel({ cameraId }: { cameraId: number | null }) {
  const [payload, setPayload] = useState<any>(null)
  const url = useMemo(() => {
    if (!cameraId) return null
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    return `${proto}://${location.host}/cameras/${cameraId}/status`
  }, [cameraId])

  const { readyState } = useWebSocket(url, (msg) => {
    try { setPayload(JSON.parse(msg)) } catch {}
  })

  const health = payload?.health
  const status = health?.status || 'init'
  const badgeClass = status === 'ok' ? 'ok' : status === 'degraded' ? 'degraded' : 'error'

  return (
    <div>
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <h3 style={{ margin: 0 }}>实时</h3>
        <span className={`badge ${badgeClass}`}>{status}</span>
      </div>
      {payload?.frame ? (
        <img className="frame" src={`data:image/jpeg;base64,${payload.frame}`} />
      ) : (
        <div style={{ color: '#6b7280' }}>等待数据（WS {readyState}）</div>
      )}
      <div style={{ marginTop: 8 }}>
        <strong>场景:</strong> {payload?.scene_active ? '有人在岗' : '无人'} · <strong>IDs:</strong> {payload?.active_ids?.join(', ') || '—'}
        {payload?.alarm && (
          <div className="badge error" style={{ marginLeft: 8 }}>告警 {payload?.alarm_message || ''}</div>
        )}
      </div>
    </div>
  )}

function EventTimeline({ events, day, presentColor = '#10b981' }: { events: any[]; day: string; presentColor?: string }) {
  // day: 'YYYY-MM-DD' local date
  const dayStart = new Date(day + 'T00:00:00').getTime()
  const dayEnd = dayStart + 24 * 60 * 60 * 1000

  const segments = (events || []).map((e) => {
    const s = new Date(e.start_ts).getTime()
    const t = new Date(e.end_ts).getTime()
    // clip to [dayStart, dayEnd]
    const start = Math.max(s, dayStart)
    const end = Math.min(t, dayEnd)
    return { start, end }
  }).filter(seg => seg.end > seg.start)

  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ position: 'relative', height: 16, background: '#e5e7eb', borderRadius: 8, overflow: 'hidden' }}>
        {segments.map((seg, idx) => {
          const left = ((seg.start - dayStart) / (dayEnd - dayStart)) * 100
          const width = ((seg.end - seg.start) / (dayEnd - dayStart)) * 100
          return (
            <div key={idx} style={{ position: 'absolute', left: left + '%', width: width + '%', top: 0, bottom: 0, background: presentColor }} />
          )
        })}
      </div>
      <div className="row" style={{ justifyContent: 'space-between', fontSize: 12, color: '#6b7280', marginTop: 4 }}>
        <span>00:00</span>
        <span>06:00</span>
        <span>12:00</span>
        <span>18:00</span>
        <span>24:00</span>
      </div>
    </div>
  )
}
