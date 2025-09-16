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
  const [windowMeters, setWindowMeters] = useState<number>(200)
  const [taskSeconds, setTaskSeconds] = useState<number>(30)
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
      <div className="row" style={{ marginTop: 8 }}>
        <label style={{ marginTop: 0 }}>录像窗口(m)</label>
        <input type="number" min={0} step={10} value={windowMeters} onChange={(e) => setWindowMeters(Math.max(0, Number(e.target.value)||0))} style={{ width: 120 }} />
        <label style={{ marginTop: 0 }}>任务阈值(s)</label>
        <input type="number" min={0} step={5} value={taskSeconds} onChange={(e) => setTaskSeconds(Math.max(0, Number(e.target.value)||0))} style={{ width: 120 }} />
        <span style={{ color: '#6b7280' }}>(后端仍按30s，展示用)</span>
      </div>
      <EventTimeline
        events={[]}
        day={new Date().toISOString().slice(0,10)}
        presentColor="#10b981"
        roll={payload?.roll}
        recording={!!payload?.recording}
        recordingSegment={typeof payload?.recording_segment === 'string' ? payload?.recording_segment : ''}
        onDutySeconds={typeof payload?.on_duty_seconds === 'number' ? payload?.on_duty_seconds : 0}
        windowMeters={windowMeters}
        taskSeconds={taskSeconds}
      />
    </div>
  )}

type RollInfo = { coil_number?: string | number; total_m?: number; processed_m?: number; remaining_m?: number }

function EventTimeline({ events, day, presentColor = '#10b981', roll, recording, recordingSegment, onDutySeconds, windowMeters = 0, taskSeconds }: {
  events: any[];
  day: string;
  presentColor?: string;
  roll?: RollInfo;
  recording?: boolean;
  recordingSegment?: string;
  onDutySeconds?: number;
  windowMeters?: number;
  taskSeconds?: number;
}) {
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
      {(() => {
        try {
          const rollInfo = roll || {}
          const total = typeof rollInfo.total_m === 'number' ? rollInfo.total_m : NaN
          const processed = typeof rollInfo.processed_m === 'number' ? rollInfo.processed_m : NaN
          const remaining = typeof rollInfo.remaining_m === 'number' ? rollInfo.remaining_m : (Number.isFinite(total) && Number.isFinite(processed) ? Math.max(0, total - processed) : NaN)
          const coilNum = rollInfo?.coil_number ?? '—'
          const onDuty = typeof onDutySeconds === 'number' ? onDutySeconds : 0

          if (!Number.isFinite(total) || !Number.isFinite(processed)) {
            return <div style={{ marginTop: 8, color: '#6b7280' }}>无卷信息</div>
          }

          const width = 600, height = 80
          const circleX = 50, circleY = 30, radius = 18
          const baselineY = 50
          const px0 = circleX + radius + 8
          const endX = width - 20
          const usable = endX - px0
          const ratio = Math.max(0, Math.min(1, processed / total))
          const px = px0 + usable * ratio
          const win = Math.max(0, Number.isFinite(windowMeters) ? windowMeters : 0)
          const headEnd = px0 + usable * Math.max(0, Math.min(1, win / total))
          const tailStart = px0 + usable * Math.max(0, Math.min(1, (total - win) / total))

          return (
            <div style={{ marginTop: 8 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <svg width={width} height={height} style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 4 }}>
                  <g>
                    <line x1={circleX} y1={baselineY} x2={endX} y2={baselineY} stroke="#9ca3af" strokeWidth={2} />
                    <circle cx={circleX} cy={circleY} r={radius} fill="#3f51b5" stroke="#2c3e9e" />
                    <rect x={px0} y={baselineY - 6} width={Math.max(0, headEnd - px0)} height={12} fill="rgba(33,150,243,0.15)" />
                    <rect x={tailStart} y={baselineY - 6} width={Math.max(0, endX - tailStart)} height={12} fill="rgba(244,67,54,0.15)" />
                    <line x1={px0} y1={baselineY} x2={px} y2={baselineY} stroke="#22c55e" strokeWidth={4} />
                    <line x1={px} y1={baselineY - 8} x2={px} y2={baselineY + 8} stroke="#22c55e" strokeWidth={2} />
                    {recording ? <circle cx={18} cy={18} r={6} fill="#ef4444"><animate attributeName="r" values="5;7;5" dur="1s" repeatCount="indefinite"/></circle> : null}
                  </g>
                </svg>
                <div style={{ fontSize: 14, color: '#374151' }}>
                  <div><strong>卷</strong> #{coilNum} · <strong>已过</strong> {processed.toFixed(1)} m · <strong>剩余</strong> {Number.isFinite(remaining) ? remaining.toFixed(1) : '—'} m / <strong>共</strong> {total.toFixed(0)} m</div>
                  <div style={{ marginTop: 4, color: '#6b7280' }}>
                    <span style={{ display: 'inline-block', padding: '2px 6px', borderRadius: 9999, background: recording ? '#ef4444' : '#9ca3af', color: '#fff', marginRight: 8 }}>{recording ? `REC ${recordingSegment || ''}` : 'Idle'}</span>
                    <span>在岗累计：{onDuty.toFixed(1)} s</span>
                  </div>
                </div>
              </div>
            </div>
          )
        } catch {
          return null
        }
      })()}
    </div>
  )
}
