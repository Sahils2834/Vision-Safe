import { useEffect, useRef, useState, useCallback, type ChangeEvent } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

const API = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

type Alert = { id: string; worker_id: number; zone_id?: string; missing_ppe: string[]; severity: string; message?: string; created_at: string }
type Camera = { id: number; name: string }
type ModelInfo = { id: string; name: string; loaded: boolean; active: boolean; classes: number; device?: string; fps_estimate?: number; description?: string }
type ModelsStatus = { status: string; active_model: string; device: string; device_name: string; models: ModelInfo[]; models_loaded_count: number }
type Health = { status: string; model_ready: boolean; ppe_model_ready: boolean; model_name?: string; active_model?: string; active_streams: number; device?: string; device_name?: string }
type Telemetry = { fps: number; worker_count: number; violation_count: number; compliance_rate: number; active_model: string; device: string; alert_count: number }
type WorkerItem = { id: number; box: number[]; confidence: number; head_zone: number[]; is_safe: boolean; has_helmet: boolean; helmet_confidence: number; matched_helmet_box?: number[] | null; has_gloves?: boolean; gloves_confidence?: number; has_vest?: boolean; vest_confidence?: number; has_boots?: boolean; boots_confidence?: number; has_goggles?: boolean; goggles_confidence?: number; detected_ppe?: string[]; missing_ppe: string[]; status: string }
type HelmetItem = { box: number[]; confidence: number; matched_worker_id?: number | null }
type PPEItem = { label: string; box: number[]; confidence: number; matched_worker_id?: number | null }
type DiagnosisResponse = { success: boolean; active_model: string; workers: WorkerItem[]; helmets: HelmetItem[]; gloves?: PPEItem[]; vests?: PPEItem[]; boots?: PPEItem[]; goggles?: PPEItem[]; all_ppe?: PPEItem[]; safe_workers_count: number; total_workers_count: number; violation_count: number; compliance_rate: number; latency_ms: number; safety_index: string; annotated_image_base64?: string }
type BenchmarkData = Record<string, { mAP50: number; precision: number; recall: number; f1: number }>

function ppeIcon(ppe: string) {
  const p = ppe.toLowerCase()
  if (p.includes('helmet') || p.includes('hardhat')) return '🪖'
  if (p.includes('glove'))  return '🧤'
  if (p.includes('vest') || p.includes('kit')) return '🦺'
  if (p.includes('boot') || p.includes('shoe')) return '🥾'
  if (p.includes('goggle')) return '🥽'
  return '⚠️'
}

function ppeName(ppe: string) {
  const p = ppe.toLowerCase()
  if (p.includes('helmet') || p.includes('hardhat')) return 'No Helmet'
  if (p.includes('glove'))  return 'No Gloves'
  if (p.includes('vest') || p.includes('kit')) return 'No Safety Vest'
  if (p.includes('boot') || p.includes('shoe')) return 'No Safety Boots'
  if (p.includes('goggle')) return 'No Goggles'
  return `Missing: ${ppe}`
}

function Clock() {
  const [t, setT] = useState(new Date())
  useEffect(() => { const id = setInterval(() => setT(new Date()), 1000); return () => clearInterval(id) }, [])
  const day = t.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' })
  const time = t.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
  return <span className="topbar-time">{day} &nbsp;·&nbsp; {time}</span>
}

const I = (d: string) => <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" style={{width:'17px',height:'17px',flexShrink:0}}><path d={d}/></svg>
const IconDashboard = () => <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" style={{width:'17px',height:'17px',flexShrink:0}}><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>
const IconReports = () => I("M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z")
const IconCamera  = () => I("M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z")
const IconUsers   = () => I("M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z")
const IconShield  = () => I("M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z")
const IconAlert   = () => I("M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z")
const IconVideo   = () => I("M7 4v16M17 4v16M3 8h4m10 0h4M3 12h18M3 16h4m10 0h4M4 20h16a1 1 0 001-1V5a1 1 0 00-1-1H4a1 1 0 00-1 1v14a1 1 0 001 1z")
const IconZap     = () => I("M13 10V3L4 14h7v7l9-11h-7z")
const IconRefresh = () => I("M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15")
const IconUpload  = () => I("M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12")
const IconPhoto   = () => I("M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z")
const IconCpu     = () => <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" style={{width:'14px',height:'14px'}}><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M1 9h3M1 15h3M20 9h3M20 15h3"/></svg>
const IconStop    = () => <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" style={{width:'14px',height:'14px'}}><rect x="3" y="3" width="18" height="18" rx="3"/></svg>
const IconPlay    = () => I("M10 8l6 4-6 4V8z")

function App() {
  const [page, setPage] = useState<'dashboard' | 'reports' | 'inspector'>('dashboard')
  const [health, setHealth] = useState<Health | null>(null)
  const [modelsStatus, setModelsStatus] = useState<ModelsStatus | null>(null)
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null)
  const [cameras, setCameras] = useState<Camera[]>([])
  const [camera, setCamera] = useState(0)
  const [stream, setStream] = useState('')
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [busy, setBusy] = useState(false)
  const [switchingModel, setSwitchingModel] = useState(false)
  const [inputMode, setInputMode] = useState<'camera' | 'video'>('camera')
  const [diagResult, setDiagResult] = useState<DiagnosisResponse | null>(null)
  const [diagBusy, setDiagBusy] = useState(false)
  const [diagPreview, setDiagPreview] = useState<string | null>(null)
  const [benchmarks, setBenchmarks] = useState<BenchmarkData | null>(null)
  const uploadStreamRef = useRef<HTMLInputElement>(null)
  const uploadDiagRef   = useRef<HTMLInputElement>(null)

  const request = async (path: string, options?: RequestInit) => {
    const res = await fetch(`${API}${path}`, options)
    if (!res.ok) { const e = await res.json().catch(() => null); throw new Error(e?.detail ?? `HTTP ${res.status}`) }
    return res.json()
  }

  const refreshData = useCallback(async () => {
    try {
      const [h, m, t, a] = await Promise.all([request('/health') as Promise<Health>, request('/models/status') as Promise<ModelsStatus>, request('/telemetry') as Promise<Telemetry>, request('/alerts') as Promise<Alert[]>])
      setHealth(h); setModelsStatus(m); setTelemetry(t); setAlerts(a)
    } catch { /* offline */ }
  }, [])

  const scanCameras = async () => { setBusy(true); try { setCameras(await request('/cameras') as Camera[]) } catch { setCameras([]) } finally { setBusy(false) } }
  const switchModel = async (key: string) => { setSwitchingModel(true); try { await request('/models/select', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model_key: key }) }); await refreshData() } catch(e) { console.error(e) } finally { setSwitchingModel(false) } }

  useEffect(() => {
    refreshData(); scanCameras()
    request('/api/benchmark/baseline').then(d => setBenchmarks(d as BenchmarkData)).catch(() => {})
    const id = setInterval(refreshData, 2500)
    return () => clearInterval(id)
  }, [refreshData])

  useEffect(() => {
    if (!stream) return
    const id = setInterval(() => { const img = document.getElementById('live-feed') as HTMLImageElement | null; if (img) img.src = `${API}/streams/${stream}/frame?t=${Date.now()}` }, 200)
    return () => clearInterval(id)
  }, [stream])

  const startCamera = async () => { setBusy(true); try { if (stream) await request(`/streams/${stream}/stop`, { method: 'POST' }); const r = await request('/streams/webcam/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ camera_index: camera }) }); setStream(r.stream_id); refreshData() } catch { setStream('') } finally { setBusy(false) } }
  const chooseVideo = async (e: ChangeEvent<HTMLInputElement>) => { const file = e.target.files?.[0]; if (!file) return; setBusy(true); try { if (stream) await request(`/streams/${stream}/stop`, { method: 'POST' }); const fd = new FormData(); fd.append('file', file); const r = await request('/streams/video', { method: 'POST', body: fd }); setStream(r.stream_id); refreshData() } catch { setStream('') } finally { setBusy(false); e.target.value = '' } }
  const stopStream = async () => { if (!stream) return; await request(`/streams/${stream}/stop`, { method: 'POST' }); setStream(''); refreshData() }
  const handlePhotoUpload = async (e: ChangeEvent<HTMLInputElement>) => { const file = e.target.files?.[0]; if (!file) return; setDiagPreview(URL.createObjectURL(file)); setDiagBusy(true); try { const fd = new FormData(); fd.append('file', file); const res = await request('/api/diagnose', { method: 'POST', body: fd }) as DiagnosisResponse; setDiagResult(res); setPage('inspector') } catch(err) { console.error(err) } finally { setDiagBusy(false) } }

  const online = health?.status === 'ok'
  const activeModelKey = modelsStatus?.active_model ?? health?.active_model ?? 'yolo11n'
  const workerCount = telemetry?.worker_count ?? 0
  const violationCount = telemetry?.violation_count ?? 0
  const complianceRate = telemetry?.compliance_rate ?? 100
  const activeStreams = health?.active_streams ?? 0

  const Topbar = ({ title, subtitle }: { title: string; subtitle: string }) => (
    <div className="topbar">
      <div className="topbar-title"><h2>{title}</h2><p>{subtitle}</p></div>
      <div className="topbar-actions">
        <Clock />
        {activeModelKey && <div className="model-badge"><IconCpu /> {activeModelKey.toUpperCase()}</div>}
      </div>
    </div>
  )

  const Dashboard = () => (
    <>
      <Topbar title="Industrial Safety Dashboard" subtitle="Monitor PPE compliance and detect workplace safety violations in real time." />
      <div className="page-body">
        <div className="summary-cards">
          <div className="summary-card"><div className="sc-icon blue"><IconUsers /></div><div className="sc-data"><div className="sc-value">{workerCount}</div><div className="sc-label">Total Workers</div><div className="sc-sub">{stream ? 'Currently detected' : 'Start monitoring'}</div></div></div>
          <div className="summary-card"><div className="sc-icon green"><IconShield /></div><div className="sc-data"><div className={`sc-value ${complianceRate >= 80 ? 'text-green' : 'text-red'}`}>{complianceRate.toFixed(0)}%</div><div className="sc-label">PPE Compliant</div><div className="sc-sub">{complianceRate >= 80 ? 'Site is compliant' : 'Action required'}</div></div></div>
          <div className="summary-card"><div className="sc-icon red"><IconAlert /></div><div className="sc-data"><div className={`sc-value ${violationCount > 0 ? 'text-red' : 'text-green'}`}>{violationCount}</div><div className="sc-label">Active Violations</div><div className="sc-sub">{alerts.length} total logged</div></div></div>
          <div className="summary-card"><div className="sc-icon purple"><IconCamera /></div><div className="sc-data"><div className="sc-value">{activeStreams > 0 ? `${activeStreams}` : cameras.length > 0 ? `0 / ${cameras.length}` : '--'}</div><div className="sc-label">Active Cameras</div><div className="sc-sub">{stream ? 'Monitoring live' : 'No active feed'}</div></div></div>
        </div>

        <div className="monitor-section">
          <div className="monitor-card">
            <div className="monitor-header">
              <div className="monitor-header-left">
                <IconCamera />
                <span className="monitor-source">{stream ? (inputMode === 'camera' ? `Camera ${camera} — Construction Site` : 'Video Feed') : 'No Source Selected'}</span>
                {stream && <div className="live-badge"><span className="pulse"/> LIVE</div>}
              </div>
              <div className="monitor-header-right">
                {stream && telemetry && <span style={{fontSize:'11px',color:'var(--text-muted)',fontWeight:600}}>{telemetry.fps.toFixed(0)} FPS</span>}
              </div>
            </div>
            <div className="monitor-video">
              {stream ? (
                <img id="live-feed" src={`${API}/streams/${stream}/frame`} alt="Live PPE stream" />
              ) : (
                <div className="monitor-empty">
                  <div className="monitor-empty-icon"><IconCamera /></div>
                  <h3>No Active Feed</h3>
                  <p>Start a camera or upload a video file to begin real-time PPE monitoring with AI detection overlays.</p>
                  <div style={{display:'flex',gap:'10px',justifyContent:'center'}}>
                    <button className="btn btn-primary btn-sm" onClick={startCamera} disabled={!online||busy}><IconPlay /> Camera</button>
                    <button className="btn btn-secondary btn-sm" onClick={()=>uploadStreamRef.current?.click()} disabled={!online||busy}><IconVideo /> Upload Video</button>
                  </div>
                </div>
              )}
            </div>
            <div className="monitor-footer">
              {stream ? (
                <>
                  <button className="btn btn-secondary btn-sm" onClick={()=>setInputMode(m=>m==='camera'?'video':'camera')}><IconRefresh /> Switch Source</button>
                  <button className="btn btn-danger btn-sm" style={{marginLeft:'auto'}} onClick={stopStream}><IconStop /> Stop Monitoring</button>
                </>
              ) : (
                <>
                  <button className="btn btn-primary btn-sm" onClick={startCamera} disabled={!online||busy}><IconPlay /> {busy?'Starting...':'Start Camera'}</button>
                  <button className="btn btn-secondary btn-sm" onClick={()=>uploadStreamRef.current?.click()} disabled={!online||busy}><IconUpload /> Upload Video</button>
                  <input ref={uploadStreamRef} type="file" accept="video/*" onChange={chooseVideo} hidden />
                </>
              )}
            </div>
          </div>

          <div className="controls-card">
            <div className="controls-header">📡 Source &amp; Controls</div>
            <div className="controls-body">
              <div>
                <div className="ctrl-label">Input Source</div>
                <div className="input-tabs-row">
                  <button className={`input-tab-btn ${inputMode==='camera'?'active':''}`} onClick={()=>setInputMode('camera')}><IconCamera /> Camera</button>
                  <button className={`input-tab-btn ${inputMode==='video'?'active':''}`} onClick={()=>setInputMode('video')}><IconVideo /> Video</button>
                </div>
              </div>
              {inputMode === 'camera' ? (
                <div>
                  <div className="ctrl-label">Select Camera</div>
                  <select className="ctrl-select" value={camera} onChange={e=>setCamera(Number(e.target.value))} disabled={!online||busy}>
                    {cameras.length ? cameras.map(c=><option key={c.id} value={c.id}>{c.name}</option>) : <option value={0}>Camera 0 (Default)</option>}
                  </select>
                  <div style={{display:'flex',gap:'6px',marginTop:'8px'}}>
                    <button className="btn btn-secondary btn-sm" style={{flex:1}} onClick={scanCameras} disabled={!online||busy}><IconRefresh /> Scan</button>
                    <button className="btn btn-primary btn-sm" style={{flex:1}} onClick={startCamera} disabled={!online||busy}><IconPlay /> {busy?'Starting...':'Start'}</button>
                  </div>
                  <div style={{fontSize:'11px',color:'var(--text-muted)',marginTop:'6px'}}>✓ {cameras.length} camera(s) found</div>
                </div>
              ) : (
                <div>
                  <div className="ctrl-label">Upload Video</div>
                  <button className="btn btn-secondary btn-full" onClick={()=>uploadStreamRef.current?.click()} disabled={!online||busy}><IconUpload /> Select Video File</button>
                  <input ref={uploadStreamRef} type="file" accept="video/*" onChange={chooseVideo} hidden />
                  <div style={{fontSize:'11px',color:'var(--text-muted)',marginTop:'6px'}}>Supports .mp4 .avi .mkv</div>
                </div>
              )}
              {stream && <button className="btn btn-danger btn-full btn-sm" onClick={stopStream}><IconStop /> Stop Monitoring</button>}
              <hr className="divider"/>
              <div>
                <div className="ctrl-label">Live Stats</div>
                <div className="telemetry-mini">
                  <div className="tel-row"><span className="tel-row-label"><IconUsers/> Workers</span><span className="tel-row-value blue">{workerCount}</span></div>
                  <div className="tel-row"><span className="tel-row-label"><IconAlert/> Violations</span><span className={`tel-row-value ${violationCount>0?'red':'green'}`}>{violationCount}</span></div>
                  <div className="tel-row"><span className="tel-row-label"><IconShield/> Compliance</span><span className={`tel-row-value ${complianceRate>=80?'green':'red'}`}>{complianceRate.toFixed(0)}%</span></div>
                  <div className="tel-row"><span className="tel-row-label"><IconZap/> FPS</span><span className="tel-row-value">{telemetry?.fps?telemetry.fps.toFixed(0):'--'}</span></div>
                </div>
              </div>
              <hr className="divider"/>
              <div>
                <div className="ctrl-label">AI Model</div>
                <div className="model-selector">
                  {modelsStatus?.models.map(m=>(
                    <button key={m.id} className={`model-opt-btn ${m.id===activeModelKey?'active':''}`} onClick={()=>switchModel(m.id)} disabled={switchingModel}>
                      <span>{m.name}</span><span className="model-fps">{m.fps_estimate} FPS</span>
                    </button>
                  ))}
                </div>
              </div>
              <hr className="divider"/>
              <div>
                <div className="ctrl-label">Photo Inspector</div>
                <button className="btn btn-ghost btn-full btn-sm" onClick={()=>uploadDiagRef.current?.click()} disabled={diagBusy}><IconPhoto /> {diagBusy?'Analyzing...':'Inspect a Photo'}</button>
                <input ref={uploadDiagRef} type="file" accept="image/*" onChange={handlePhotoUpload} hidden />
              </div>
            </div>
          </div>
        </div>

        <div className="detections-section">
          <div className="detections-header">
            <h3>🔔 Recent Safety Alerts</h3>
            <button className="view-all-link" onClick={()=>setPage('reports')}>View All Reports →</button>
          </div>
          {alerts.length===0 ? (
            <div className="detections-empty"><div className="empty-icon">🟢</div><p>No violations detected yet.<br/>Start a camera feed to begin monitoring.</p></div>
          ) : (
            <div className="detections-grid">
              {alerts.slice(0,6).map(alert=>{
                const ppe=alert.missing_ppe[0]??''
                return (
                  <div key={alert.id} className="detection-card">
                    <div className="detection-thumb" style={{background:alert.missing_ppe.length>0?'rgba(239,68,68,0.07)':'rgba(16,185,129,0.07)'}}><span style={{fontSize:'32px'}}>{ppeIcon(ppe)}</span></div>
                    <div className="detection-info">
                      <div className="detection-type">{alert.missing_ppe.length>0?'🔴':'🟢'} {ppeName(ppe)}</div>
                      <div className="detection-meta">Worker #{alert.worker_id}<br/>{new Date(alert.created_at).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit'})}</div>
                      <div className={`detection-status ${alert.missing_ppe.length>0?'violation':'compliant'}`}>{alert.missing_ppe.length>0?'⚠ Violation':'✓ Compliant'}</div>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>
    </>
  )

  const Reports = () => {
    const [filter, setFilter] = useState('all')
    const helmets = alerts.filter(a=>a.missing_ppe.some(p=>p.toLowerCase().includes('helmet')))
    const gloves  = alerts.filter(a=>a.missing_ppe.some(p=>p.toLowerCase().includes('glove')))
    const vests   = alerts.filter(a=>a.missing_ppe.some(p=>p.toLowerCase().includes('vest')||p.toLowerCase().includes('kit')))
    const boots   = alerts.filter(a=>a.missing_ppe.some(p=>p.toLowerCase().includes('boot')||p.toLowerCase().includes('shoe')))
    const filtered = filter==='all' ? alerts : filter==='critical' ? alerts.filter(a=>a.severity==='critical') : filter==='warning' ? alerts.filter(a=>a.severity==='warning') : alerts.filter(a=>a.missing_ppe.some(p=>p.toLowerCase().includes(filter)))
    return (
      <>
        <Topbar title="Reports" subtitle="Review PPE detections, compliance results, and workplace safety violations." />
        <div className="page-body">
          <div className="reports-page">
            <div className="reports-summary">
              <div className="rpt-card"><div className="rpt-card-icon">📋</div><div className="rpt-card-value">{alerts.length}</div><div className="rpt-card-label">Total Violations</div><div className="rpt-card-sub">All detected violations</div></div>
              <div className="rpt-card"><div className="rpt-card-icon">🪖</div><div className="rpt-card-value">{helmets.length}</div><div className="rpt-card-label">Helmet Violations</div><div className="rpt-card-sub">No hardhat detected</div></div>
              <div className="rpt-card"><div className="rpt-card-icon">🧤</div><div className="rpt-card-value">{gloves.length}</div><div className="rpt-card-label">Glove Violations</div><div className="rpt-card-sub">No gloves detected</div></div>
              <div className="rpt-card"><div className="rpt-card-icon">🦺</div><div className="rpt-card-value">{vests.length}</div><div className="rpt-card-label">Vest Violations</div><div className="rpt-card-sub">No safety vest detected</div></div>
            </div>
            <div className="logs-card">
              <div className="logs-header">
                <h3>Detection Logs</h3>
                <div className="logs-header-actions">
                  <select className="logs-filter-select" value={filter} onChange={e=>setFilter(e.target.value)}>
                    <option value="all">All Violations</option>
                    <option value="critical">Critical Only</option>
                    <option value="warning">Warnings Only</option>
                    <option value="helmet">🪖 No Helmet</option>
                    <option value="glove">🧤 No Gloves</option>
                    <option value="vest">🦺 No Safety Vest</option>
                    <option value="boot">🥾 No Safety Boots</option>
                    <option value="goggle">🥽 No Goggles</option>
                  </select>
                  <button className="btn btn-secondary btn-sm" onClick={refreshData}><IconRefresh /> Refresh</button>
                </div>
              </div>
              {filtered.length===0 ? (
                <div className="logs-empty"><div className="empty-icon">📭</div><p>No violations in this category yet.</p><small>Start monitoring to detect safety violations in real time.</small></div>
              ) : (
                <div className="logs-table-wrap">
                  <table className="logs-table">
                    <thead><tr><th>Time</th><th>Worker</th><th>Detection / Violation</th><th>Severity</th><th>Status</th><th>Details</th></tr></thead>
                    <tbody>
                      {filtered.map(alert=>(
                        <tr key={alert.id}>
                          <td style={{whiteSpace:'nowrap',color:'var(--text-muted)',fontSize:'12px'}}>{new Date(alert.created_at).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit'})}</td>
                          <td style={{fontWeight:600}}>Worker #{alert.worker_id}</td>
                          <td><div style={{display:'flex',flexWrap:'wrap',gap:'4px'}}>{alert.missing_ppe.map((p,i)=><span key={i} className="logs-ppe-badge violation">{ppeIcon(p)} {ppeName(p)}</span>)}</div></td>
                          <td><span className={`logs-alert-severity ${alert.severity}`}>{alert.severity}</span></td>
                          <td><span className="logs-status-badge violation">⚠ Violation</span></td>
                          <td style={{color:'var(--text-muted)',fontSize:'12px',maxWidth:'200px'}}>{alert.message?alert.message.replace(/Worker #\d+ — /,''):`Missing: ${alert.missing_ppe[0]??''}`}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
            {benchmarks && (
              <div className="logs-card">
                <div className="logs-header"><h3>🤖 AI Model Benchmarks</h3></div>
                <div className="logs-table-wrap">
                  <table className="logs-table">
                    <thead><tr><th>Model</th><th>mAP@50</th><th>Precision</th><th>Recall</th><th>F1 Score</th></tr></thead>
                    <tbody>
                      {Object.entries(benchmarks).map(([model,stats])=>(
                        <tr key={model}>
                          <td style={{fontWeight:700,textTransform:'uppercase'}}>{model}</td>
                          <td className="logs-conf">{(stats.mAP50*100).toFixed(1)}%</td>
                          <td className="logs-conf">{(stats.precision*100).toFixed(1)}%</td>
                          <td className="logs-conf">{(stats.recall*100).toFixed(1)}%</td>
                          <td className="logs-conf">{(stats.f1*100).toFixed(1)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
      </>
    )
  }

  const Inspector = () => (
    <>
      <Topbar title="Photo Inspector" subtitle="Upload a site photo for detailed AI-powered PPE spatial analysis." />
      <div className="page-body">
        <div className="inspector-page">
          <div className="inspector-main-card">
            <div className="inspector-header">
              <h3>📷 Inspection Image</h3>
              <button className="btn btn-secondary btn-sm" onClick={()=>uploadDiagRef.current?.click()} disabled={diagBusy}><IconUpload /> {diagBusy?'Analyzing...':'Upload New Photo'}</button>
              <input ref={uploadDiagRef} type="file" accept="image/*" onChange={handlePhotoUpload} hidden />
            </div>
            <div className="inspector-image-area">
              {diagBusy ? (
                <div style={{textAlign:'center',color:'#94A3B8'}}><div style={{fontSize:'40px',marginBottom:'12px'}}>⚡</div><p>Running multi-PPE spatial analysis...</p></div>
              ) : diagResult?.annotated_image_base64 ? (
                <img src={diagResult.annotated_image_base64} alt="Annotated PPE Inspection"/>
              ) : diagPreview ? (
                <img src={diagPreview} alt="Uploaded preview"/>
              ) : (
                <div style={{textAlign:'center',color:'#94A3B8',padding:'3rem'}}>
                  <div style={{fontSize:'48px',marginBottom:'16px'}}>📸</div>
                  <h3 style={{color:'#CBD5E1',fontSize:'18px',fontWeight:600,marginBottom:'8px'}}>No Image Uploaded</h3>
                  <p style={{fontSize:'13px',marginBottom:'20px',color:'#64748B',lineHeight:1.5}}>Upload a construction site photo to detect PPE compliance for all workers.</p>
                  <button className="btn btn-primary" onClick={()=>uploadDiagRef.current?.click()}><IconUpload /> Select Photo</button>
                </div>
              )}
            </div>
          </div>
          <div className="inspector-sidebar-card">
            <div className="inspector-sidebar-header">Inspection Audit</div>
            <div className="inspector-sidebar-body">
              {diagResult ? (
                <>
                  <div className="audit-stat"><span className="audit-stat-label">AI Model</span><span className="audit-stat-value">{diagResult.active_model.toUpperCase()}</span></div>
                  <div className="audit-stat"><span className="audit-stat-label">Compliance Rate</span><span className={`audit-stat-value ${diagResult.compliance_rate>=80?'text-green':'text-red'}`}>{diagResult.compliance_rate.toFixed(1)}%</span></div>
                  <div className="audit-stat"><span className="audit-stat-label">Safety Status</span><span className={`audit-stat-value ${diagResult.safety_index==='SAFE'?'text-green':'text-red'}`}>{diagResult.safety_index==='SAFE'?'✅ SAFE':'⚠️ '+diagResult.safety_index}</span></div>
                  <div className="audit-stat"><span className="audit-stat-label">Latency</span><span className="audit-stat-value">{diagResult.latency_ms.toFixed(0)} ms</span></div>
                  <div className="section-label" style={{marginTop:'4px'}}>Detected PPE Items</div>
                  <div className="ppe-grid">
                    {[['🪖 Helmets',diagResult.helmets?.length??0],['🧤 Gloves',diagResult.gloves?.length??0],['🦺 Vests',diagResult.vests?.length??0],['🥾 Boots',diagResult.boots?.length??0],['🥽 Goggles',diagResult.goggles?.length??0],['👷 Workers',diagResult.workers.length]].map(([label,count])=>(
                      <div key={label as string} className="ppe-count-cell"><span>{label}</span><strong>{count}</strong></div>
                    ))}
                  </div>
                  <div className="section-label" style={{marginTop:'4px'}}>Workers ({diagResult.workers.length})</div>
                  <div className="worker-audit-list">
                    {diagResult.workers.map(w=>(
                      <div key={w.id} className={`worker-audit-item ${w.is_safe?'safe':'unsafe'}`}>
                        <div className="wai-header"><b>Worker #{w.id}</b><span className="wai-badge">{w.status}</span></div>
                        <div className="wai-info">
                          <div>PPE: {w.detected_ppe&&w.detected_ppe.length>0?w.detected_ppe.join(', '):'None'}</div>
                          {w.missing_ppe.length>0&&<div className="wai-missing">Missing: {w.missing_ppe.join(', ')}</div>}
                          <div style={{color:'var(--text-light)',fontSize:'10.5px',marginTop:'2px'}}>Confidence: {(w.confidence*100).toFixed(0)}%</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </>
              ) : (
                <div className="empty-state" style={{padding:'32px 16px'}}><div className="es-icon">🔍</div><h4>No Analysis Yet</h4><p>Upload a photo above to see detailed PPE analysis.</p></div>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  )

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="brand-icon">⛑️</div>
          <div className="brand-text"><h1>VisionSafe</h1><p>Industrial Safety Monitor</p></div>
        </div>
        <nav className="sidebar-nav">
          <div className="nav-section-label">Main</div>
          <button className={`nav-item ${page==='dashboard'?'active':''}`} onClick={()=>setPage('dashboard')}><IconDashboard /> Dashboard</button>
          <button className={`nav-item ${page==='reports'?'active':''}`} onClick={()=>setPage('reports')}><IconReports /> Reports</button>
          <div className="nav-section-label" style={{marginTop:'14px'}}>Tools</div>
          <button className={`nav-item ${page==='inspector'?'active':''}`} onClick={()=>setPage('inspector')}><IconPhoto /> Photo Inspector</button>
        </nav>
        <div className="sidebar-footer">
          <div className="system-status">
            <div className={`status-dot ${online?'online':'offline'}`}/>
            <div className="status-label">AI Engine: <span>{online?'Online':'Offline'}</span></div>
          </div>
          {online&&health?.device_name&&<div style={{marginTop:'6px',fontSize:'11px',color:'var(--text-light)'}}>🖥️ {health.device_name}</div>}
        </div>
      </aside>
      <div className="main-content">
        {page==='dashboard' && <Dashboard />}
        {page==='reports'   && <Reports />}
        {page==='inspector' && <Inspector />}
      </div>
    </div>
  )
}

createRoot(document.getElementById('root')!).render(<App />)
