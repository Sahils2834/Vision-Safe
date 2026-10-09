import { useEffect, useRef, useState, type ChangeEvent } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

const API = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

type Alert = { 
  id: string; 
  worker_id: number; 
  zone_id?: string; 
  missing_ppe: string[]; 
  severity: string; 
  message?: string; 
  created_at: string 
}

type Camera = { id: number; name: string }

type ModelInfo = {
  id: string;
  name: string;
  loaded: boolean;
  active: boolean;
  classes: number;
  device?: string;
  fps_estimate?: number;
  description?: string;
}

type ModelsStatus = {
  status: string;
  active_model: string;
  device: string;
  device_name: string;
  models: ModelInfo[];
  models_loaded_count: number;
}

type Health = { 
  status: string; 
  model_ready: boolean; 
  ppe_model_ready: boolean; 
  model_name?: string; 
  active_model?: string;
  active_streams: number;
  device?: string;
  device_name?: string;
}

type Telemetry = {
  fps: number;
  worker_count: number;
  violation_count: number;
  compliance_rate: number;
  active_model: string;
  device: string;
  alert_count: number;
}

type WorkerItem = {
  id: number;
  box: number[];
  confidence: number;
  head_zone: number[];
  is_safe: boolean;
  has_helmet: boolean;
  helmet_confidence: number;
  matched_helmet_box?: number[] | null;
  missing_ppe: string[];
  status: string;
}

type HelmetItem = {
  box: number[];
  confidence: number;
  matched_worker_id?: number | null;
}

type DiagnosisResponse = {
  success: boolean;
  active_model: string;
  workers: WorkerItem[];
  helmets: HelmetItem[];
  safe_workers_count: number;
  total_workers_count: number;
  violation_count: number;
  compliance_rate: number;
  latency_ms: number;
  safety_index: string;
  annotated_image_base64?: string;
}

type BenchmarkData = Record<string, {
  mAP50: number;
  'mAP50-95'?: number;
  mAP50_95?: number;
  precision: number;
  recall: number;
  f1: number;
}>

function App() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'benchmark'>('dashboard')
  const [health, setHealth] = useState<Health | null>(null)
  const [modelsStatus, setModelsStatus] = useState<ModelsStatus | null>(null)
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null)
  const [cameras, setCameras] = useState<Camera[]>([])
  const [camera, setCamera] = useState(0)
  const [stream, setStream] = useState('')
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [busy, setBusy] = useState(false)
  const [switchingModel, setSwitchingModel] = useState(false)
  
  // Input Mode (Camera, Video)
  const [inputMode, setInputMode] = useState<'camera' | 'video'>('camera')

  // Single-image diagnosis state
  const [diagFile, setDiagFile] = useState<File | null>(null)
  const [diagResult, setDiagResult] = useState<DiagnosisResponse | null>(null)
  const [diagBusy, setDiagBusy] = useState(false)
  const [diagPreview, setDiagPreview] = useState<string | null>(null)

  // Benchmark data state
  const [benchmarks, setBenchmarks] = useState<BenchmarkData | null>(null)

  const uploadStreamRef = useRef<HTMLInputElement>(null)
  const uploadDiagRef = useRef<HTMLInputElement>(null)

  const request = async (path: string, options?: RequestInit) => {
    const response = await fetch(`${API}${path}`, options)
    if (!response.ok) {
      const err = await response.json().catch(() => null)
      throw new Error(err?.detail ?? `Request failed (${response.status})`)
    }
    return response.json()
  }

  const refreshData = async () => {
    try {
      const h = await request('/health') as Health
      setHealth(h)
      
      const m = await request('/models/status') as ModelsStatus
      setModelsStatus(m)

      const t = await request('/telemetry') as Telemetry
      setTelemetry(t)

      const a = await request('/alerts') as Alert[]
      setAlerts(a)
    } catch {
      // offline
    }
  }

  const fetchBenchmarks = async () => {
    try {
      const data = await request('/api/benchmark/baseline') as BenchmarkData
      setBenchmarks(data)
    } catch {
      // fallback
    }
  }

  const scanCameras = async () => {
    setBusy(true)
    try {
      const found = await request('/cameras') as Camera[]
      setCameras(found)
    } catch {
      setCameras([])
    } finally {
      setBusy(false)
    }
  }

  const switchModel = async (modelKey: string) => {
    setSwitchingModel(true)
    try {
      await request('/models/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_key: modelKey })
      })
      await refreshData()
    } catch (err) {
      console.error('Failed to switch model:', err)
    } finally {
      setSwitchingModel(false)
    }
  }

  useEffect(() => {
    refreshData()
    scanCameras()
    fetchBenchmarks()
    const timer = window.setInterval(refreshData, 2500)
    return () => clearInterval(timer)
  }, [])

  useEffect(() => {
    if (!stream) return
    const timer = window.setInterval(() => {
      const image = document.getElementById('feed') as HTMLImageElement | null
      if (image) image.src = `${API}/streams/${stream}/frame?t=${Date.now()}`
    }, 200)
    return () => clearInterval(timer)
  }, [stream])

  const startCamera = async () => {
    setBusy(true)
    try {
      if (stream) await request(`/streams/${stream}/stop`, { method: 'POST' })
      const result = await request('/streams/webcam/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera_index: camera })
      })
      setStream(result.stream_id)
      refreshData()
    } catch {
      setStream('')
    } finally {
      setBusy(false)
    }
  }

  const chooseVideo = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return
    setBusy(true)
    try {
      if (stream) await request(`/streams/${stream}/stop`, { method: 'POST' })
      const data = new FormData()
      data.append('file', file)
      const result = await request('/streams/video', { method: 'POST', body: data })
      setStream(result.stream_id)
      refreshData()
    } catch {
      setStream('')
    } finally {
      setBusy(false)
      event.target.value = ''
    }
  }

  const stopStream = async () => {
    if (!stream) return
    await request(`/streams/${stream}/stop`, { method: 'POST' })
    setStream('')
    refreshData()
  }

  const handlePhotoUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return
    setDiagFile(file)
    setDiagPreview(URL.createObjectURL(file))
    setDiagBusy(true)
    
    try {
      const formData = new FormData()
      formData.append('file', file)
      const res = await request('/api/diagnose', {
        method: 'POST',
        body: formData
      }) as DiagnosisResponse
      setDiagResult(res)
      setActiveTab('inspector') // auto-switch
    } catch (err) {
      console.error('Diagnosis failed:', err)
    } finally {
      setDiagBusy(false)
    }
  }

  const online = health?.status === 'ok'
  const criticalCount = alerts.filter(a => a.severity === 'critical').length
  const activeModelKey = modelsStatus?.active_model ?? health?.active_model ?? 'yolo11n'

  return (
    <div className="app-container">
      {/* Top Header Bar */}
      <header className="top-nav">
        <div className="nav-brand">
          <div className="nav-logo">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
          </div>
          <h1>VisionSafe</h1>
        </div>
        
        <nav className="nav-links">
          <button className={`nav-link ${activeTab === 'dashboard' ? 'active' : ''}`} onClick={() => setActiveTab('dashboard')}>
            Dashboard
          </button>
          <button className={`nav-link ${activeTab === 'benchmark' ? 'active' : ''}`} onClick={() => { setActiveTab('benchmark'); fetchBenchmarks(); }}>
            Reports
          </button>
        </nav>
        
        <div className="nav-end">
          <div className={`status-pill ${online ? 'online' : 'offline'}`}>
            <span className="dot"></span>
            {online ? 'Online' : 'Offline'}
          </div>
        </div>
      </header>

      {/* Telemetry Ribbon */}
      <div className="telemetry-ribbon">
        <div className="tel-card">
          <div className="tel-icon bg-blue">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path strokeLinecap="round" strokeLinejoin="round" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
            </svg>
          </div>
          <div className="tel-data">
            <strong>{activeModelKey ? `YOLO-${activeModelKey.toUpperCase()}` : 'Ready'}</strong>
            <span>Active AI Model</span>
          </div>
        </div>
        
        <div className="tel-card">
          <div className="tel-icon bg-amber">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
          <div className="tel-data">
            <strong>{telemetry?.violation_count ?? criticalCount}</strong>
            <span>Critical Alerts</span>
          </div>
        </div>

        <div className="tel-card">
          <div className="tel-icon bg-green">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path strokeLinecap="round" strokeLinejoin="round" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
            </svg>
          </div>
          <div className="tel-data">
            <strong>{telemetry?.worker_count ?? 0}</strong>
            <span>Workers Tracked</span>
          </div>
        </div>

        <div className="tel-card">
          <div className="tel-icon bg-purple">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <div className="tel-data">
            <strong>{telemetry?.fps ? `${telemetry.fps.toFixed(0)} FPS` : '--'}</strong>
            <span>System Latency</span>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="main-layout">
        
        {/* DASHBOARD VIEW */}
        {activeTab === 'dashboard' && (
          <>
            {/* Left Sidebar: Controls */}
            <aside className="sidebar-left">
              <div className="input-tabs">
                <button 
                  className={`input-tab ${inputMode === 'camera' ? 'active' : ''}`}
                  onClick={() => setInputMode('camera')}
                >
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
                  CAMERA
                </button>
                <button 
                  className={`input-tab ${inputMode === 'video' ? 'active' : ''}`}
                  onClick={() => setInputMode('video')}
                >
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M7 4v16M17 4v16M3 8h4m10 0h4M3 12h18M3 16h4m10 0h4M4 20h16a1 1 0 001-1V5a1 1 0 00-1-1H4a1 1 0 00-1 1v14a1 1 0 001 1z" /></svg>
                  VIDEO
                </button>
              </div>

              <div className="input-controls">
                {inputMode === 'camera' ? (
                  <>
                    <select value={camera} onChange={e => setCamera(Number(e.target.value))} disabled={!online || busy}>
                      {cameras.length ? cameras.map(c => <option key={c.id} value={c.id}>{c.name}</option>) : <option value={0}>Camera 0</option>}
                    </select>
                    <div className="btn-group">
                      <button className="btn-secondary" onClick={scanCameras} disabled={!online || busy}>Scan</button>
                      <button className="btn-primary" onClick={startCamera} disabled={!online || busy}>
                        {busy ? 'Starting...' : '▶ Start'}
                      </button>
                    </div>
                    <small className="help-text">✓ Found {cameras.length} camera(s).</small>
                  </>
                ) : (
                  <>
                    <button className="btn-primary w-full" onClick={() => uploadStreamRef.current?.click()} disabled={!online || busy}>
                      📁 Select Video File
                    </button>
                    <input ref={uploadStreamRef} type="file" accept="video/*" onChange={chooseVideo} hidden />
                    <small className="help-text">Supports .mp4, .avi, .mkv</small>
                  </>
                )}
              </div>

              <div className="history-section">
                <h3 className="section-title">DETECTION HISTORY</h3>
                {alerts.length ? (
                  <div className="history-list">
                    {alerts.slice(0, 5).map(item => (
                      <div key={item.id} className="history-item">
                        <div className="hi-header">
                          <span className={`tag ${item.severity}`}>{item.severity}</span>
                          <span className="time">{new Date(item.created_at).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
                        </div>
                        <p>Worker #{item.worker_id}: {item.missing_ppe.join(', ')}</p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="empty-state-small">
                    <div className="es-icon">🌱</div>
                    <p>No alerts yet.<br/>Start a feed to monitor.</p>
                  </div>
                )}
              </div>
            </aside>

            {/* Center: Video Feed */}
            <section className="main-feed-area">
              <div className="feed-header">
                <div className={`stream-status ${stream ? 'active' : 'idle'}`}>
                  <span className="dot"></span>
                  {stream ? 'LIVE STREAM' : 'IDLE'}
                </div>
                {stream && <button className="btn-danger-sm" onClick={stopStream}>Stop</button>}
              </div>

              <div className="video-container">
                {stream ? (
                  <img id="feed" src={`${API}/streams/${stream}/frame`} alt="Live PPE Stream" />
                ) : (
                  <div className="empty-video">
                    <div className="ev-icon">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path strokeLinecap="round" strokeLinejoin="round" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
                    </div>
                    <h2>No active feed.</h2>
                    <p>Start a camera or upload media to begin.</p>
                    <div className="ev-actions">
                      <button onClick={() => setInputMode('camera')} className="btn-outline">📹 Camera</button>
                      <button onClick={() => setInputMode('video')} className="btn-outline">🎞️ Video</button>
                    </div>
                  </div>
                )}
              </div>
            </section>

            {/* Right Sidebar: AI Analysis */}
            <aside className="sidebar-right">
              <div className="right-tabs">
                <div className="rt-tab active">🤖 AI Analysis</div>
                <div className="rt-tab">⚙️ Models</div>
              </div>

              <div className="analysis-content">
                {!stream ? (
                  <div className="empty-state-small">
                    <div className="es-icon">🔬</div>
                    <h4>No Analysis Yet</h4>
                    <p>Start a camera feed or upload a video to get AI-powered PPE compliance diagnosis.</p>
                  </div>
                ) : (
                  <div className="live-analysis">
                    <div className="la-card">
                      <span className="la-label">Compliance Score</span>
                      <div className="la-value text-blue">{telemetry?.compliance_rate.toFixed(0) ?? 100}%</div>
                    </div>
                    
                    <h4 className="mt-4 mb-2 text-sm font-semibold text-slate-500 uppercase">Available Models</h4>
                    <div className="model-list">
                      {modelsStatus?.models.map(m => (
                        <button 
                          key={m.id} 
                          className={`model-select-btn ${activeModelKey === m.id ? 'active' : ''}`}
                          onClick={() => switchModel(m.id)}
                          disabled={switchingModel}
                        >
                          <div className="ms-name">{m.name}</div>
                          <div className="ms-desc">{m.fps_estimate} FPS</div>
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </aside>
          </>
        )}

        {/* BENCHMARK / REPORTS VIEW */}
        {activeTab === 'benchmark' && (
          <div className="benchmark-view">
             <div className="panel-white">
                <h2>Model Benchmarks & Empirical Validation</h2>
                <p className="text-slate-500 mb-6">Comparison of YOLO architectures on the PPE validation dataset.</p>
                
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Architecture</th>
                      <th>Precision</th>
                      <th>Recall</th>
                      <th>Macro F1</th>
                      <th>mAP@50</th>
                      <th>mAP@50-95</th>
                      <th>Type</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="bg-slate-50">
                      <td><b>Base Paper Benchmark</b></td>
                      <td>94.77%</td>
                      <td>—</td>
                      <td>94.65%</td>
                      <td>N/A</td>
                      <td>N/A</td>
                      <td><span className="badge-gray">Reference</span></td>
                    </tr>
                    {benchmarks && Object.entries(benchmarks).map(([key, data]) => {
                      const isBest = key === 'ensemble'
                      return (
                        <tr key={key} className={isBest ? 'bg-blue-50' : ''}>
                          <td>
                            <b>YOLO-{key.toUpperCase()}</b>
                            {isBest && <span className="text-xs text-blue-600 block">★ BEST PERFORMANCE</span>}
                          </td>
                          <td>{(data.precision * 100).toFixed(2)}%</td>
                          <td>{(data.recall * 100).toFixed(2)}%</td>
                          <td><b>{(data.f1 * 100).toFixed(2)}%</b></td>
                          <td>{(data.mAP50 * 100).toFixed(2)}%</td>
                          <td>{((data['mAP50-95'] ?? data.mAP50_95 ?? 0) * 100).toFixed(2)}%</td>
                          <td>
                            <span className="badge-blue">{isBest ? 'Ensemble' : 'Standard'}</span>
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
             </div>
          </div>
        )}

      </div>
    </div>
  )
}

createRoot(document.getElementById('root')!).render(<App />)
