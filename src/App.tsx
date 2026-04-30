import { useState, useCallback, useRef, useEffect } from 'react'
import ModelViewer from './ModelViewer'
import { extractLandmarks } from './pose'
import { calculateDrag } from './drag'
import { getRecommendations } from './gemma'
import { createImageTo3DTask, pollTask } from './meshy'
import { saveSubject, getLeaderboard, uploadGlb, type SubjectRecord } from './db'
import { listSucceededTasks, type MeshyListItem } from './meshy'
import type { AnalysisState, GlbMeasurements, BodyMeasurements, DragResults } from './types'

const DEMO_MODE = true

const CLOUD_NAME    = import.meta.env.VITE_CLOUDINARY_CLOUD_NAME as string
const UPLOAD_PRESET = import.meta.env.VITE_CLOUDINARY_UPLOAD_PRESET as string

declare global {
  interface Window {
    cloudinary: {
      openUploadWidget: (
        options: Record<string, unknown>,
        callback: (
          error: unknown,
          result: { event: string; info: { secure_url: string; public_id: string } }
        ) => void
      ) => void
    }
  }
}

function openWidget(cb: (url: string) => void) {
  window.cloudinary.openUploadWidget(
    {
      cloudName: CLOUD_NAME,
      uploadPreset: UPLOAD_PRESET,
      sources: ['local', 'camera'],
      multiple: false,
      cropping: false,
      showPoweredBy: false,
      styles: {
        palette: {
          window: '#0f0f0f', windowBorder: '#2a2a2a',
          tabIcon: 'var(--green)', menuIcons: '#666',
          textDark: '#e8e8e8', textLight: '#080808',
          link: 'var(--green)', action: 'var(--green)',
          inactiveTabIcon: '#444', error: '#ef5350',
          inProgress: '#4fc3f7', complete: 'var(--green)',
          sourceBg: '#1a1a1a',
        },
      },
    },
    (err, result) => {
      if (!err && result.event === 'success') cb(result.info.secure_url)
    }
  )
}

async function toBase64(url: string): Promise<string> {
  const res = await fetch(url)
  const blob = await res.blob()
  return new Promise(resolve => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as string)
    reader.readAsDataURL(blob)
  })
}

async function uploadGlbToSupabase(glbUrl: string): Promise<string> {
  const proxyUrl = glbUrl.replace('https://assets.meshy.ai', '/meshy-assets')
  const blob = await fetch(proxyUrl).then(r => {
    if (!r.ok) throw new Error(`GLB fetch failed: ${r.status}`)
    return r.blob()
  })
  const filename = `${Date.now()}-${Math.random().toString(36).slice(2)}.glb`
  return uploadGlb(blob, filename)
}

const INIT: AnalysisState = {
  landmarks: null, measurements: null, glbMeasurements: null, drag: null,
  recommendations: null, visualRecommendations: null, annotatedImageUrl: null,
  loading: false, loadingMessage: '', error: null,
}

// ─── Types ───────────────────────────────────────────────────────────────────

// ─── Sub-components ───────────────────────────────────────────────────────��───

function SectionLabel({ n, children }: { n: string; children: React.ReactNode }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 10,
      fontFamily: 'var(--mono)', fontSize: 10, letterSpacing: 1.5,
      color: 'var(--green)',
      marginBottom: 14, paddingBottom: 10, borderBottom: '1px solid #1a1a1a',
    }}>
      <span style={{
        display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
        width: 20, height: 20, borderRadius: '50%',
        border: '1px solid var(--green)', fontSize: 9, flexShrink: 0,
      }}>{n}</span>
      {children}
    </div>
  )
}

function renderTooltip(text: string): React.ReactNode {
  return text.split('\n').map((line, i) => {
    if (line === '') return <div key={i} style={{ height: '0.55em' }} />
    const parts = line.split(/(\[m\][^[]*\[\/m\]|\[c\][^[]*\[\/c\])/)
    return (
      <div key={i}>
        {parts.map((p, j) => {
          if (p.startsWith('[m]') && p.endsWith('[/m]'))
            return <span key={j} style={{ color: 'var(--green)' }}>{p.slice(3, -4)}</span>
          if (p.startsWith('[c]') && p.endsWith('[/c]'))
            return <span key={j} style={{ color: '#c87533' }}>{p.slice(3, -4)}</span>
          return <span key={j}>{p}</span>
        })}
      </div>
    )
  })
}

function DataRow({ label, value, accent, dim, tooltip }: {
  label: string; value: string; accent?: boolean; dim?: boolean; tooltip?: string
}) {
  const [open, setOpen] = useState(false)
  return (
    <div>
      <div style={{
        display: 'flex', justifyContent: 'space-between', gap: 8,
        fontFamily: 'var(--mono)', fontSize: 12, lineHeight: 2.1,
      }}>
        <span style={{ color: dim ? '#777' : '#aaa', display: 'flex', alignItems: 'center', gap: 5 }}>
          {label}
          {tooltip && (
            <button
              onClick={() => setOpen(o => !o)}
              style={{
                background: 'none', border: 'none', cursor: 'pointer', padding: 0,
                fontFamily: 'var(--mono)', fontSize: 12, lineHeight: 1,
                color: open ? 'var(--green)' : '#999',
                transition: 'color 0.1s',
              }}
            >↴</button>
          )}
        </span>
        <span style={{ color: accent ? '#ffffff' : dim ? '#888' : 'var(--green)', fontWeight: accent ? 700 : 400, textAlign: 'right' }}>
          {value}
        </span>
      </div>
      {open && tooltip && (
        <div style={{
          background: '#0d0d0d', borderLeft: '2px solid #252525',
          padding: '9px 12px', marginBottom: 2,
          fontFamily: 'var(--mono)', fontSize: 10, color: '#777',
          lineHeight: 1.9,
        }}>
          {renderTooltip(tooltip)}
        </div>
      )}
    </div>
  )
}


function Leaderboard({ onClose, onLoadRecord, onImport, demo = false }: {
  onClose: () => void
  onLoadRecord: (r: SubjectRecord) => void
  onImport: (imageUrl: string, glbUrl: string) => void
  demo?: boolean
}) {
  const [records, setRecords] = useState<SubjectRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [page, setPage] = useState(1)
  const [showImport, setShowImport] = useState(false)
  const PAGE_SIZE = 10
  const [importImage, setImportImage] = useState('')
  const [importGlb, setImportGlb] = useState('')
  const [meshyTasks, setMeshyTasks] = useState<MeshyListItem[]>([])
  const [meshyLoading, setMeshyLoading] = useState(false)
  const [meshyFetched, setMeshyFetched] = useState(false)

  useEffect(() => {
    getLeaderboard().then(data => { setRecords(data); setLoading(false) })
  }, [])

  function handleImportSubmit() {
    const img = importImage.trim()
    const glb = importGlb.trim()
    if (!img || !glb) return
    onImport(img, glb)
    onClose()
  }

  function loadMeshyHistory() {
    if (meshyFetched) return
    setMeshyLoading(true)
    listSucceededTasks().then(tasks => {
      setMeshyTasks(tasks)
      setMeshyLoading(false)
      setMeshyFetched(true)
    })
  }

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 100,
        background: 'rgba(0,0,0,0.82)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}
    >
      <div
        onClick={e => e.stopPropagation()}
        style={{
          background: '#0f0f0f', border: '1px solid #2a2a2a',
          width: 'min(660px, 94vw)', maxHeight: '80vh',
          display: 'flex', flexDirection: 'column', overflow: 'hidden',
        }}
      >
        {/* Header */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '16px 20px', borderBottom: '1px solid #1a1a1a', flexShrink: 0,
        }}>
          <span style={{ fontFamily: 'var(--mono)', fontSize: 11, letterSpacing: 1.5, color: 'var(--green)' }}>
            Global leaderboard — lowest Cd wins
          </span>
          <button onClick={onClose} style={{
            background: 'none', border: 'none', color: '#666', cursor: 'pointer',
            fontFamily: 'var(--mono)', fontSize: 18, lineHeight: 1, padding: '0 4px',
          }}>×</button>
        </div>

        {/* Import bar */}
        {!demo && <div style={{ borderBottom: '1px solid #1a1a1a', padding: '10px 20px', flexShrink: 0 }}>
          {!showImport ? (
            <button
              onClick={() => setShowImport(true)}
              style={{
                background: 'none', border: 'none', cursor: 'pointer',
                fontFamily: 'var(--mono)', fontSize: 9, letterSpacing: 1,
                color: '#555', padding: 0,
              }}
            >
              + import existing Cloudinary entry
            </button>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {/* Meshy history browser */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ fontFamily: 'var(--mono)', fontSize: 9, color: '#666', letterSpacing: 0.5 }}>
                  Pick a GLB from your Meshy history, then add the matching photo URL
                </div>
                {!meshyFetched && (
                  <button onClick={loadMeshyHistory} style={{
                    background: 'none', border: 'none', cursor: 'pointer',
                    fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--green)', padding: 0,
                  }}>
                    {meshyLoading ? 'Loading...' : 'Load history'}
                  </button>
                )}
              </div>

              {meshyFetched && meshyTasks.length === 0 && (
                <div style={{ fontFamily: 'var(--mono)', fontSize: 9, color: '#444' }}>
                  No succeeded tasks found in your Meshy account.
                </div>
              )}

              {meshyTasks.length > 0 && (
                <div style={{
                  maxHeight: 140, overflowY: 'auto',
                  border: '1px solid #1a1a1a', background: '#0a0a0a',
                }}>
                  {meshyTasks.map(t => (
                    <div
                      key={t.id}
                      onClick={() => setImportGlb(t.glbUrl)}
                      style={{
                        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                        padding: '6px 10px', borderBottom: '1px solid #141414',
                        cursor: 'pointer', background: importGlb === t.glbUrl ? '#1a1a1a' : 'transparent',
                      }}
                      onMouseEnter={e => { if (importGlb !== t.glbUrl) e.currentTarget.style.background = '#111' }}
                      onMouseLeave={e => { if (importGlb !== t.glbUrl) e.currentTarget.style.background = 'transparent' }}
                    >
                      <span style={{ fontFamily: 'var(--mono)', fontSize: 9, color: '#888' }}>
                        {t.createdAt ? new Date(t.createdAt).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }) : t.id.slice(0, 12) + '…'}
                      </span>
                      <span style={{ fontFamily: 'var(--mono)', fontSize: 9, color: importGlb === t.glbUrl ? 'var(--green)' : '#555' }}>
                        {importGlb === t.glbUrl ? '✓ selected' : 'select GLB'}
                      </span>
                    </div>
                  ))}
                </div>
              )}

              <input
                placeholder="Photo URL from Cloudinary  (https://res.cloudinary.com/…)"
                value={importImage}
                onChange={e => setImportImage(e.target.value)}
                style={{
                  background: '#0a0a0a', border: '1px solid #2a2a2a', color: '#ccc',
                  fontFamily: 'var(--mono)', fontSize: 10, padding: '6px 10px', outline: 'none',
                }}
              />
              <input
                placeholder="GLB URL — auto-filled when you select above, or paste manually"
                value={importGlb}
                onChange={e => setImportGlb(e.target.value)}
                style={{
                  background: '#0a0a0a', border: '1px solid #2a2a2a', color: '#ccc',
                  fontFamily: 'var(--mono)', fontSize: 10, padding: '6px 10px', outline: 'none',
                }}
              />
              <div style={{ display: 'flex', gap: 8 }}>
                <button
                  onClick={handleImportSubmit}
                  disabled={!importImage.trim() || !importGlb.trim()}
                  className="btn-primary"
                  style={{ padding: '6px 16px', fontSize: 9, letterSpacing: 1.5 }}
                >
                  Import + analyze
                </button>
                <button
                  onClick={() => { setShowImport(false); setImportImage(''); setImportGlb('') }}
                  className="btn-secondary"
                  style={{ padding: '6px 12px', fontSize: 9 }}
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>}

        {/* Body */}
        <div style={{ overflowY: 'auto', flex: 1 }}>
          {loading && (
            <div style={{
              padding: 40, textAlign: 'center',
              fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--green)',
            }} className="pulse">
              Loading...
            </div>
          )}

          {!loading && records.length === 0 && (
            <div style={{
              padding: 40, textAlign: 'center',
              fontFamily: 'var(--mono)', fontSize: 11, color: '#555',
            }}>
              No entries yet. Be the first.
            </div>
          )}

          {records.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE).map((r, i) => {
            const rank = (page - 1) * PAGE_SIZE + i
            return (
              <div
                key={r.id}
                onClick={() => { onLoadRecord(r); onClose() }}
                style={{
                  display: 'flex', alignItems: 'center', gap: 14,
                  padding: '10px 20px', borderBottom: '1px solid #141414',
                  cursor: 'pointer', transition: 'background 0.12s',
                }}
                onMouseEnter={e => (e.currentTarget.style.background = '#161616')}
                onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
              >
                <span style={{
                  fontFamily: 'var(--mono)', fontSize: 11,
                  color: rank === 0 ? 'var(--green)' : '#555',
                  width: 24, flexShrink: 0, textAlign: 'right',
                }}>
                  {rank + 1}
                </span>
                <img
                  src={r.image_url}
                  style={{ width: 44, height: 56, objectFit: 'cover', flexShrink: 0, border: '1px solid #1a1a1a' }}
                />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontFamily: 'var(--mono)', fontSize: 13, color: 'var(--green)' }}>
                    Cd {r.cd_score.toFixed(4)}
                  </div>
                  <div style={{ fontFamily: 'var(--mono)', fontSize: 9, color: '#999', marginTop: 2 }}>
                    {new Date(r.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                    {' · click to load 3D model'}
                  </div>
                </div>
                {rank === 0 && (
                  <span style={{ fontFamily: 'var(--mono)', fontSize: 8, color: 'var(--green)', letterSpacing: 1 }}>
                    most aerodynamic
                  </span>
                )}
              </div>
            )
          })}
        </div>

        {/* Pagination */}
        {records.length > PAGE_SIZE && (
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            padding: '10px 20px', borderTop: '1px solid #1a1a1a', flexShrink: 0,
          }}>
            <button
              onClick={() => setPage(p => Math.max(1, p - 1))}
              disabled={page === 1}
              style={{
                background: 'none', border: 'none', cursor: page === 1 ? 'default' : 'pointer',
                fontFamily: 'var(--mono)', fontSize: 10, color: page === 1 ? '#333' : '#777', padding: 0,
              }}
            >← prev</button>
            <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: '#555' }}>
              {page} / {Math.ceil(records.length / PAGE_SIZE)}
            </span>
            <button
              onClick={() => setPage(p => Math.min(Math.ceil(records.length / PAGE_SIZE), p + 1))}
              disabled={page === Math.ceil(records.length / PAGE_SIZE)}
              style={{
                background: 'none', border: 'none',
                cursor: page === Math.ceil(records.length / PAGE_SIZE) ? 'default' : 'pointer',
                fontFamily: 'var(--mono)', fontSize: 10,
                color: page === Math.ceil(records.length / PAGE_SIZE) ? '#333' : '#777', padding: 0,
              }}
            >next →</button>
          </div>
        )}
      </div>
    </div>
  )
}

function GemmaThinking() {
  const CHARS = ['+', '-', "'", '`']
  const [t, setT] = useState(0)
  const [elapsed, setElapsed] = useState(0)
  const startRef = useRef(Date.now())
  useEffect(() => {
    const id = setInterval(() => {
      setT(n => (n + 1) % 4)
      setElapsed(Date.now() - startRef.current)
    }, 90)
    return () => clearInterval(id)
  }, [])
  const str = Array.from({ length: 9 }, (_, i) => CHARS[(t + i) % 4]).join(' ')
  return (
    <div style={{ fontFamily: 'var(--mono)', fontSize: 13, color: 'var(--green)', letterSpacing: 2, display: 'flex', alignItems: 'center', gap: 14 }}>
      <span style={{ color: '#555', fontSize: 11, letterSpacing: 0.5, flexShrink: 0 }}>
        [{(elapsed / 1000).toFixed(1)}s]
      </span>
      {str}
    </div>
  )
}

function DemoBanner() {
  return (
    <div style={{
      background: '#1a0010', borderBottom: '1px solid #ff69b4',
      padding: '8px 20px', textAlign: 'center',
      fontFamily: 'var(--mono)', fontSize: 10, letterSpacing: 1,
      color: '#ff69b4',
    }}>
      DEMO MODE — leaderboard is live, uploads and AI are disabled. Sorry, API credits are expensive :/
    </div>
  )
}

function UploadPrompt({ onUpload }: { onUpload: () => void }) {
  const [hover, setHover] = useState(false)
  return (
    <button
      onClick={onUpload}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        display: 'block', width: '100%', textAlign: 'left',
        border: `1px dashed ${hover ? 'var(--green)' : '#444'}`,
        background: 'transparent', padding: '24px 20px',
        cursor: 'pointer', transition: 'border-color 0.15s',
      }}
    >
      <div style={{
        fontFamily: 'var(--mono)', fontSize: 11, letterSpacing: 1,
        color: hover ? 'var(--green)' : '#cfcfcfff', marginBottom: 10,
      }}>
        UPLOAD SUBJECT PHOTOGRAPH
      </div>
      <div style={{ fontFamily: 'var(--sans)', fontSize: 12, color: '#c5c5c5ff', lineHeight: 1.65 }}>
        Front-facing, full body, arms slightly away from torso.<br />
        Plain background preferred. Good lighting required.
      </div>
    </button>
  )
}

// ─── App ─────────────────────────────────────────────────────────────────────

export default function App() {
  const [frontUrl, setFrontUrl]           = useState<string | null>(null)
  const [meshyModelUrl, setMeshyModelUrl] = useState<string | null>(null)
  const [meshyLoading, setMeshyLoading]   = useState(false)
  const [meshyMessage, setMeshyMessage]   = useState('')
  const [analysis, setAnalysis]           = useState<AnalysisState>(INIT)
  const [showLeaderboard, setShowLeaderboard] = useState(false)
  const [gemmaOnDemand, setGemmaOnDemand]     = useState(false)
  const [gemmaThinking, setGemmaThinking]     = useState(false)
  const [demoGemmaShown, setDemoGemmaShown]   = useState(false)

  // Pipeline state refs — stable across renders, safe to read in async callbacks
  const poseDataRef          = useRef<{ measurements: BodyMeasurements; drag: DragResults } | null>(null)
  const gemmaOnDemandRef     = useRef(false)
  const gemmaResultRef       = useRef<string | null>(null)
  const userModelRenderedRef = useRef(false)
  const gemmaRunningRef      = useRef(false)
  // URL refs so async callbacks always see the latest values without stale closures
  const frontUrlRef          = useRef<string | null>(null)
  const meshyUrlRef          = useRef<string | null>(null)
  const skipNextSaveRef      = useRef(false)   // prevents duplicate DB insert on leaderboard reload

  // ── Reveal gate: shows Gemma result only after the user's 3D model has rendered ──

  const tryRevealGemma = useCallback(() => {
    if (gemmaResultRef.current !== null && userModelRenderedRef.current) {
      setAnalysis(s => ({ ...s, recommendations: gemmaResultRef.current! }))
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Pipelines ──────────────────────────────────────────────────────────────

  async function runPoseAnalysis(url: string) {
    setAnalysis(s => ({ ...s, loading: true, loadingMessage: 'Loading photographic data...', error: null }))
    try {
      const canvas = document.createElement('canvas')
      const img = new Image()
      img.crossOrigin = 'anonymous'
      await new Promise<void>((res, rej) => {
        img.onload = () => {
          canvas.width = img.naturalWidth
          canvas.height = img.naturalHeight
          canvas.getContext('2d')!.drawImage(img, 0, 0)
          res()
        }
        img.onerror = () => rej(new Error('Image load failed'))
        img.src = url
      })

      setAnalysis(s => ({ ...s, loadingMessage: 'Extracting biomechanical landmarks...' }))
      const { landmarks, measurements, annotatedCanvas } = await extractLandmarks(canvas)
      const annotatedImageUrl = annotatedCanvas.toDataURL('image/jpeg', 0.88)

      setAnalysis(s => ({ ...s, loadingMessage: 'Computing aerodynamic coefficients...' }))
      const drag = calculateDrag(measurements, null)

      poseDataRef.current = { measurements, drag }
      setAnalysis(s => ({
        ...s, landmarks, measurements, drag, annotatedImageUrl,
        loading: false, loadingMessage: '',
      }))

      if (!gemmaOnDemandRef.current) void runGemma(measurements, drag)
    } catch (err) {
      setAnalysis(s => ({
        ...s, loading: false, loadingMessage: '',
        error: err instanceof Error ? err.message : 'Analysis failed',
      }))
    }
  }

  async function runGemma(measurements: BodyMeasurements, drag: DragResults) {
    if (gemmaRunningRef.current) return
    gemmaRunningRef.current = true
    try {
      const recommendations = await getRecommendations(measurements, drag, null)
      gemmaResultRef.current = recommendations
      tryRevealGemma()
    } catch (err) {
      setAnalysis(s => ({
        ...s, error: err instanceof Error ? err.message : 'Gemma analysis failed',
      }))
    } finally {
      gemmaRunningRef.current = false
      setGemmaThinking(false)
    }
  }

  async function runMeshy(url: string) {
    if (DEMO_MODE) return
    const key = import.meta.env.VITE_MESHY_API_KEY
    if (!key || key.startsWith('your_')) return
    setMeshyLoading(true)
    setMeshyMessage('Generating 3D model...')
    try {
      const base64 = await toBase64(url)
      const taskId = await createImageTo3DTask(base64)
      const glbUrl = await pollTask(taskId, pct => {
        setMeshyMessage(`Generating 3D model... ${pct}%`)
      })

      setMeshyMessage('Uploading to storage...')
      let finalUrl = glbUrl
      try {
        finalUrl = await uploadGlbToSupabase(glbUrl)
      } catch (err) {
        console.warn('[AeroMaxx] Supabase GLB upload failed, using proxy URL:', err)
        finalUrl = glbUrl.replace('https://assets.meshy.ai', '/meshy-assets')
      }

      meshyUrlRef.current = finalUrl
      setMeshyModelUrl(finalUrl)
    } catch (err) {
      console.error('[AeroMaxx] Meshy failed:', err)
    } finally {
      setMeshyLoading(false)
      setMeshyMessage('')
    }
  }

  // Fires when ModelViewer loads and measures the GLB geometry.
  // isUserModel=true only when the user's Meshy GLB has rendered (not the default fallback).
  const handleGeometryMeasured = useCallback((glb: GlbMeasurements, isUserModel: boolean) => {
    const pose = poseDataRef.current
    const drag = pose ? calculateDrag(pose.measurements, glb) : null
    setAnalysis(s => ({ ...s, glbMeasurements: glb, ...(drag ? { drag } : {}) }))
    if (isUserModel) {
      setMeshyLoading(false)
      setMeshyMessage('')
      userModelRenderedRef.current = true
      tryRevealGemma()
      if (!skipNextSaveRef.current && drag && frontUrlRef.current && meshyUrlRef.current) {
        void saveSubject(frontUrlRef.current, meshyUrlRef.current, drag.Cd)
      }
      skipNextSaveRef.current = false
    }
  }, [tryRevealGemma]) // eslint-disable-line react-hooks/exhaustive-deps

  const handleManualImport = useCallback((imageUrl: string, glbUrl: string) => {
    frontUrlRef.current          = imageUrl
    meshyUrlRef.current          = glbUrl
    gemmaOnDemandRef.current     = true
    setGemmaOnDemand(true)
    // skipNextSaveRef stays false — this is a new entry, we want to persist it
    setFrontUrl(imageUrl)
    setMeshyModelUrl(glbUrl)
    setMeshyLoading(true)
    setMeshyMessage('Loading 3D model from database...')
    setAnalysis(INIT)
    poseDataRef.current          = null
    gemmaResultRef.current       = null
    userModelRenderedRef.current = false
    gemmaRunningRef.current      = false
    void runPoseAnalysis(imageUrl)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const handleLeaderboardLoad = useCallback((record: SubjectRecord) => {
    frontUrlRef.current          = record.image_url
    meshyUrlRef.current          = record.glb_url
    skipNextSaveRef.current      = true   // already in DB — don't re-insert
    gemmaOnDemandRef.current     = true
    setGemmaOnDemand(true)
    setDemoGemmaShown(false)
    setFrontUrl(record.image_url)
    setMeshyModelUrl(record.glb_url)
    setMeshyLoading(true)
    setMeshyMessage('Loading 3D model from database...')
    setAnalysis(INIT)
    poseDataRef.current          = null
    gemmaResultRef.current       = null
    userModelRenderedRef.current = false
    gemmaRunningRef.current      = false
    void runPoseAnalysis(record.image_url)
    // Meshy is skipped — GLB already exists; handleGeometryMeasured will fire
    // when the viewer loads it and trigger tryRevealGemma once Gemma finishes
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const handleUpload = useCallback((url: string) => {
    frontUrlRef.current          = url
    meshyUrlRef.current          = null
    gemmaOnDemandRef.current     = false
    setGemmaOnDemand(false)
    setFrontUrl(url)
    setMeshyModelUrl(null)
    setAnalysis(INIT)
    poseDataRef.current          = null
    gemmaResultRef.current       = null
    userModelRenderedRef.current = false
    gemmaRunningRef.current      = false
    setDemoGemmaShown(false)
    if (DEMO_MODE) {
      gemmaOnDemandRef.current = true
      setGemmaOnDemand(true)
      userModelRenderedRef.current = true
    } else {
      const meshyKey = import.meta.env.VITE_MESHY_API_KEY as string
      if (!meshyKey || meshyKey.startsWith('your_')) userModelRenderedRef.current = true
    }
    void runPoseAnalysis(url)
    void runMeshy(url)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const { measurements: m, drag: d, glbMeasurements: glb } = analysis

  return (
    <div className="app-shell">

      {DEMO_MODE && <DemoBanner />}

      {/* ── Header ── */}

    <header className="app-header" style={{ position: 'relative' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 16 }}>
        <span style={{
          fontFamily: 'var(--mono)', fontSize: 17, fontWeight: 600,
          color: 'var(--green)', letterSpacing: 2,
        }}>
          AEROMAXX
        </span>
      </div>
      <span style={{
        fontFamily: 'var(--mono)', fontSize: 13, letterSpacing: 1,
        color: '#cacacaff',
        position: 'absolute', left: '50%', transform: 'translateX(-50%)',
      }}>
        Looksmaxx your aerodynamics → streamline your life.
      </span>
      <button
        className="btn-secondary"
        onClick={() => setShowLeaderboard(true)}
        style={{ padding: '6px 14px', fontSize: 10, letterSpacing: 1.5 }}
      >
        Leaderboard
      </button>
    </header>

    {showLeaderboard && (
      <Leaderboard
        onClose={() => setShowLeaderboard(false)}
        onLoadRecord={handleLeaderboardLoad}
        onImport={handleManualImport}
        demo={DEMO_MODE}
      />
    )}


      {/* ── Two-column body ── */}
      <div className="app-body">

        {/* Left — 3D viewer */}
        <div className="viewer-col" style={{ position: 'relative' }}>
          <ModelViewer
            modelUrl={meshyModelUrl}
            cdValue={d?.Cd ?? 0.8}
            loading={meshyLoading}
            loadingMessage={meshyMessage}
            onGeometryMeasured={handleGeometryMeasured}
          />
        </div>

        {/* Right — upload + stats */}
        <div className="stats-col">

          {/* 01 Upload */}
          <div className="panel-section">
            <SectionLabel n="1">SUBJECT INPUT</SectionLabel>
            {!frontUrl ? (
              DEMO_MODE ? (
                <div style={{
                  border: '1px dashed #2a2a2a', padding: '24px 20px',
                  fontFamily: 'var(--mono)', fontSize: 11,
                }}>
                  <div style={{ color: '#ff69b4', letterSpacing: 1, marginBottom: 10 }}>
                    UPLOAD DISABLED IN DEMO MODE
                  </div>
                  <div style={{ color: '#999', fontSize: 10, lineHeight: 1.7 }}>
                    In the full version, you'd upload a photo here and get a
                    live 3D model + aerodynamic analysis.<br />
                    Click <span style={{ color: '#ff69b4' }}>LEADERBOARD</span> in the header to browse existing entries.
                  </div>
                </div>
              ) : (
                <UploadPrompt onUpload={() => openWidget(handleUpload)} />
              )
            ) : (
              <div>
                {/* Raw input + annotated pose scan side by side */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                  <div style={{ position: 'relative' }}>
                    <img
                      src={frontUrl}
                      alt="Subject"
                      style={{
                        width: '100%', maxHeight: 360, objectFit: 'contain',
                        objectPosition: 'top', display: 'block',
                        border: '1px solid #1a1a1a', background: '#0a0a0a',
                      }}
                    />
                    <div style={{
                      position: 'absolute', top: 6, left: 6,
                      fontFamily: 'var(--mono)', fontSize: 8, letterSpacing: 0.5,
                      color: 'var(--green)', background: 'rgba(8,8,8,0.85)', padding: '2px 5px',
                    }}>
                      Input
                    </div>
                  </div>

                  <div style={{ position: 'relative' }}>
                    {analysis.annotatedImageUrl ? (
                      <img
                        src={analysis.annotatedImageUrl}
                        alt="Pose scan"
                        style={{
                          width: '100%', maxHeight: 360, objectFit: 'contain',
                          objectPosition: 'top', display: 'block',
                          border: '1px solid #1a1a1a', background: '#0a0a0a',
                        }}
                      />
                    ) : (
                      <div style={{
                        width: '100%', height: 225,
                        border: '1px solid #1a1a1a', background: '#0a0a0a',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                      }}>
                        <span style={{
                          fontFamily: 'var(--mono)', fontSize: 10, letterSpacing: 1,
                          color: 'var(--green)',
                        }} className="pulse">
                          Scanning...
                        </span>
                      </div>
                    )}
                    <div style={{
                      position: 'absolute', top: 6, left: 6,
                      fontFamily: 'var(--mono)', fontSize: 8, letterSpacing: 0.5,
                      color: 'var(--green)', background: 'rgba(8,8,8,0.85)', padding: '2px 5px',
                    }}>
                      Pose Scan
                    </div>
                  </div>
                </div>

                {analysis.loading && (
                  <div style={{
                    fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--green)',
                    letterSpacing: 0.5, marginTop: 12,
                  }} className="pulse">
                    {analysis.loadingMessage}
                  </div>
                )}

                {analysis.error && (
                  <div style={{
                    fontFamily: 'var(--mono)', fontSize: 12, color: '#ef5350',
                    border: '1px solid #ef5350', padding: '10px 14px', marginTop: 12,
                    letterSpacing: 0,
                  }}>
                    ⚠ {analysis.error}
                  </div>
                )}

                {!DEMO_MODE && (
                  <button
                    className="btn-secondary"
                    onClick={() => openWidget(handleUpload)}
                    style={{ marginTop: 12, width: '100%' }}
                  >
                    New subject
                  </button>
                )}
              </div>
            )}
          </div>

          {/* 02 Analysis report — appears as soon as pose analysis completes */}
          {d && m && (
            <div className="panel-section">
              <SectionLabel n="2">AERODYNAMIC REPORT</SectionLabel>

              <DataRow
                label="Drag Coefficient (Cd)" value={d.Cd.toFixed(4)} accent
                tooltip={[
                  `Cd = [c]0.80[/c] base + [m]postural penalty[/m]${glb ? ` + [m]depth correction[/m]` : ''}`,
                  ``,
                  `base [c]0.80[/c]  — Hoerner (1965), Shanebrook & Jaszczak (1976)`,
                  `  published range for standing adult: 0.7 – 1.3`,
                  ``,
                  `hunch: [m]${m.hunchScore.toFixed(3)}[/m] × [c]0.08[/c] = +[m]${(m.hunchScore * 0.08).toFixed(4)}[/m] Cd`,
                  `  [c]0.08[/c] Cd/unit — conservative (published studies: 5–15%)`,
                  ...(m.shoulderToHipRatio > 1.3
                    ? [``, `s/h ratio [m]${m.shoulderToHipRatio.toFixed(3)}[/m] > [c]1.3[/c] → [c]+0.03[/c] Cd — lateral sway penalty`]
                    : m.shoulderToHipRatio < 0.9
                    ? [``, `s/h ratio [m]${m.shoulderToHipRatio.toFixed(3)}[/m] < [c]0.9[/c] → [c]+0.04[/c] Cd — lateral sway penalty`]
                    : []),
                  ...(glb
                    ? [``,
                       `depth/width [m]${glb.depthToWidthRatio.toFixed(3)}[/m]:`,
                       glb.depthToWidthRatio < 0.35
                         ? `  flat-body penalty: ([c]0.35[/c]−[m]${glb.depthToWidthRatio.toFixed(3)}[/m])×[c]0.40[/c] = +[m]${((0.35-glb.depthToWidthRatio)*0.40).toFixed(4)}[/m]`
                         : glb.depthToWidthRatio > 0.70
                         ? `  deep-body penalty: ([m]${glb.depthToWidthRatio.toFixed(3)}[/m]−[c]0.70[/c])×[c]0.20[/c] = +[m]${((glb.depthToWidthRatio-0.70)*0.20).toFixed(4)}[/m]`
                         : `  within optimal range [c]0.35–0.70[/c] → no correction`]
                    : [`  (no 3D model — depth correction not applied)`]),
                  ``,
                  `→ Cd = [m]${d.Cd.toFixed(4)}[/m]`,
                ].join('\n')}
              />
              <DataRow
                label="Frontal Area" value={`${d.frontalArea.toFixed(4)} m²`}
                tooltip={glb
                  ? [
                      `A = [m]realWidth[/m] × [c]height_ref[/c] × [c]fill_factor[/c]`,
                      `  = [m]${glb.realWidth.toFixed(4)} m[/m] × [c]1.75 m[/c] × [c]0.73[/c] = [m]${d.frontalArea.toFixed(4)} m²[/m]`,
                      ``,
                      `[c]1.75 m[/c] — normalized human height (5′9″ reference)`,
                      `[c]0.73[/c]   — fill factor: Kyle & Burke (1984), range 0.68–0.78`,
                    ].join('\n')
                  : [
                      `A = [m]shoulderWidth[/m] × [c]height_ref[/c] × [c]fill_factor[/c]`,
                      `  = [m]${m.realShoulderWidth.toFixed(4)} m[/m] × [c]1.75 m[/c] × [c]0.73[/c] = [m]${d.frontalArea.toFixed(4)} m²[/m]`,
                      ``,
                      `[c]1.75 m[/c] — normalized human height (5′9″ reference)`,
                      `[c]0.73[/c]   — fill factor: Kyle & Burke (1984), range 0.68–0.78`,
                    ].join('\n')}
              />
              <DataRow
                label="  └ area source"
                value={glb ? '3D model' : '2D estimate'}
                dim
              />

              {glb && (
                <>
                  <DataRow
                    label="Body Width (3D)" value={`${glb.realWidth.toFixed(3)} m`}
                    tooltip={[
                      `realWidth = ([m]bbox.size.x[/m] / [m]bbox.size.y[/m]) × [c]1.7526[/c]`,
                      `  [m]bbox[/m]: Three.js Box3.setFromObject() on the loaded GLB`,
                      `  size.x/size.y = width-to-height ratio of 3D bounding box`,
                      `  [c]1.7526 m[/c] = standardized height (5′9″)`,
                      ``,
                      `→ [m]${glb.realWidth.toFixed(4)} m[/m]`,
                    ].join('\n')}
                  />
                  <DataRow
                    label="Body Depth (3D)" value={`${glb.realDepth.toFixed(3)} m`}
                    tooltip={[
                      `realDepth = ([m]bbox.size.z[/m] / [m]bbox.size.y[/m]) × [c]1.7526[/c]`,
                      `  size.z = front-to-back extent of 3D bounding box`,
                      `  [c]1.7526 m[/c] = standardized height (5′9″)`,
                      ``,
                      `→ [m]${glb.realDepth.toFixed(4)} m[/m]`,
                    ].join('\n')}
                  />
                  <DataRow
                    label="Depth/Width Ratio" value={glb.depthToWidthRatio.toFixed(3)}
                    tooltip={[
                      `r = [m]realDepth[/m] / [m]realWidth[/m]`,
                      `  = [m]${glb.realDepth.toFixed(4)}[/m] / [m]${glb.realWidth.toFixed(4)}[/m] = [m]${glb.depthToWidthRatio.toFixed(4)}[/m]`,
                      ``,
                      `optimal range: [c]0.35 – 0.70[/c]`,
                      `  r < [c]0.35[/c] → flat-plate behavior, large separated wake`,
                      `  r > [c]0.70[/c] → excess depth increases pressure drag`,
                      `  r ≈ [c]0.50[/c] → minimum wake, best natural streamlining`,
                      `  penalty model from Hoerner (1965) bluff-body data`,
                    ].join('\n')}
                  />
                </>
              )}

              <DataRow
                label="Shoulder Width" value={`${m.realShoulderWidth.toFixed(4)} m`}
                tooltip={[
                  `1. [m]shoulderWidth_px[/m] = dist(landmark 11, 12) = [m]${m.shoulderWidth.toFixed(1)} px[/m]`,
                  `   MediaPipe PoseLandmarker — dist = √((Δx·W)²+(Δy·H)²)`,
                  ``,
                  `2. pixelsPerMeter = [m]bodyHeight_px[/m] / [c]1.7526 m[/c]`,
                  `   [c]1.7526 m[/c] = standardized reference height (5′9″)`,
                  `   [m]bodyHeight_px[/m] = dist(nose → ankle midpoint)`,
                  ``,
                  `3. realShoulderWidth = [m]${m.shoulderWidth.toFixed(1)} px[/m] / [m]${m.pixelsPerMeter.toFixed(1)} px·m⁻¹[/m]`,
                  `   = [m]${m.realShoulderWidth.toFixed(4)} m[/m]`,
                ].join('\n')}
              />
              <DataRow
                label="Hip Width" value={`${m.realHipWidth.toFixed(4)} m`}
                tooltip={[
                  `Same pipeline as shoulder width, landmarks 23 & 24 (hips)`,
                  ``,
                  `[m]hipWidth_px[/m] = dist(leftHip, rightHip) = [m]${m.hipWidth.toFixed(1)} px[/m]`,
                  `pixelsPerMeter = [m]${m.pixelsPerMeter.toFixed(1)} px·m⁻¹[/m]`,
                  `realHipWidth = [m]${m.hipWidth.toFixed(1)}[/m] / [m]${m.pixelsPerMeter.toFixed(1)}[/m] = [m]${m.realHipWidth.toFixed(4)} m[/m]`,
                ].join('\n')}
              />
              <DataRow
                label="Hunch Score (0–1)" value={m.hunchScore.toFixed(4)}
                tooltip={[
                  `hunchScore = |[m]shoulderMidX[/m] − [m]hipMidX[/m]| / ([m]shoulderWidth_px[/m] × [c]0.3[/c])`,
                  `  clamped to [0, 1]`,
                  ``,
                  `[m]shoulderMidX[/m]: midpoint of landmarks 11 & 12`,
                  `[m]hipMidX[/m]:      midpoint of landmarks 23 & 24`,
                  `[c]0.3[/c]: offset ≥ 30% of shoulder width → maximal hunch (score = 1)`,
                  `  captures kyphosis & forward head posture from lateral offset`,
                  ``,
                  `→ [m]${m.hunchScore.toFixed(4)}[/m]  (${m.hunchScore < 0.2 ? 'minimal' : m.hunchScore < 0.5 ? 'moderate' : 'significant'} postural asymmetry)`,
                ].join('\n')}
              />
              <DataRow
                label="Postural Cd Penalty" value={`+${d.posturalPenalty.toFixed(4)}`}
                tooltip={[
                  `penalty = hunch_component + asymmetry_component`,
                  ``,
                  `hunch: [m]${m.hunchScore.toFixed(4)}[/m] × [c]0.08[/c] = +[m]${(m.hunchScore * 0.08).toFixed(4)}[/m] Cd`,
                  `  [c]0.08[/c] Cd/unit — mid-range of published 5–15% kyphosis increase`,
                  ...(m.shoulderToHipRatio > 1.3
                    ? [``, `s/h [m]${m.shoulderToHipRatio.toFixed(3)}[/m] > [c]1.3[/c] → [c]+0.03[/c] Cd — broad-shoulder lateral sway`]
                    : m.shoulderToHipRatio < 0.9
                    ? [``, `s/h [m]${m.shoulderToHipRatio.toFixed(3)}[/m] < [c]0.9[/c] → [c]+0.04[/c] Cd — narrow-shoulder lateral sway`]
                    : [``, `s/h [m]${m.shoulderToHipRatio.toFixed(3)}[/m] within normal range [c]0.9–1.3[/c]`]),
                  ``,
                  `→ total = +[m]${d.posturalPenalty.toFixed(4)}[/m] Cd`,
                ].join('\n')}
              />
              <DataRow
                label="Drag Force @ Walk" value={`${d.dragForce.toFixed(4)} N`}
                tooltip={[
                  `F = ½ × [c]ρ[/c] × [c]v²[/c] × [m]Cd[/m] × [m]A[/m]`,
                  ``,
                  `[c]ρ = 1.225 kg·m⁻³[/c]  ISA standard atmosphere, sea level 15°C`,
                  `[c]v = 1.4 m·s⁻¹[/c]    ACSM avg adult walking speed (range 1.3–1.5)`,
                  `[m]Cd = ${d.Cd.toFixed(4)}[/m]`,
                  `[m]A  = ${d.frontalArea.toFixed(4)} m²[/m]`,
                  ``,
                  `= 0.5 × [c]1.225[/c] × [c]${(1.4*1.4).toFixed(2)}[/c] × [m]${d.Cd.toFixed(4)}[/m] × [m]${d.frontalArea.toFixed(4)}[/m]`,
                  `= [m]${d.dragForce.toFixed(4)} N[/m]`,
                ].join('\n')}
              />

              <div style={{ borderTop: '1px solid #1a1a1a', margin: '8px 0' }} />
              <DataRow
                label="Lifetime Energy Lost" value={`${d.lifetimeEnergy.toLocaleString(undefined, { maximumFractionDigits: 0 })} J`} accent
                tooltip={[
                  `E = [m]F[/m] × d   (work = force × distance)`,
                  ``,
                  `d = [c]v[/c] × [c]walk_hrs[/c] × [c]3600[/c] × [c]365[/c] × [c]lifespan[/c]`,
                  `  = [c]1.4[/c] × [c]4[/c] × [c]3600[/c] × [c]365[/c] × [c]75[/c] = [c]${(1.4*4*3600*365*75).toLocaleString()} m[/c]`,
                  ``,
                  `[c]4 hr·day⁻¹[/c]    NHS physical activity guideline estimate`,
                  `[c]75 yr[/c]         WHO 2023 global average lifespan`,
                  ``,
                  `E = [m]${d.dragForce.toFixed(4)} N[/m] × [c]${(1.4*4*3600*365*75).toLocaleString()} m[/c]`,
                  `= [m]${d.lifetimeEnergy.toLocaleString(undefined, { maximumFractionDigits: 0 })} J[/m]`,
                ].join('\n')}
              />
              <DataRow
                label="Energy Wasted" value={`${d.bigMacs.toFixed(1)} Big Macs`}
                tooltip={[
                  `bigMacs = [m]E_lifetime[/m] / [c]E_per_BigMac[/c]`,
                  ``,
                  `[c]E_per_BigMac = 550 kcal × 4,184 J·kcal⁻¹ = 2,301,200 J[/c]`,
                  `  [c]550 kcal[/c] = McDonald's official nutrition data`,
                  ``,
                  `= [m]${d.lifetimeEnergy.toLocaleString(undefined, { maximumFractionDigits: 0 })} J[/m] / [c]2,301,200[/c]`,
                  `= [m]${d.bigMacs.toFixed(2)} Big Macs[/m] over [c]75 years[/c]`,
                ].join('\n')}
              />
              <DataRow
                label="Metabolic cost of drag" value={`${d.daysLost.toFixed(1)} rest-days`}
                tooltip={[
                  `rest_days = [m]E_lifetime[/m] / ([c]P_rest[/c] × [c]86,400 s·day⁻¹[/c])`,
                  ``,
                  `[c]P_rest = 80 W[/c]  avg adult resting metabolic rate`,
                  `  Hall et al. (2012), NEJM — typical range 60–100 W`,
                  ``,
                  `= [m]${d.lifetimeEnergy.toLocaleString(undefined, { maximumFractionDigits: 0 })} J[/m] / ([c]80[/c] × [c]86,400[/c])`,
                  `= [m]${d.daysLost.toFixed(2)} days[/m]`,
                  ``,
                  `your resting metabolism (heart, lungs, thermoregulation)`,
                  `could run on drag's lifetime energy cost for this long.`,
                  `not literal time lost — a metabolic energy scale.`,
                ].join('\n')}
              />
            </div>
          )}

          {/* 03 Drag reduction protocol */}
          {(analysis.recommendations || demoGemmaShown || (d && m && (gemmaOnDemand || gemmaThinking))) && (
            <div className="panel-section">
              <SectionLabel n="3">DRAG REDUCTION PROTOCOL</SectionLabel>
              {demoGemmaShown ? (
                <div style={{
                  fontFamily: 'var(--mono)', fontSize: 11, color: '#ff69b4',
                  border: '1px solid #3a0020', background: '#0f0008',
                  padding: '12px 14px', lineHeight: 1.8,
                }}>
                  Gemma is disabled in demo mode. In the full version, she analyses
                  your posture data and generates a personalised drag reduction
                  protocol — specific exercises, clothing adjustments, and posture
                  corrections ranked by projected Cd improvement.
                </div>
              ) : analysis.recommendations ? (
                <div style={{
                  fontFamily: 'var(--sans)', fontSize: 13, color: '#ccc',
                  lineHeight: 1.8, whiteSpace: 'pre-wrap',
                }}>
                  {analysis.recommendations}
                </div>
              ) : gemmaThinking ? (
                <GemmaThinking />
              ) : (
                <button
                  className="btn-secondary"
                  style={{ width: '100%' }}
                  onClick={() => {
                    setGemmaOnDemand(false)
                    gemmaOnDemandRef.current = false
                    setGemmaThinking(true)
                    if (DEMO_MODE) {
                      setTimeout(() => {
                        setGemmaThinking(false)
                        setDemoGemmaShown(true)
                      }, 1500)
                    } else if (poseDataRef.current) {
                      void runGemma(poseDataRef.current.measurements, poseDataRef.current.drag)
                    }
                  }}
                >
                  Ask Gemma for suggestions
                </button>
              )}
            </div>
          )}

        </div>
      </div>

      {/* ── Full-width footer ── */}
      <footer style={{
        flexShrink: 0,
        borderTop: '1px solid #1a1a1a',
        padding: '10px 24px',
        fontFamily: 'var(--mono)', fontSize: 10, color: '#666',
        textAlign: 'center', lineHeight: 1.8,
        background: '#0a0a0a',
      }}>
        Made by{' '}
        <a href="https://linkedin.com/in/karolinadubiel" target="_blank" rel="noopener noreferrer"
          style={{ color: 'var(--green)', textDecoration: 'none' }}>
          Karolina Dubiel
        </a>
        {' '}for LAHacks 2026.{' '}
        <a href="https://github.com/dubiels/aeromaxx" target="_blank" rel="noopener noreferrer"
          style={{ color: 'var(--green)', textDecoration: 'none' }}>
          Learn more about the algorithm →
        </a>
      </footer>
    </div>
  )
}
