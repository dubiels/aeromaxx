import { useState, useCallback, useRef, useEffect } from 'react'
import ModelViewer from './ModelViewer'
import { extractLandmarks } from './pose'
import { calculateDrag } from './drag'
import { getRecommendations } from './gemma'
import { createImageTo3DTask, pollTask } from './meshy'
import { saveSubject, getLeaderboard, uploadGlb, type SubjectRecord } from './db'
import { listSucceededTasks, type MeshyListItem } from './meshy'
import type { AnalysisState, GlbMeasurements, BodyMeasurements, DragResults } from './types'

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

interface GalleryItem {
  glbUrl: string
  thumb: string
}

// ─── Sub-components ───────────────────────────────────────────────────────────

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

function DataRow({ label, value, accent, dim }: { label: string; value: string; accent?: boolean; dim?: boolean }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', gap: 8,
      fontFamily: 'var(--mono)', fontSize: 12, lineHeight: 2.1,
    }}>
      <span style={{ color: dim ? '#777' : '#aaa' }}>{label}</span>
      <span style={{ color: accent ? '#ffffff' : dim ? '#888' : 'var(--green)', fontWeight: accent ? 700 : 400, textAlign: 'right' }}>
        {value}
      </span>
    </div>
  )
}

function GalleryBar({ items, onLoad }: { items: GalleryItem[]; onLoad: (item: GalleryItem) => void }) {
  const [hovered, setHovered] = useState(false)
  if (items.length === 0) return null

  const W = 56, H = 42, OVERLAP = 38

  return (
    <div
      style={{
        position: 'relative',
        height: H,
        width: hovered ? items.length * (W + 4) - 4 : W + (items.length - 1) * (W - OVERLAP),
        transition: 'width 0.22s ease',
        overflow: 'hidden',
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {items.map((item, i) => (
        <div
          key={item.glbUrl}
          onClick={() => onLoad(item)}
          style={{
            position: 'absolute',
            left: hovered ? i * (W + 4) : i * (W - OVERLAP),
            top: 0,
            width: W,
            height: H,
            cursor: 'pointer',
            border: '1px solid #2a2a2a',
            overflow: 'hidden',
            transition: 'left 0.22s ease',
            zIndex: items.length - i,
          }}
        >
          <img src={item.thumb} style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} />
        </div>
      ))}
    </div>
  )
}

function Leaderboard({ onClose, onLoadRecord, onImport }: {
  onClose: () => void
  onLoadRecord: (r: SubjectRecord) => void
  onImport: (imageUrl: string, glbUrl: string) => void
}) {
  const [records, setRecords] = useState<SubjectRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [showImport, setShowImport] = useState(false)
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
        <div style={{ borderBottom: '1px solid #1a1a1a', padding: '10px 20px', flexShrink: 0 }}>
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
        </div>

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

          {records.map((r, i) => (
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
                color: i === 0 ? 'var(--green)' : '#555',
                width: 24, flexShrink: 0, textAlign: 'right',
              }}>
                {i + 1}
              </span>
              <img
                src={r.image_url}
                style={{ width: 44, height: 56, objectFit: 'cover', flexShrink: 0, border: '1px solid #1a1a1a' }}
              />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontFamily: 'var(--mono)', fontSize: 13, color: 'var(--green)' }}>
                  Cd {r.cd_score.toFixed(4)}
                </div>
                <div style={{ fontFamily: 'var(--mono)', fontSize: 9, color: '#555', marginTop: 2 }}>
                  {new Date(r.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                  {' · click to load 3D model'}
                </div>
              </div>
              {i === 0 && (
                <span style={{ fontFamily: 'var(--mono)', fontSize: 8, color: 'var(--green)', letterSpacing: 1 }}>
                  most aerodynamic
                </span>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function ThinkingSpike() {
  const CHARS = ['+', '-', "'", '`']
  const [t, setT] = useState(0)
  useEffect(() => {
    const id = setInterval(() => setT(n => (n + 1) % 4), 90)
    return () => clearInterval(id)
  }, [])
  const str = Array.from({ length: 9 }, (_, i) => CHARS[(t + i) % 4]).join(' ')
  return (
    <div style={{ fontFamily: 'var(--mono)', fontSize: 13, color: 'var(--green)', letterSpacing: 2 }}>
      {str}
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
  const [gallery, setGallery]             = useState<GalleryItem[]>([])
  const [showLeaderboard, setShowLeaderboard] = useState(false)
  const [gemmaOnDemand, setGemmaOnDemand]     = useState(false)
  const [gemmaThinking, setGemmaThinking]     = useState(false)

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
      userModelRenderedRef.current = true
      tryRevealGemma()
      if (!skipNextSaveRef.current && drag && frontUrlRef.current && meshyUrlRef.current) {
        void saveSubject(frontUrlRef.current, meshyUrlRef.current, drag.Cd)
      }
      skipNextSaveRef.current = false
    }
  }, [tryRevealGemma]) // eslint-disable-line react-hooks/exhaustive-deps

  const handleSnapshot = useCallback((dataUrl: string, glbUrl: string) => {
    setGallery(prev => prev.some(g => g.glbUrl === glbUrl) ? prev : [...prev, { glbUrl, thumb: dataUrl }])
  }, [])

  const handleGalleryLoad = useCallback((item: GalleryItem) => {
    setMeshyModelUrl(item.glbUrl)
  }, [])

  const handleManualImport = useCallback((imageUrl: string, glbUrl: string) => {
    frontUrlRef.current          = imageUrl
    meshyUrlRef.current          = glbUrl
    gemmaOnDemandRef.current     = true
    setGemmaOnDemand(true)
    // skipNextSaveRef stays false — this is a new entry, we want to persist it
    setFrontUrl(imageUrl)
    setMeshyModelUrl(glbUrl)
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
    setFrontUrl(record.image_url)
    setMeshyModelUrl(record.glb_url)
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
    // If Meshy is disabled, no 3D model will ever render — reveal Gemma immediately on completion
    const meshyKey = import.meta.env.VITE_MESHY_API_KEY as string
    if (!meshyKey || meshyKey.startsWith('your_')) userModelRenderedRef.current = true
    void runPoseAnalysis(url)
    void runMeshy(url)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const { measurements: m, drag: d, glbMeasurements: glb } = analysis

  return (
    <div className="app-shell">

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
            onSnapshot={handleSnapshot}
          />
          {gallery.length > 0 && (
            <div style={{ position: 'absolute', bottom: 16, left: 16, zIndex: 10 }}>
              <GalleryBar items={gallery} onLoad={handleGalleryLoad} />
            </div>
          )}
        </div>

        {/* Right — upload + stats */}
        <div className="stats-col">

          {/* 01 Upload */}
          <div className="panel-section">
            <SectionLabel n="1">SUBJECT INPUT</SectionLabel>
            {!frontUrl ? (
              <UploadPrompt onUpload={() => openWidget(handleUpload)} />
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

                <button
                  className="btn-secondary"
                  onClick={() => openWidget(handleUpload)}
                  style={{ marginTop: 12, width: '100%' }}
                >
                  New subject
                </button>
              </div>
            )}
          </div>

          {/* 02 Analysis report — appears as soon as pose analysis completes */}
          {d && m && (
            <div className="panel-section">
              <SectionLabel n="2">AERODYNAMIC REPORT</SectionLabel>

              <DataRow label="Drag Coefficient (Cd)" value={d.Cd.toFixed(4)} accent />
              <DataRow label="Frontal Area" value={`${d.frontalArea.toFixed(4)} m²`} />
              <DataRow
                label="  └ area source"
                value={glb ? '3D model' : '2D estimate'}
                dim
              />

              {glb && (
                <>
                  <DataRow label="Body Width (3D)" value={`${glb.realWidth.toFixed(3)} m`} />
                  <DataRow label="Body Depth (3D)" value={`${glb.realDepth.toFixed(3)} m`} />
                  <DataRow label="Depth/Width Ratio" value={glb.depthToWidthRatio.toFixed(3)} />
                </>
              )}

              <DataRow label="Shoulder Width" value={`${m.realShoulderWidth.toFixed(4)} m`} />
              <DataRow label="Hip Width" value={`${m.realHipWidth.toFixed(4)} m`} />
              <DataRow label="Hunch Score (0–1)" value={m.hunchScore.toFixed(4)} />
              <DataRow label="Postural Cd Penalty" value={`+${d.posturalPenalty.toFixed(4)}`} />
              <DataRow label="Drag Force @ Walk" value={`${d.dragForce.toFixed(4)} N`} />

              <div style={{ borderTop: '1px solid #1a1a1a', margin: '8px 0' }} />
              <DataRow label="Lifetime Energy Lost" value={`${d.lifetimeEnergy.toLocaleString(undefined, { maximumFractionDigits: 0 })} J`} accent />
              <DataRow label="Energy Wasted" value={`${d.bigMacs.toFixed(1)} Big Macs`} />
              <DataRow label="Time Lost to Drag" value={`${d.daysLost.toFixed(1)} days of your life`} />
            </div>
          )}

          {/* 03 Drag reduction protocol */}
          {(analysis.recommendations || (d && m && (gemmaOnDemand || gemmaThinking))) && (
            <div className="panel-section">
              <SectionLabel n="3">Drag reduction protocol</SectionLabel>
              {analysis.recommendations ? (
                <div style={{
                  fontFamily: 'var(--sans)', fontSize: 13, color: '#ccc',
                  lineHeight: 1.8, whiteSpace: 'pre-wrap',
                }}>
                  {analysis.recommendations}
                </div>
              ) : gemmaThinking ? (
                <ThinkingSpike />
              ) : (
                <button
                  className="btn-secondary"
                  style={{ width: '100%' }}
                  onClick={() => {
                    gemmaOnDemandRef.current = false
                    setGemmaOnDemand(false)
                    setGemmaThinking(true)
                    if (poseDataRef.current) {
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
