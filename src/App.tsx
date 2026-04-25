import { useState, useCallback } from 'react'
import ModelViewer from './ModelViewer'
import { extractLandmarks } from './pose'
import { calculateDrag } from './drag'
import { getRecommendations } from './groq'
import { createImageTo3DTask, pollTask } from './meshy'
import type { AnalysisState } from './types'

const CLOUD_NAME = import.meta.env.VITE_CLOUDINARY_CLOUD_NAME as string
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
          tabIcon: '#00ff88', menuIcons: '#666',
          textDark: '#e8e8e8', textLight: '#080808',
          link: '#00ff88', action: '#00ff88',
          inactiveTabIcon: '#444', error: '#ef5350',
          inProgress: '#4fc3f7', complete: '#00ff88',
          sourceBg: '#1a1a1a',
        },
      },
    },
    (err, result) => {
      if (!err && result.event === 'success') cb(result.info.secure_url)
    }
  )
}

const INIT: AnalysisState = {
  landmarks: null, measurements: null, drag: null,
  recommendations: null, visualRecommendations: null,
  loading: false, loadingMessage: '', error: null,
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function SectionLabel({ n, children }: { n: string; children: React.ReactNode }) {
  return (
    <div style={{
      fontFamily: 'var(--mono)', fontSize: 9, letterSpacing: 4,
      color: '#00ff88', textTransform: 'uppercase',
      marginBottom: 14, paddingBottom: 10, borderBottom: '1px solid #1a1a1a',
    }}>
      {n} // {children}
    </div>
  )
}

function DataRow({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', gap: 8,
      fontFamily: 'var(--mono)', fontSize: 12, lineHeight: 2.1,
    }}>
      <span style={{ color: '#444' }}>{label}</span>
      <span style={{ color: accent ? '#ffffff' : '#00ff88', fontWeight: accent ? 700 : 400, textAlign: 'right' }}>
        {value}
      </span>
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
        border: `1px dashed ${hover ? '#00ff88' : '#2a2a2a'}`,
        background: 'transparent', padding: '24px 20px',
        cursor: 'pointer', transition: 'border-color 0.15s',
      }}
    >
      <div style={{
        fontFamily: 'var(--mono)', fontSize: 10, letterSpacing: 4,
        color: hover ? '#00ff88' : '#3a3a3a', marginBottom: 10,
        textTransform: 'uppercase',
      }}>
        Upload Subject Photograph
      </div>
      <div style={{ fontFamily: 'var(--sans)', fontSize: 12, color: '#3a3a3a', lineHeight: 1.65 }}>
        Front-facing, full body, arms slightly away from torso.<br />
        Plain background preferred. Good lighting required.
      </div>
    </button>
  )
}

// ─── App ─────────────────────────────────────────────────────────────────────

export default function App() {
  const [frontUrl, setFrontUrl] = useState<string | null>(null)
  const [meshyModelUrl, setMeshyModelUrl] = useState<string | null>(null)
  const [meshyLoading, setMeshyLoading] = useState(false)
  const [meshyMessage, setMeshyMessage] = useState('')
  const [analysis, setAnalysis] = useState<AnalysisState>(INIT)

  // ── Pipelines ──────────────────────────────────────────────────────────────

  async function runAnalysis(url: string) {
    setAnalysis(s => ({ ...s, loading: true, loadingMessage: 'LOADING PHOTOGRAPHIC DATA...', error: null }))
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

      setAnalysis(s => ({ ...s, loadingMessage: 'EXTRACTING BIOMECHANICAL LANDMARKS...' }))
      const { landmarks, measurements } = await extractLandmarks(canvas)

      setAnalysis(s => ({ ...s, loadingMessage: 'COMPUTING AERODYNAMIC COEFFICIENTS...' }))
      const drag = calculateDrag(measurements)

      setAnalysis(s => ({ ...s, landmarks, measurements, drag, loading: false, loadingMessage: '' }))

      setAnalysis(s => ({ ...s, loading: true, loadingMessage: 'QUERYING AERODYNAMICS DATABASE...' }))
      const recommendations = await getRecommendations(measurements, drag)
      setAnalysis(s => ({ ...s, recommendations, loading: false, loadingMessage: '' }))
    } catch (err) {
      setAnalysis(s => ({
        ...s, loading: false, loadingMessage: '',
        error: err instanceof Error ? err.message : 'Analysis failed',
      }))
    }
  }

  async function runMeshy(url: string) {
    const key = import.meta.env.VITE_MESHY_API_KEY
    if (!key || key.startsWith('your_')) return
    setMeshyLoading(true)
    setMeshyMessage('GENERATING 3D MODEL...')
    try {
      const taskId = await createImageTo3DTask(url)
      const glbUrl = await pollTask(taskId, pct => {
        setMeshyMessage(`GENERATING 3D MODEL... ${pct}%`)
      })
      setMeshyModelUrl(glbUrl)
    } catch (err) {
      console.error('[AeroMaxx] Meshy failed:', err)
    } finally {
      setMeshyLoading(false)
      setMeshyMessage('')
    }
  }

  const handleUpload = useCallback((url: string) => {
    setFrontUrl(url)
    setMeshyModelUrl(null)
    setAnalysis(INIT)
    void runAnalysis(url)
    void runMeshy(url)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const { measurements: m, drag: d } = analysis

  return (
    <div className="app-shell">

      {/* ── Header ── */}
      <header className="app-header">
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 16 }}>
          <span style={{
            fontFamily: 'var(--mono)', fontSize: 17, fontWeight: 700,
            color: '#00ff88', letterSpacing: 5,
          }}>
            AEROMAXX
          </span>
        </div>
        <span style={{
          fontFamily: 'var(--mono)', fontSize: 9, letterSpacing: 3,
          color: '#2a2a2a', textTransform: 'uppercase',
        }}>
          AERODYNAMIC BODY ANALYSIS SYSTEM V2.1
        </span>
      </header>

      {/* ── Two-column body ── */}
      <div className="app-body">

        {/* Left — 3D viewer */}
        <div className="viewer-col">
          <ModelViewer
            modelUrl={meshyModelUrl}
            cdValue={d?.Cd ?? 0.8}
            loading={meshyLoading}
            loadingMessage={meshyMessage}
          />
        </div>

        {/* Right — upload + stats */}
        <div className="stats-col">

          {/* 01 Upload */}
          <div className="panel-section">
            <SectionLabel n="01">SUBJECT INPUT</SectionLabel>
            {!frontUrl ? (
              <UploadPrompt onUpload={() => openWidget(handleUpload)} />
            ) : (
              <div>
                <div style={{ position: 'relative' }}>
                  <img
                    src={frontUrl}
                    alt="Subject"
                    style={{
                      width: '100%', maxHeight: 180, objectFit: 'cover',
                      objectPosition: 'top', display: 'block',
                      border: '1px solid #1a1a1a',
                    }}
                  />
                  <div style={{
                    position: 'absolute', top: 6, left: 6,
                    fontFamily: 'var(--mono)', fontSize: 8, letterSpacing: 2,
                    color: '#00ff88', background: 'rgba(8,8,8,0.8)', padding: '2px 6px',
                  }}>
                    FRONT VIEW
                  </div>
                </div>

                {analysis.loading && (
                  <div style={{
                    fontFamily: 'var(--mono)', fontSize: 10, color: '#00ff88',
                    letterSpacing: 2, marginTop: 12,
                  }} className="pulse">
                    {analysis.loadingMessage}
                  </div>
                )}

                {analysis.error && (
                  <div style={{
                    fontFamily: 'var(--mono)', fontSize: 11, color: '#ef5350',
                    border: '1px solid #ef5350', padding: '10px 14px', marginTop: 12,
                    letterSpacing: 1,
                  }}>
                    ⚠ {analysis.error}
                  </div>
                )}

                <button
                  className="btn-secondary"
                  onClick={() => openWidget(handleUpload)}
                  style={{ marginTop: 12, width: '100%' }}
                >
                  NEW SUBJECT
                </button>
              </div>
            )}
          </div>

          {/* 02 Analysis report */}
          {d && m && (
            <div className="panel-section">
              <SectionLabel n="02">AERODYNAMIC REPORT</SectionLabel>
              <DataRow label="Drag Coefficient (Cd)" value={d.Cd.toFixed(4)} accent />
              <DataRow label="Frontal Area" value={`${d.frontalArea.toFixed(4)} m²`} />
              <DataRow label="Shoulder Width" value={`${m.realShoulderWidth.toFixed(4)} m`} />
              <DataRow label="Hip Width" value={`${m.realHipWidth.toFixed(4)} m`} />
              <DataRow label="Hunch Score (0–1)" value={m.hunchScore.toFixed(4)} />
              <DataRow label="Drag Force @ Walk" value={`${d.dragForce.toFixed(4)} N`} />
              <div style={{ borderTop: '1px solid #1a1a1a', margin: '8px 0' }} />
              <DataRow label="Lifetime Energy Lost" value={`${d.lifetimeEnergy.toLocaleString(undefined, { maximumFractionDigits: 0 })} J`} accent />
              <DataRow label="Energy Wasted" value={`${d.bigMacs.toFixed(1)} Big Macs`} />
              <DataRow label="Time Lost to Drag" value={`${d.daysLost.toFixed(1)} days of your life`} />
              <DataRow label="Postural Penalty" value={`+${d.posturalPenalty.toFixed(4)} Cd`} />
            </div>
          )}

          {/* 03 Drag reduction protocol */}
          {analysis.recommendations && (
            <div className="panel-section">
              <SectionLabel n="03">DRAG REDUCTION PROTOCOL</SectionLabel>
              <div style={{
                fontFamily: 'var(--sans)', fontSize: 13, color: '#ccc',
                lineHeight: 1.8, whiteSpace: 'pre-wrap',
              }}>
                {analysis.recommendations}
              </div>
            </div>
          )}

          <div style={{
            fontFamily: 'var(--mono)', fontSize: 8, color: '#1e1e1e',
            letterSpacing: 3, textAlign: 'center', paddingTop: 8,
          }}>
            AEROMAXX SYSTEMS // BIOMECHANICAL EFFICIENCY DIVISION
          </div>

        </div>
      </div>
    </div>
  )
}
