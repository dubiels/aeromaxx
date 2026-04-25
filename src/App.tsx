import { useState, useCallback, useRef } from 'react'
import ModelViewer from './ModelViewer'
import { extractLandmarks } from './pose'
import { calculateDrag } from './drag'
import { getRecommendations } from './gemma'
import { createImageTo3DTask, pollTask } from './meshy'
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

async function uploadGlbToCloudinary(glbUrl: string): Promise<string> {
  const proxyUrl = glbUrl.replace('https://assets.meshy.ai', '/meshy-assets')
  const blob = await fetch(proxyUrl).then(r => {
    if (!r.ok) throw new Error(`GLB fetch failed: ${r.status}`)
    return r.blob()
  })
  const form = new FormData()
  form.append('file', blob, 'model.glb')
  form.append('upload_preset', UPLOAD_PRESET)
  const res = await fetch(`https://api.cloudinary.com/v1_1/${CLOUD_NAME}/raw/upload`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) throw new Error(`Cloudinary GLB upload failed: ${res.status}`)
  const data = await res.json()
  return data.secure_url as string
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

  // Pipeline state refs — stable across renders, safe to read in async callbacks
  const poseDataRef     = useRef<{ measurements: BodyMeasurements; drag: DragResults } | null>(null)
  const glbReadyRef     = useRef<GlbMeasurements | null>(null)
  const gemmaRunningRef = useRef(false)

  // ── Gemma gate: fires only when BOTH pose AND 3D geometry are available ──────

  const tryRunGemma = useCallback(async () => {
    if (gemmaRunningRef.current) return
    const pose = poseDataRef.current
    const glb  = glbReadyRef.current
    if (!pose || !glb) return

    gemmaRunningRef.current = true
    const drag = calculateDrag(pose.measurements, glb)
    setAnalysis(s => ({ ...s, drag, loading: true, loadingMessage: 'Querying aerodynamics database...' }))
    try {
      const recommendations = await getRecommendations(pose.measurements, drag, glb)
      setAnalysis(s => ({ ...s, recommendations, loading: false, loadingMessage: '' }))
    } catch (err) {
      setAnalysis(s => ({
        ...s, loading: false, loadingMessage: '',
        error: err instanceof Error ? err.message : 'Gemma analysis failed',
      }))
    } finally {
      gemmaRunningRef.current = false
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

      void tryRunGemma()
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
    setMeshyMessage('Generating 3D model...')
    try {
      const base64 = await toBase64(url)
      const taskId = await createImageTo3DTask(base64)
      const glbUrl = await pollTask(taskId, pct => {
        setMeshyMessage(`Generating 3D model... ${pct}%`)
      })

      setMeshyMessage('Uploading to CDN...')
      let finalUrl = glbUrl
      try {
        finalUrl = await uploadGlbToCloudinary(glbUrl)
      } catch (err) {
        console.warn('[AeroMaxx] Cloudinary GLB upload failed, using proxy URL:', err)
        finalUrl = glbUrl.replace('https://assets.meshy.ai', '/meshy-assets')
      }

      setMeshyModelUrl(finalUrl)
    } catch (err) {
      console.error('[AeroMaxx] Meshy failed:', err)
    } finally {
      setMeshyLoading(false)
      setMeshyMessage('')
    }
  }

  // Fires when ModelViewer loads and measures the GLB geometry.
  // Stores the 3D data and attempts to run Gemma if pose is also ready.
  const handleGeometryMeasured = useCallback((glb: GlbMeasurements) => {
    glbReadyRef.current = glb
    setAnalysis(s => ({ ...s, glbMeasurements: glb }))
    void tryRunGemma()
  }, [tryRunGemma])

  const handleSnapshot = useCallback((dataUrl: string, glbUrl: string) => {
    setGallery(prev => prev.some(g => g.glbUrl === glbUrl) ? prev : [...prev, { glbUrl, thumb: dataUrl }])
  }, [])

  const handleGalleryLoad = useCallback((item: GalleryItem) => {
    setMeshyModelUrl(item.glbUrl)
  }, [])

  const handleUpload = useCallback((url: string) => {
    setFrontUrl(url)
    setMeshyModelUrl(null)
    setAnalysis(INIT)
    poseDataRef.current     = null
    glbReadyRef.current     = null
    gemmaRunningRef.current = false
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
    </header>


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

          {/* 03 Drag reduction protocol — appears after both pose + 3D geometry complete */}
          {analysis.recommendations && (
            <div className="panel-section">
              <SectionLabel n="3">Drag reduction protocol</SectionLabel>
              <div style={{
                fontFamily: 'var(--sans)', fontSize: 13, color: '#ccc',
                lineHeight: 1.8, whiteSpace: 'pre-wrap',
              }}>
                {analysis.recommendations}
              </div>
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
