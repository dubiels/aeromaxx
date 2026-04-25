import { useState, useRef, useCallback, useEffect } from 'react'
import type { AnalysisState } from './types'
import { extractLandmarks } from './pose'
import { calculateDrag } from './drag'
import { getRecommendations, getVisionAnalysis } from './groq'

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

function openWidget(callback: (url: string, publicId: string) => void) {
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
          window: '#111111',
          windowBorder: '#2a2a2a',
          tabIcon: '#00ff88',
          menuIcons: '#888',
          textDark: '#e8e8e8',
          textLight: '#0a0a0a',
          link: '#00ff88',
          action: '#00ff88',
          inactiveTabIcon: '#555',
          error: '#ef5350',
          inProgress: '#4fc3f7',
          complete: '#00ff88',
          sourceBg: '#1a1a1a',
        },
      },
    },
    (error, result) => {
      if (!error && result.event === 'success') {
        callback(result.info.secure_url, result.info.public_id)
      }
    }
  )
}

function loadImageOntoCanvas(url: string, canvas: HTMLCanvasElement): Promise<void> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      canvas.width = img.naturalWidth
      canvas.height = img.naturalHeight
      canvas.getContext('2d')!.drawImage(img, 0, 0)
      resolve()
    }
    img.onerror = () => reject(new Error('Failed to load image. Ensure Cloudinary CORS is enabled.'))
    img.src = url
  })
}

function extractEdges(canvas: HTMLCanvasElement): { edgeTop: Int32Array; edgeBottom: Int32Array } {
  const { width: w, height: h } = canvas
  const data = canvas.getContext('2d')!.getImageData(0, 0, w, h).data
  const edgeTop = new Int32Array(w).fill(-1)
  const edgeBottom = new Int32Array(w).fill(-1)

  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      if (data[(y * w + x) * 4 + 3] > 50) {
        if (edgeTop[x] === -1) edgeTop[x] = y
        edgeBottom[x] = y
      }
    }
  }

  console.log('[AeroMaxx] Edge arrays computed, sample col 200:',
    { top: edgeTop[200], bottom: edgeBottom[200] })

  return { edgeTop, edgeBottom }
}

type Trail = { x: number; y: number; deflection: number }

function startCFD(
  canvas: HTMLCanvasElement,
  silhouetteUrl: string,
  edgeTop: Int32Array,
  edgeBottom: Int32Array
): () => void {
  const ctx = canvas.getContext('2d')!
  const w = canvas.width
  const h = canvas.height
  const N = 80
  const SPEED = 2.5
  const INFLUENCE = 40

  const particles = Array.from({ length: N }, (_, i) => {
    const startY = (i + 0.5) * (h / N)
    return { x: Math.random() * w, y: startY, startY, trail: [] as Trail[] }
  })

  const silImg = new Image()
  silImg.crossOrigin = 'anonymous'
  silImg.src = silhouetteUrl

  function deflectionColor(d: number): string {
    if (d < 2) return '#4fc3f7'
    if (d < 5) return '#81c784'
    if (d < 10) return '#ffb74d'
    return '#ef5350'
  }

  let animId: number
  let cancelled = false

  function tick() {
    if (cancelled) return

    ctx.fillStyle = '#0a0a0a'
    ctx.fillRect(0, 0, w, h)

    for (const p of particles) {
      if (p.x > w + 20) {
        p.x = 0
        p.y = p.startY
        p.trail = []
      }

      const col = Math.round(p.x)
      let deflection = 0
      let dy = 0

      if (col >= 0 && col < w) {
        const top = edgeTop[col]
        const bot = edgeBottom[col]

        if (top !== -1 && bot !== -1) {
          const distTop = p.y - top     // positive = below top edge
          const distBot = bot - p.y     // positive = above bottom edge

          if (distTop >= 0 && distBot >= 0) {
            // Inside silhouette — push to nearest edge
            deflection = INFLUENCE
            dy = distTop <= distBot ? -(INFLUENCE * 0.4) : INFLUENCE * 0.4
          } else {
            // Outside — deflect if within influence zone
            const gapAbove = -distTop   // how far above the top edge we are
            const gapBelow = -distBot   // how far below the bottom edge we are

            if (gapAbove > 0 && gapAbove < INFLUENCE) {
              deflection = INFLUENCE - gapAbove
              dy = -deflection * 0.12
            } else if (gapBelow > 0 && gapBelow < INFLUENCE) {
              deflection = INFLUENCE - gapBelow
              dy = deflection * 0.12
            }
          }
        }
      }

      p.y = Math.max(0, Math.min(h - 1, p.y + dy))
      p.trail.push({ x: p.x, y: p.y, deflection })
      if (p.trail.length > 35) p.trail.shift()
      p.x += SPEED

      if (p.trail.length > 1) {
        for (let i = 1; i < p.trail.length; i++) {
          const prev = p.trail[i - 1]
          const curr = p.trail[i]
          ctx.beginPath()
          ctx.strokeStyle = deflectionColor(curr.deflection)
          ctx.globalAlpha = 0.35 + 0.65 * (i / p.trail.length)
          ctx.lineWidth = 1.5
          ctx.moveTo(prev.x, prev.y)
          ctx.lineTo(curr.x, curr.y)
          ctx.stroke()
        }
      }
    }

    ctx.globalAlpha = 1

    // Draw silhouette on top — transparent pixels show streamlines, body hides them
    if (silImg.complete && silImg.naturalWidth > 0) {
      ctx.drawImage(silImg, 0, 0, w, h)
    }

    animId = requestAnimationFrame(tick)
  }

  tick()
  return () => { cancelled = true; cancelAnimationFrame(animId) }
}

// ─── Sub-components ──────────────────────────────────────────────────────────

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      fontFamily: 'var(--mono)', fontSize: 10, letterSpacing: 4,
      color: 'var(--text-muted)', textTransform: 'uppercase',
      marginBottom: 16, paddingBottom: 10, borderBottom: '1px solid var(--border)',
    }}>
      {children}
    </div>
  )
}

function DataRow({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 16, lineHeight: 2.2 }}>
      <span style={{ color: 'var(--text-dim)', minWidth: 240 }}>{label}:</span>
      <span style={{ color: highlight ? '#fff' : 'var(--green)', textAlign: 'right', fontWeight: highlight ? 700 : 400 }}>
        {value}
      </span>
    </div>
  )
}

function UploadArea({ onClick, label, sub, dim }: {
  onClick: () => void; label: string; sub: string; dim?: boolean
}) {
  const [hover, setHover] = useState(false)
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        width: '100%',
        border: `2px dashed ${hover ? 'var(--green)' : dim ? 'var(--border)' : '#2e2e2e'}`,
        background: 'transparent',
        padding: '44px 32px',
        cursor: 'pointer',
        textAlign: 'center',
        transition: 'border-color 0.2s',
        borderRadius: 3,
      }}
    >
      <div style={{
        fontFamily: 'var(--mono)', fontSize: 11, letterSpacing: 4,
        color: dim ? 'var(--text-dim)' : hover ? 'var(--green)' : '#4a4a4a',
        marginBottom: 10, transition: 'color 0.2s',
      }}>
        {label}
      </div>
      <div style={{ fontFamily: 'var(--sans)', fontSize: 12, color: 'var(--text-muted)' }}>
        {sub}
      </div>
      <div style={{
        marginTop: 20, fontSize: 28, lineHeight: 1,
        color: hover ? 'var(--green)' : dim ? '#2a2a2a' : '#333',
        transition: 'color 0.2s',
      }}>
        +
      </div>
    </button>
  )
}

// ─── Main App ────────────────────────────────────────────────────────────────

const INITIAL_STATE: AnalysisState = {
  landmarks: null, measurements: null, drag: null,
  recommendations: null, visualRecommendations: null,
  loading: false, loadingMessage: '', error: null,
}

export default function App() {
  const [frontUrl, setFrontUrl] = useState<string | null>(null)
  const [sideUrl, setSideUrl] = useState<string | null>(null)
  const [sidePublicId, setSidePublicId] = useState<string | null>(null)
  const [state, setState] = useState<AnalysisState>(INITIAL_STATE)

  const frontCanvasRef = useRef<HTMLCanvasElement>(null)
  const cfdCanvasRef = useRef<HTMLCanvasElement>(null)
  const cfdCleanupRef = useRef<(() => void) | null>(null)
  const processedSideRef = useRef<string | null>(null)

  const setLoading = (msg: string) =>
    setState(s => ({ ...s, loading: true, loadingMessage: msg, error: null }))
  const clearLoading = () =>
    setState(s => ({ ...s, loading: false, loadingMessage: '' }))

  // ── Upload handlers ────────────────────────────────────────────────────────

  const handleFrontUpload = useCallback(() => {
    openWidget((url) => {
      setFrontUrl(url)
      setState(s => ({
        ...s, landmarks: null, measurements: null, drag: null,
        recommendations: null, visualRecommendations: null, error: null,
      }))
    })
  }, [])

  const handleSideUpload = useCallback(() => {
    openWidget((url, publicId) => {
      setSideUrl(url)
      setSidePublicId(publicId)
      processedSideRef.current = null  // allow re-processing on new upload
    })
  }, [])

  // ── Analysis pipeline ──────────────────────────────────────────────────────

  const runAnalysis = useCallback(async () => {
    if (!frontUrl || !frontCanvasRef.current) return

    setState(s => ({ ...s, loading: true, loadingMessage: 'LOADING PHOTOGRAPHIC DATA...', error: null }))

    try {
      await loadImageOntoCanvas(frontUrl, frontCanvasRef.current)

      setState(s => ({ ...s, loadingMessage: 'EXTRACTING BIOMECHANICAL LANDMARKS...' }))
      const { landmarks, measurements } = await extractLandmarks(frontCanvasRef.current)

      setState(s => ({ ...s, loadingMessage: 'COMPUTING AERODYNAMIC COEFFICIENTS...' }))
      const drag = calculateDrag(measurements)

      setState(s => ({ ...s, landmarks, measurements, drag, loading: false, loadingMessage: '' }))

      // Groq recommendations (non-blocking UI update)
      setState(s => ({ ...s, loading: true, loadingMessage: 'QUERYING AERODYNAMICS DATABASE...' }))
      const recommendations = await getRecommendations(measurements, drag)
      setState(s => ({ ...s, recommendations, loading: false, loadingMessage: '' }))

    } catch (err) {
      setState(s => ({
        ...s, loading: false, loadingMessage: '',
        error: err instanceof Error ? err.message : 'Analysis failed. Please try again.',
      }))
    }
  }, [frontUrl])

  // ── CFD pipeline — triggers when side photo + drag results are both ready ──

  useEffect(() => {
    if (!sidePublicId || !state.drag || !cfdCanvasRef.current) return
    if (processedSideRef.current === sidePublicId) return
    processedSideRef.current = sidePublicId

    const canvas = cfdCanvasRef.current
    const bgRemovedUrl = `https://res.cloudinary.com/${CLOUD_NAME}/image/upload/e_background_removal/${sidePublicId}`
    const captureFrontUrl = frontUrl
    const captureMeasurements = state.measurements

    ;(async () => {
      setLoading('EXTRACTING BODY SILHOUETTE...')

      const img = new Image()
      img.crossOrigin = 'anonymous'

      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve()
        img.onerror = () => reject(new Error(
          'Silhouette extraction failed. Ensure the Cloudinary AI Background Removal add-on is enabled on your account.'
        ))
        img.src = bgRemovedUrl
      }).catch(err => {
        setState(s => ({ ...s, loading: false, loadingMessage: '', error: (err as Error).message }))
        processedSideRef.current = null
        throw err
      })

      // Scale to fit display, max 640px wide
      const MAX = 640
      const scale = Math.min(MAX / img.naturalWidth, MAX / img.naturalHeight, 1)
      canvas.width = Math.round(img.naturalWidth * scale)
      canvas.height = Math.round(img.naturalHeight * scale)

      // Draw to offscreen canvas for edge extraction
      const offscreen = document.createElement('canvas')
      offscreen.width = canvas.width
      offscreen.height = canvas.height
      offscreen.getContext('2d')!.drawImage(img, 0, 0, canvas.width, canvas.height)

      const { edgeTop, edgeBottom } = extractEdges(offscreen)

      // Stop any previous CFD animation
      if (cfdCleanupRef.current) {
        cfdCleanupRef.current()
        cfdCleanupRef.current = null
      }

      setState(s => ({ ...s, loading: false, loadingMessage: '' }))

      const cleanup = startCFD(canvas, bgRemovedUrl, edgeTop, edgeBottom)
      cfdCleanupRef.current = cleanup

      // Vision analysis if we have front photo
      if (captureFrontUrl && captureMeasurements) {
        setState(s => ({ ...s, loading: true, loadingMessage: 'PERFORMING VISUAL AERODYNAMIC SCAN...' }))
        try {
          const visualRecommendations = await getVisionAnalysis(captureFrontUrl, bgRemovedUrl)
          setState(s => ({ ...s, visualRecommendations, loading: false, loadingMessage: '' }))
        } catch (err) {
          console.error('[AeroMaxx] Vision analysis failed:', err)
          clearLoading()
        }
      }
    })()
  }, [sidePublicId, state.drag]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    return () => { if (cfdCleanupRef.current) cfdCleanupRef.current() }
  }, [])

  const { measurements: m, drag: d } = state

  return (
    <div style={{ minHeight: '100vh', padding: '48px 20px 80px', maxWidth: 860, margin: '0 auto' }}>

      {/* Header */}
      <header style={{ marginBottom: 72, textAlign: 'center' }}>
        <div style={{
          fontFamily: 'var(--mono)', fontSize: 10, letterSpacing: 6,
          color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 10,
        }}>
          BIOMECHANICAL SYSTEMS DIVISION
        </div>
        <h1 style={{
          fontFamily: 'var(--mono)', fontSize: 'clamp(36px, 9vw, 64px)',
          fontWeight: 700, color: 'var(--green)', letterSpacing: 6, margin: '0 0 10px',
        }}>
          AEROMAXX
        </h1>
        <div style={{
          fontFamily: 'var(--mono)', fontSize: 11, letterSpacing: 3,
          color: 'var(--text-dim)', textTransform: 'uppercase',
        }}>
          AERODYNAMIC BODY ANALYSIS SYSTEM v2.1
        </div>
      </header>

      {/* Global loading overlay */}
      {state.loading && (
        <div style={{
          position: 'fixed', inset: 0,
          background: 'rgba(10,10,10,0.88)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          zIndex: 200, backdropFilter: 'blur(4px)',
        }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{
              fontFamily: 'var(--mono)', color: 'var(--green)',
              fontSize: 13, letterSpacing: 3,
            }} className="pulse">
              {state.loadingMessage}
            </div>
            <div style={{
              marginTop: 20, fontFamily: 'var(--mono)',
              fontSize: 10, color: 'var(--text-muted)', letterSpacing: 4,
            }}>
              ░░░ PROCESSING ░░░
            </div>
          </div>
        </div>
      )}

      {/* ── SECTION 01: Front photo upload ── */}
      <section style={{ marginBottom: 56 }}>
        <SectionLabel>01 // SUBJECT PHOTOGRAPHIC INPUT</SectionLabel>

        {!frontUrl ? (
          <UploadArea
            onClick={handleFrontUpload}
            label="UPLOAD FRONT-FACING PHOTOGRAPH"
            sub="Full body, neutral stance. Arms slightly away from torso for accurate shoulder extraction."
          />
        ) : (
          <div style={{ display: 'flex', gap: 28, alignItems: 'flex-start', flexWrap: 'wrap' }}>
            <div style={{ position: 'relative', flexShrink: 0 }}>
              <img
                src={frontUrl}
                alt="Front profile"
                style={{ height: 220, display: 'block', border: '1px solid var(--border)', borderRadius: 3 }}
              />
              <div style={{
                position: 'absolute', top: 8, left: 8,
                fontFamily: 'var(--mono)', fontSize: 9, letterSpacing: 2,
                color: 'var(--green)', background: 'rgba(0,0,0,0.75)', padding: '3px 7px',
              }}>
                FRONT VIEW
              </div>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12, paddingTop: 4 }}>
              {!d && (
                <button
                  className="btn-primary"
                  onClick={runAnalysis}
                  disabled={state.loading}
                >
                  RUN AERODYNAMIC ANALYSIS
                </button>
              )}
              {d && (
                <div style={{
                  fontFamily: 'var(--mono)', fontSize: 10, letterSpacing: 2,
                  color: 'var(--green)', padding: '8px 0',
                }}>
                  ✓ ANALYSIS COMPLETE
                </div>
              )}
              <button
                className="btn-secondary"
                onClick={handleFrontUpload}
                disabled={state.loading}
              >
                REPLACE PHOTOGRAPH
              </button>
            </div>
          </div>
        )}
      </section>

      {/* Error display */}
      {state.error && (
        <div style={{
          fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--red)',
          border: '1px solid var(--red)', padding: '14px 18px', marginBottom: 32,
          letterSpacing: 1, lineHeight: 1.6,
        }}>
          ⚠ {state.error}
        </div>
      )}

      {/* ── SECTION 02: Analysis results ── */}
      {d && m && (
        <section style={{ marginBottom: 56 }}>
          <SectionLabel>02 // AERODYNAMIC ANALYSIS REPORT</SectionLabel>
          <div style={{
            fontFamily: 'var(--mono)', fontSize: 13,
            background: 'var(--surface)', border: '1px solid var(--border)',
            padding: '28px 32px',
          }}>
            <div style={{ color: 'var(--green)', marginBottom: 6, letterSpacing: 3, fontSize: 11 }}>
              AERODYNAMIC ANALYSIS REPORT
            </div>
            <div style={{ color: 'var(--border)', marginBottom: 18, fontSize: 11 }}>
              {'━'.repeat(42)}
            </div>
            <DataRow label="Drag Coefficient (Cd)" value={d.Cd.toFixed(4)} highlight />
            <DataRow label="Frontal Area" value={`${d.frontalArea.toFixed(4)} m²`} />
            <DataRow label="Shoulder Width" value={`${m.realShoulderWidth.toFixed(4)} m`} />
            <DataRow label="Hip Width" value={`${m.realHipWidth.toFixed(4)} m`} />
            <DataRow label="Shoulder-to-Hip Ratio" value={m.shoulderToHipRatio.toFixed(4)} />
            <DataRow label="Hunch Score (0–1)" value={m.hunchScore.toFixed(4)} />
            <DataRow label="Drag Force @ Walk Speed" value={`${d.dragForce.toFixed(4)} N`} />
            <div style={{ color: 'var(--border)', margin: '12px 0', fontSize: 11 }}>{'─'.repeat(42)}</div>
            <DataRow label="Lifetime Drag Force" value={`${d.lifetimeEnergy.toLocaleString(undefined, { maximumFractionDigits: 0 })} J`} highlight />
            <DataRow label="Energy Wasted" value={`${d.bigMacs.toFixed(1)} Big Macs`} />
            <DataRow label="Time Lost to Drag" value={`${d.daysLost.toFixed(1)} days of your life`} />
            <DataRow label="Postural Drag Penalty" value={`+${d.posturalPenalty.toFixed(4)} Cd`} />
          </div>
        </section>
      )}

      {/* ── SECTION 03: AI recommendations ── */}
      {state.recommendations && (
        <section style={{ marginBottom: 56 }}>
          <SectionLabel>03 // DRAG REDUCTION PROTOCOL</SectionLabel>
          <div style={{
            background: 'var(--surface)', border: '1px solid var(--border)',
            padding: '28px 32px', color: 'var(--text)', fontSize: 14,
            lineHeight: 1.85, whiteSpace: 'pre-wrap', fontFamily: 'var(--sans)',
          }}>
            {state.recommendations}
          </div>
        </section>
      )}

      {/* ── SECTION 04: CFD visualization ── */}
      {d && (
        <section style={{ marginBottom: 56 }}>
          <SectionLabel>04 // COMPUTATIONAL FLUID DYNAMICS — SILHOUETTE ANALYSIS</SectionLabel>

          {!sideUrl ? (
            <UploadArea
              onClick={handleSideUpload}
              label="UPLOAD SIDE PROFILE PHOTOGRAPH"
              sub="Optional — enables real body silhouette CFD streamline simulation"
              dim
            />
          ) : (
            <div>
              <div style={{
                fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text-muted)',
                letterSpacing: 2, marginBottom: 14,
              }}>
                {state.loading ? 'PROCESSING SILHOUETTE...' : 'STREAMLINE SIMULATION ACTIVE'}
              </div>
              <canvas
                ref={cfdCanvasRef}
                style={{
                  display: 'block', maxWidth: '100%',
                  border: '1px solid var(--border)',
                  boxShadow: '0 0 40px rgba(0,255,136,0.04)',
                }}
              />
              <button
                className="btn-secondary"
                onClick={handleSideUpload}
                style={{ marginTop: 14 }}
                disabled={state.loading}
              >
                REPLACE SIDE PHOTOGRAPH
              </button>
            </div>
          )}
        </section>
      )}

      {/* ── SECTION 05: Visual AI observations ── */}
      {state.visualRecommendations && (
        <section style={{ marginBottom: 56 }}>
          <SectionLabel>05 // VISUAL AERODYNAMIC OBSERVATIONS</SectionLabel>
          <div style={{
            background: 'var(--surface)', border: '1px solid var(--border)',
            padding: '28px 32px', color: 'var(--text)', fontSize: 14,
            lineHeight: 1.85, whiteSpace: 'pre-wrap', fontFamily: 'var(--sans)',
          }}>
            {state.visualRecommendations}
          </div>
        </section>
      )}

      {/* Hidden canvas for pose detection */}
      <canvas ref={frontCanvasRef} style={{ display: 'none' }} />

      <footer style={{
        textAlign: 'center', paddingTop: 40,
        fontFamily: 'var(--mono)', fontSize: 10,
        color: 'var(--text-muted)', letterSpacing: 3,
        borderTop: '1px solid var(--border)',
      }}>
        AEROMAXX SYSTEMS // AERODYNAMIC EFFICIENCY DIVISION // {new Date().getFullYear()}
      </footer>
    </div>
  )
}
