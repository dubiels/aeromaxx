import {
  PoseLandmarker,
  FilesetResolver,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision'
import type { Landmarks, BodyMeasurements } from './types'

let landmarker: PoseLandmarker | null = null

export async function initPoseLandmarker(): Promise<void> {
  if (landmarker) return
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm'
  )
  landmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
      delegate: 'GPU',
    },
    runningMode: 'IMAGE',
    numPoses: 1,
  })
}

function dist(a: NormalizedLandmark, b: NormalizedLandmark, w: number, h: number): number {
  const dx = (a.x - b.x) * w
  const dy = (a.y - b.y) * h
  return Math.sqrt(dx * dx + dy * dy)
}

// Skeleton connections between MediaPipe landmark indices
const CONNECTIONS: [number, number][] = [
  [0, 11], [0, 12],           // nose → shoulders
  [11, 12],                    // shoulder bar
  [11, 13], [13, 15],          // left arm
  [12, 14], [14, 16],          // right arm
  [11, 23], [12, 24],          // torso sides
  [23, 24],                    // hip bar
  [23, 25], [25, 27],          // left leg
  [24, 26], [26, 28],          // right leg
]

const NAMED_LANDMARKS: [number, string][] = [
  [0, 'HEAD'],
  [11, 'L SHOULDER'], [12, 'R SHOULDER'],
  [13, 'L ELBOW'],    [14, 'R ELBOW'],
  [15, 'L WRIST'],    [16, 'R WRIST'],
  [23, 'L HIP'],      [24, 'R HIP'],
  [25, 'L KNEE'],     [26, 'R KNEE'],
  [27, 'L ANKLE'],    [28, 'R ANKLE'],
]

function drawAnnotatedCanvas(source: HTMLCanvasElement, lm: NormalizedLandmark[]): HTMLCanvasElement {
  const w = source.width
  const h = source.height
  const out = document.createElement('canvas')
  out.width = w
  out.height = h
  const ctx = out.getContext('2d')!

  ctx.drawImage(source, 0, 0)
  ctx.fillStyle = 'rgba(0,0,0,0.28)'
  ctx.fillRect(0, 0, w, h)

  // Skeleton lines
  ctx.strokeStyle = '#61e5ff'
  ctx.lineWidth = Math.max(2, w * 0.004)
  ctx.globalAlpha = 0.8
  for (const [a, b] of CONNECTIONS) {
    const pa = lm[a], pb = lm[b]
    if (!pa || !pb) continue
    ctx.beginPath()
    ctx.moveTo(pa.x * w, pa.y * h)
    ctx.lineTo(pb.x * w, pb.y * h)
    ctx.stroke()
  }

  // Landmark dots + labels
  const dotR = Math.max(4, w * 0.007)
  const fontSize = Math.max(9, Math.min(16, w * 0.016))
  ctx.font = `${fontSize}px 'JetBrains Mono', monospace`
  ctx.textAlign = 'left'

  for (const [idx, label] of NAMED_LANDMARKS) {
    const p = lm[idx]
    if (!p) continue
    const px = p.x * w
    const py = p.y * h

    ctx.globalAlpha = 1.0
    ctx.fillStyle = '#61e5ff'
    ctx.beginPath()
    ctx.arc(px, py, dotR, 0, Math.PI * 2)
    ctx.fill()

    ctx.globalAlpha = 0.85
    ctx.fillStyle = '#61e5ff'
    ctx.fillText(label, px + dotR + 3, py + fontSize * 0.38)
  }

  ctx.globalAlpha = 1.0
  return out
}

export async function extractLandmarks(
  canvas: HTMLCanvasElement
): Promise<{ landmarks: Landmarks; measurements: BodyMeasurements; annotatedCanvas: HTMLCanvasElement }> {
  await initPoseLandmarker()
  if (!landmarker) throw new Error('Landmarker not initialized')

  const result = landmarker.detect(canvas)
  if (!result.landmarks || result.landmarks.length === 0) {
    throw new Error('No pose detected. Please upload a clear full-body photo.')
  }

  const lm = result.landmarks[0]
  const w = canvas.width
  const h = canvas.height

  const nose          = lm[0]
  const leftShoulder  = lm[11]
  const rightShoulder = lm[12]
  const leftHip       = lm[23]
  const rightHip      = lm[24]
  const leftAnkle     = lm[27]
  const rightAnkle    = lm[28]

  console.log('[AeroMaxx] Landmark coords (normalized):', {
    nose:           { x: nose.x, y: nose.y },
    leftShoulder:   { x: leftShoulder.x, y: leftShoulder.y },
    rightShoulder:  { x: rightShoulder.x, y: rightShoulder.y },
    leftHip:        { x: leftHip.x, y: leftHip.y },
    rightHip:       { x: rightHip.x, y: rightHip.y },
    leftAnkle:      { x: leftAnkle.x, y: leftAnkle.y },
    rightAnkle:     { x: rightAnkle.x, y: rightAnkle.y },
  })

  const shoulderWidth = dist(leftShoulder, rightShoulder, w, h)
  const hipWidth      = dist(leftHip, rightHip, w, h)

  const ankleMidX = ((leftAnkle.x + rightAnkle.x) / 2) * w
  const ankleMidY = ((leftAnkle.y + rightAnkle.y) / 2) * h
  const noseX = nose.x * w
  const noseY = nose.y * h
  const bodyHeight = Math.sqrt(
    (noseX - ankleMidX) ** 2 + (noseY - ankleMidY) ** 2
  )

  const shoulderMidX = ((leftShoulder.x + rightShoulder.x) / 2) * w
  const hipMidX      = ((leftHip.x + rightHip.x) / 2) * w
  const maxOffset = shoulderWidth * 0.3
  const rawHunch = Math.abs(shoulderMidX - hipMidX)
  const hunchScore = Math.min(rawHunch / maxOffset, 1)

  const shoulderToHipRatio = shoulderWidth / hipWidth

  // 5′9″ = 1.7526 m — assumed standard adult height for pixel-to-metre scaling
  const ASSUMED_HEIGHT_M = 1.7526
  const pixelsPerMeter = bodyHeight / ASSUMED_HEIGHT_M
  const realShoulderWidth = shoulderWidth / pixelsPerMeter
  const realHipWidth      = hipWidth / pixelsPerMeter

  console.log('[AeroMaxx] Pixel measurements:', {
    shoulderWidth, hipWidth, bodyHeight, hunchScore, shoulderToHipRatio,
  })
  console.log('[AeroMaxx] Real-world measurements (m):', {
    realShoulderWidth, realHipWidth, pixelsPerMeter,
  })

  const landmarks: Landmarks = {
    nose:           { x: nose.x,           y: nose.y },
    leftShoulder:   { x: leftShoulder.x,   y: leftShoulder.y },
    rightShoulder:  { x: rightShoulder.x,  y: rightShoulder.y },
    leftHip:        { x: leftHip.x,        y: leftHip.y },
    rightHip:       { x: rightHip.x,       y: rightHip.y },
    leftAnkle:      { x: leftAnkle.x,      y: leftAnkle.y },
    rightAnkle:     { x: rightAnkle.x,     y: rightAnkle.y },
  }

  const measurements: BodyMeasurements = {
    shoulderWidth,
    hipWidth,
    bodyHeight,
    pixelsPerMeter,
    realShoulderWidth,
    realHipWidth,
    shoulderToHipRatio,
    hunchScore,
  }

  const annotatedCanvas = drawAnnotatedCanvas(canvas, lm)

  return { landmarks, measurements, annotatedCanvas }
}
