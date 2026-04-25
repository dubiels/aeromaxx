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

export async function extractLandmarks(
  canvas: HTMLCanvasElement
): Promise<{ landmarks: Landmarks; measurements: BodyMeasurements }> {
  await initPoseLandmarker()
  if (!landmarker) throw new Error('Landmarker not initialized')

  const result = landmarker.detect(canvas)
  if (!result.landmarks || result.landmarks.length === 0) {
    throw new Error('No pose detected. Please upload a clear full-body photo.')
  }

  const lm = result.landmarks[0]
  const w = canvas.width
  const h = canvas.height

  // Named landmarks by index
  const nose        = lm[0]
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

  // Pixel measurements
  const shoulderWidth = dist(leftShoulder, rightShoulder, w, h)
  const hipWidth      = dist(leftHip, rightHip, w, h)

  const ankleMidX = ((leftAnkle.x + rightAnkle.x) / 2) * w
  const ankleMidY = ((leftAnkle.y + rightAnkle.y) / 2) * h
  const noseX = nose.x * w
  const noseY = nose.y * h
  const bodyHeight = Math.sqrt(
    (noseX - ankleMidX) ** 2 + (noseY - ankleMidY) ** 2
  )

  // Shoulder midpoint vs hip midpoint horizontal offset → hunch
  const shoulderMidX = ((leftShoulder.x + rightShoulder.x) / 2) * w
  const hipMidX      = ((leftHip.x + rightHip.x) / 2) * w
  const maxOffset = shoulderWidth * 0.3  // normalise against shoulder width
  const rawHunch = Math.abs(shoulderMidX - hipMidX)
  const hunchScore = Math.min(rawHunch / maxOffset, 1)

  const shoulderToHipRatio = shoulderWidth / hipWidth

  // Scale to real-world meters (assumed 1.75m average height)
  const pixelsPerMeter = bodyHeight / 1.75
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

  return { landmarks, measurements }
}
