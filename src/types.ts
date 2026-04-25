export interface Landmarks {
  nose: { x: number; y: number }
  leftShoulder: { x: number; y: number }
  rightShoulder: { x: number; y: number }
  leftHip: { x: number; y: number }
  rightHip: { x: number; y: number }
  leftAnkle: { x: number; y: number }
  rightAnkle: { x: number; y: number }
}

export interface BodyMeasurements {
  shoulderWidth: number       // pixels
  hipWidth: number            // pixels
  bodyHeight: number          // pixels
  pixelsPerMeter: number
  realShoulderWidth: number   // meters
  realHipWidth: number        // meters
  shoulderToHipRatio: number
  hunchScore: number          // 0–1
}

export interface DragResults {
  Cd: number
  frontalArea: number         // m²
  dragForce: number           // N
  lifetimeEnergy: number      // J
  calories: number
  bigMacs: number
  daysLost: number
  posturalPenalty: number     // Cd units added from posture
}

export interface AnalysisState {
  landmarks: Landmarks | null
  measurements: BodyMeasurements | null
  drag: DragResults | null
  recommendations: string | null
  visualRecommendations: string | null
  loading: boolean
  loadingMessage: string
  error: string | null
}
