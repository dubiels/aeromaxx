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

// Measurements extracted from the Meshy GLB bounding box.
// More accurate than 2D photo estimates because they capture actual 3D body volume.
export interface GlbMeasurements {
  realWidth: number           // m, shoulder-to-shoulder from 3D model
  realDepth: number           // m, front-to-back body depth from 3D model
  frontalArea: number         // m², realWidth × 1.75 × 0.73
  depthToWidthRatio: number   // dimensionless streamlining factor
}

export interface AnalysisState {
  landmarks: Landmarks | null
  measurements: BodyMeasurements | null
  glbMeasurements: GlbMeasurements | null
  drag: DragResults | null
  recommendations: string | null
  visualRecommendations: string | null
  annotatedImageUrl: string | null
  loading: boolean
  loadingMessage: string
  error: string | null
}
