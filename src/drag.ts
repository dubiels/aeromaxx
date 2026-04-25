import type { BodyMeasurements, DragResults } from './types'

export function calculateDrag(m: BodyMeasurements): DragResults {
  const rho = 1.225   // air density kg/m³
  const v = 1.4       // avg walking speed m/s

  const A = m.realShoulderWidth * 1.75 * 0.73

  let Cd = 0.80
  let posturalPenalty = 0

  posturalPenalty += m.hunchScore * 0.08
  if (m.shoulderToHipRatio > 1.3) posturalPenalty += 0.03
  if (m.shoulderToHipRatio < 0.9) posturalPenalty += 0.04

  Cd += posturalPenalty

  const dragForce = 0.5 * rho * v * v * Cd * A

  const distancePerDay = v * 4 * 3600
  const lifetimeDistance = distancePerDay * 365 * 75
  const lifetimeEnergy = dragForce * lifetimeDistance

  const calories = lifetimeEnergy / 4184
  const bigMacs = lifetimeEnergy / 2300000
  const daysLost = (lifetimeEnergy / 80) / 86400

  console.log('[AeroMaxx] Drag results:', { Cd, frontalArea: A, dragForce, lifetimeEnergy, calories, bigMacs, daysLost })

  return { Cd, frontalArea: A, dragForce, lifetimeEnergy, calories, bigMacs, daysLost, posturalPenalty }
}
