import type { BodyMeasurements, DragResults, GlbMeasurements } from './types'

// Standard aerodynamic constants
const RHO = 1.225   // kg/m³ — air density, sea level 15 °C (ISA standard)
const V   = 1.4     // m/s  — average adult walking speed (ACSM literature: 1.3–1.5)

// Energy constants for the "lifetime impact" stats
const WALK_HOURS_PER_DAY = 4
const LIFESPAN_YEARS     = 75
const JOULES_PER_KCAL    = 4184
const JOULES_PER_BIGMAC  = 2_300_000   // 550 kcal × 4184 ≈ 2.3 MJ
const METABOLIC_POWER_W  = 80          // avg resting metabolic rate used to convert energy→time

export function calculateDrag(
  m: BodyMeasurements,
  glb?: GlbMeasurements | null,
): DragResults {
  // ── Frontal area (A) ─────────────────────────────────────────────────────
  // If we have a real 3D model, use its actual projected area.
  // Otherwise fall back to the 2D shoulder-width estimate.
  //   Fill factor 0.73: a standing person fills ~73 % of their bounding rectangle.
  //   Source: Kyle & Burke 1984, typical range 0.68–0.78.
  const A = glb
    ? glb.frontalArea
    : m.realShoulderWidth * 1.75 * 0.73

  // ── Base drag coefficient (Cd) ────────────────────────────────────────────
  // Published range for a standing adult: 0.7–1.3.
  // 0.80 is the consensus mid-range value (Hoerner 1965, Shanebrook & Jaszczak 1976).
  let Cd = 0.80

  if (glb) {
    // Depth-to-width ratio correction:
    // The human body is most aerodynamically efficient when the front-to-back
    // depth is roughly 45–55 % of the shoulder width (r ≈ 0.5).
    // A very flat body (r < 0.35) behaves like a bluff plate → higher Cd.
    // An unusually deep body (r > 0.70) has extra separated wake → modest Cd rise.
    // These adjustments are small (≤ 0.06) relative to posture, which dominates.
    const r = glb.depthToWidthRatio
    if (r < 0.35) Cd += (0.35 - r) * 0.40   // flat-body penalty: up to ~+0.06
    if (r > 0.70) Cd += (r - 0.70) * 0.20   // deep-body penalty: smaller effect
  }

  // ── Postural penalties ────────────────────────────────────────────────────
  // Kyphotic hunch increases effective frontal area and disrupts attached flow.
  // +0.08 Cd at maximum hunch is conservative; published studies show 5–15 %.
  let posturalPenalty = m.hunchScore * 0.08

  // Shoulder-to-hip ratio: deviations from ~1.1 indicate asymmetric load paths
  // that correlate with lateral sway, adding ~3–4 % to time-averaged drag.
  if (m.shoulderToHipRatio > 1.3) posturalPenalty += 0.03
  if (m.shoulderToHipRatio < 0.9) posturalPenalty += 0.04

  Cd += posturalPenalty

  // ── Drag force and lifetime energy ───────────────────────────────────────
  const dragForce = 0.5 * RHO * V * V * Cd * A

  const distancePerDay   = V * WALK_HOURS_PER_DAY * 3600        // metres/day
  const lifetimeDistance = distancePerDay * 365 * LIFESPAN_YEARS // metres
  const lifetimeEnergy   = dragForce * lifetimeDistance           // joules (W = F·d)

  const calories  = lifetimeEnergy / JOULES_PER_KCAL
  const bigMacs   = lifetimeEnergy / JOULES_PER_BIGMAC
  const daysLost  = (lifetimeEnergy / METABOLIC_POWER_W) / 86400

  return { Cd, frontalArea: A, dragForce, lifetimeEnergy, calories, bigMacs, daysLost, posturalPenalty }
}
