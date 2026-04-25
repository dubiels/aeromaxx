import type { BodyMeasurements, DragResults, GlbMeasurements } from './types'

const REF_FRONTAL_AREA  = 0.60
const REF_CD            = 0.80
const REF_DEPTH_WIDTH   = 0.50

export async function getRecommendations(
  m: BodyMeasurements,
  d: DragResults,
  glb?: GlbMeasurements | null,
): Promise<string> {

  const areaVsRef  = ((d.frontalArea - REF_FRONTAL_AREA) / REF_FRONTAL_AREA * 100).toFixed(1)
  const cdVsRef    = ((d.Cd - REF_CD) / REF_CD * 100).toFixed(1)
  const areaSource = glb ? '3D photogrammetric reconstruction' : '2D landmark estimation'

  const glbSection = glb ? `
3D GEOMETRY:
  Body width: ${glb.realWidth.toFixed(3)} m
  Body depth: ${glb.realDepth.toFixed(3)} m
  Depth-to-width ratio: ${glb.depthToWidthRatio.toFixed(3)} (reference: ${REF_DEPTH_WIDTH})
  Frontal area (3D): ${glb.frontalArea.toFixed(4)} m²` : ''

  const userMsg = `SUBJECT AERODYNAMIC PROFILE

DRAG EQUATION RESULTS:
  Cd: ${d.Cd.toFixed(4)} [${Number(cdVsRef) >= 0 ? '+' : ''}${cdVsRef}% vs ref]
  Frontal area: ${d.frontalArea.toFixed(4)} m² [${Number(areaVsRef) >= 0 ? '+' : ''}${areaVsRef}% vs ref] — ${areaSource}
  Drag force: ${d.dragForce.toFixed(3)} N
${glbSection}
POSTURAL ANALYSIS:
  Hunch score: ${m.hunchScore.toFixed(3)}
  Postural Cd penalty: +${d.posturalPenalty.toFixed(4)}
  Shoulder-to-hip ratio: ${m.shoulderToHipRatio.toFixed(3)}

LIFETIME IMPACT:
  Total drag work: ${d.lifetimeEnergy.toLocaleString('en-US', { maximumFractionDigits: 0 })} J
  Big Mac equivalents: ${d.bigMacs.toFixed(1)}
  Days lost: ${d.daysLost.toFixed(1)}`

  const response = await fetch(
    'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions',
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${import.meta.env.VITE_GEMINI_API_KEY}`,
      },
      body: JSON.stringify({
        model: 'gemma-4-26b-a4b-it',
        messages: [
          {
            role: 'system',
            content: `You are a biomechanical aerodynamics analyst. You have received a quantified aerodynamic profile for a human subject.

THE PHYSICS:
Drag reduction comes from two levers:
1. FRONTAL AREA (A) — reducing A by 10% reduces drag by exactly 10%.
2. DRAG COEFFICIENT (Cd) — shape and posture factor.
   - Thoracic kyphosis correction: 3–8% Cd reduction
   - Shoulder retraction: 2–5% Cd reduction
   - Core stabilisation: 3–6% Cd improvement
   - Clothing fit: 5–15% Cd difference
   - Depth-to-width ratio 0.45–0.55: 5–10% lower Cd

Write a DRAG REDUCTION PROTOCOL — exactly 4 numbered interventions ranked by expected drag reduction.
For each:
Line 1: Name and mechanism
Line 2: Quantified target
Line 3: Weekly protocol
Line 4: Expected drag reduction %

Tone: clinical, precise, no hedges.`,
          },
          { role: 'user', content: userMsg },
        ],
        temperature: 0.6,
        max_tokens: 900,
      }),
    }
  )

  const data = await response.json()
  return data.choices[0].message.content ?? ''
}